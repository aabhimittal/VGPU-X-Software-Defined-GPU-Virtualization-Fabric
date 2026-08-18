//! # vgpu-shim — the guest runtime (API remoting over the fabric)
//!
//! Milestone 2's second half: the layer a guest *application* links
//! against. It has the shape of the CUDA runtime API — allocate, copy,
//! launch, synchronize on streams — and remotes every call to a `vgpud`
//! node through `VgpuClient`. This is technique 1 from the landscape doc
//! (API remoting, the rCUDA idea) composed on top of technique 3 (the
//! mediated device model): the guest sees a familiar library; the fabric
//! sees doorbell writes.
//!
//! # The translation table
//!
//! | CUDA concept | Here | Backed by |
//! |---|---|---|
//! | context / `cudaSetDevice` | [`Device::connect`] | create + start a vGPU |
//! | `cudaMalloc` / `cudaFree` | [`Device::malloc`] / [`Device::free`] | budgeted VRAM alloc + GMMU map |
//! | `cudaMemcpy` H2D / D2H | [`Device::memcpy_htod`] / [`memcpy_dtoh`](Device::memcpy_dtoh) | host DMA through the page tables |
//! | stream | [`Stream`] | a channel (submission ring) |
//! | `cudaMemsetAsync`, D2D copy | [`Device::memset_async`] / [`memcpy_dtod_async`](Device::memcpy_dtod_async) | commands on the stream's ring |
//! | kernel `<<<grid>>>` launch | [`Device::launch`] | `KernelLaunch` with a real ISA program |
//! | `cudaStreamSynchronize` | [`Device::synchronize`] | the fence protocol (below) |
//!
//! # How synchronize works — the fence protocol, spelled out
//!
//! CUDA's `cudaStreamSynchronize` promises: every operation submitted to
//! the stream *before this call* has completed. The shim implements that
//! promise the way every real driver does:
//!
//! 1. submit `FenceSignal(n)` (n = this stream's next fence number) —
//!    rings are FIFO, so the fence completes only after everything ahead
//!    of it;
//! 2. wait until the channel's completed fence reaches `n`.
//!
//! "Wait" here is *client-driven*: the shim polls the fence and, when it
//! has not landed, asks the node to run (`tick`). Progress never depends
//! on the daemon's optional wall-clock auto-tick, and tests stay
//! deterministic. The poll also detects dead streams: if a tick makes no
//! progress and the fence still hasn't landed, the queue drained without
//! reaching our fence — which, on an in-order ring, can only mean the
//! channel faulted; the shim reports [`ShimError::StreamFaulted`] instead
//! of spinning forever.

use std::net::ToSocketAddrs;

use vgpu_core::cmd::Command;
use vgpu_core::isa::Instr;
use vgpu_core::types::{ChannelId, GpuVirtAddr, VgpuId, MAX_DMA_BYTES};
use vgpu_core::vgpu::VgpuProfile;
use vgpu_proto::{ClientError, VgpuClient};

/// Cycle budget per synchronize poll — big enough to drain typical work
/// in a few round-trips, small enough that one guest's sync cannot hog
/// the node inside a single request.
const SYNC_TICK_BUDGET: u64 = 1_000_000;
/// Poll-round bound for one synchronize. With each round driving up to
/// `SYNC_TICK_BUDGET` cycles this is an enormous amount of work; hitting
/// it means something is wedged, and an error beats a hang.
const SYNC_MAX_ROUNDS: u32 = 10_000;

/// A device pointer, as the guest sees one: an opaque handle that is
/// really a guest VA. Guests can do arithmetic on it via [`DevicePtr::offset`]
/// exactly like real device pointers — and a wild result faults *their*
/// channel and nothing else, which is the whole isolation story.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DevicePtr(pub GpuVirtAddr);

impl DevicePtr {
    /// Pointer arithmetic, guest-side.
    pub fn offset(self, bytes: u64) -> DevicePtr {
        DevicePtr(GpuVirtAddr(self.0 .0.wrapping_add(bytes)))
    }
}

/// A stream: an ordered lane of asynchronous work, backed by one channel.
/// Owns its fence counter; syncs on stream A never wait on stream B's
/// work — the same independence CUDA streams promise.
#[derive(Debug)]
pub struct Stream {
    channel: ChannelId,
    next_fence: u64,
}

/// Shim-level failures.
#[derive(Debug)]
pub enum ShimError {
    /// The underlying RPC failed (transport, protocol, or a typed device
    /// error such as a budget violation on `malloc`).
    Client(ClientError),
    /// The stream's channel died: its queue drained without reaching the
    /// sync fence (a prior command faulted — wild pointer, watchdog…).
    /// The stream is dead; the device and other streams are fine.
    StreamFaulted,
    /// `SYNC_MAX_ROUNDS` polls made progress but never landed the fence —
    /// pathological, and surfaced rather than hung.
    SyncTimeout,
}

impl std::fmt::Display for ShimError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Client(e) => write!(f, "rpc failed: {e}"),
            Self::StreamFaulted => write!(f, "stream faulted: a prior command was killed"),
            Self::SyncTimeout => write!(f, "synchronize exceeded its poll bound"),
        }
    }
}

impl std::error::Error for ShimError {}

impl From<ClientError> for ShimError {
    fn from(e: ClientError) -> Self {
        Self::Client(e)
    }
}

/// Shim result alias.
pub type ShimResult<T> = Result<T, ShimError>;

/// A guest's handle to one vGPU: the moral equivalent of a CUDA context.
pub struct Device {
    client: VgpuClient,
    vgpu: VgpuId,
}

impl Device {
    /// Connect to a node, admit a vGPU under `profile`, and start it —
    /// everything `cudaSetDevice` + context creation does, made explicit.
    pub fn connect(addr: impl ToSocketAddrs, profile: VgpuProfile) -> ShimResult<Self> {
        let mut client = VgpuClient::connect(addr).map_err(|e| ShimError::Client(e.into()))?;
        let vgpu = client.create_vgpu(profile)?;
        client.start_vgpu(vgpu)?;
        Ok(Self { client, vgpu })
    }

    /// `cudaMalloc`: device memory, charged against the profile budget.
    pub fn malloc(&mut self, bytes: u64) -> ShimResult<DevicePtr> {
        Ok(DevicePtr(self.client.alloc_memory(self.vgpu, bytes)?))
    }

    /// `cudaFree`. Must be the exact pointer `malloc` returned.
    pub fn free(&mut self, ptr: DevicePtr) -> ShimResult<()> {
        Ok(self.client.free_memory(self.vgpu, ptr.0)?)
    }

    /// `cudaMemcpyHostToDevice` — synchronous host DMA.
    ///
    /// Transfers larger than the device's per-transfer ceiling are split
    /// automatically. The ceiling exists so the device never sizes a
    /// buffer from an unbounded client-supplied length
    /// (`types::MAX_DMA_BYTES`); chunking here is what keeps that limit
    /// invisible to guests, exactly as real drivers hide DMA-ring
    /// segmentation behind a single `cudaMemcpy`.
    pub fn memcpy_htod(&mut self, dst: DevicePtr, data: &[u8]) -> ShimResult<()> {
        let chunk = MAX_DMA_BYTES as usize;
        for (i, part) in data.chunks(chunk).enumerate() {
            let at = dst.offset((i * chunk) as u64);
            self.client.dma_write(self.vgpu, at.0, part)?;
        }
        Ok(())
    }

    /// `cudaMemcpyDeviceToHost` — synchronous host DMA (chunked, as above).
    pub fn memcpy_dtoh(&mut self, src: DevicePtr, len: u64) -> ShimResult<Vec<u8>> {
        let mut out = Vec::with_capacity(len as usize);
        let mut done = 0u64;
        while done < len {
            let take = MAX_DMA_BYTES.min(len - done);
            out.extend_from_slice(&self.client.dma_read(self.vgpu, src.offset(done).0, take)?);
            done += take;
        }
        Ok(out)
    }

    /// `cudaStreamCreate`: a new channel with an independent fence line.
    pub fn stream_create(&mut self) -> ShimResult<Stream> {
        let channel = self.client.create_channel(self.vgpu)?;
        Ok(Stream {
            channel,
            next_fence: 1,
        })
    }

    /// `cudaMemsetAsync` on a stream.
    pub fn memset_async(
        &mut self,
        stream: &Stream,
        dst: DevicePtr,
        len: u64,
        value: u8,
    ) -> ShimResult<()> {
        self.submit(
            stream,
            Command::MemFill {
                dst: dst.0,
                len,
                value,
            },
        )
    }

    /// Device-to-device copy on a stream.
    pub fn memcpy_dtod_async(
        &mut self,
        stream: &Stream,
        dst: DevicePtr,
        src: DevicePtr,
        len: u64,
    ) -> ShimResult<()> {
        self.submit(
            stream,
            Command::MemCopy {
                src: src.0,
                dst: dst.0,
                len,
            },
        )
    }

    /// The kernel launch: `threads` copies of `program`, args in `r1..`.
    /// Device pointers go in as their raw VAs — exactly how real kernel
    /// parameters carry device pointers.
    pub fn launch(
        &mut self,
        stream: &Stream,
        name: &str,
        threads: u32,
        args: &[u64],
        program: Vec<Instr>,
    ) -> ShimResult<()> {
        self.submit(
            stream,
            Command::KernelLaunch {
                name: name.to_string(),
                threads,
                args: args.to_vec(),
                program,
            },
        )
    }

    /// `cudaStreamSynchronize`: everything submitted to `stream` before
    /// this call has completed when it returns (see module docs for the
    /// fence protocol and dead-stream detection).
    pub fn synchronize(&mut self, stream: &mut Stream) -> ShimResult<()> {
        let target = stream.next_fence;
        stream.next_fence += 1;
        self.submit(stream, Command::FenceSignal { value: target })?;

        for _ in 0..SYNC_MAX_ROUNDS {
            if self.client.fence_value(self.vgpu, stream.channel)? >= target {
                return Ok(());
            }
            let report = self.client.tick(SYNC_TICK_BUDGET)?;
            if report.cycles == 0 {
                // The node had nothing left to run, yet our fence — which
                // sits *behind* any prior work on this in-order ring —
                // never signaled. Only a killed channel does that.
                return if self.client.fence_value(self.vgpu, stream.channel)? >= target {
                    Ok(())
                } else {
                    Err(ShimError::StreamFaulted)
                };
            }
        }
        Err(ShimError::SyncTimeout)
    }

    /// Tear the vGPU down (all VRAM scrubbed and returned). Consumes the
    /// device, so no call can race the teardown.
    pub fn destroy(mut self) -> ShimResult<()> {
        Ok(self.client.destroy_vgpu(self.vgpu)?)
    }

    fn submit(&mut self, stream: &Stream, cmd: Command) -> ShimResult<()> {
        Ok(self.client.submit(self.vgpu, stream.channel, cmd)?)
    }
}
