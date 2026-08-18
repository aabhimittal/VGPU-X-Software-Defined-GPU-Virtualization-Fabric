//! The blocking client: a typed façade over one socket.
//!
//! Milestone 2's guest shim links against this — when an intercepted
//! `cudaMalloc` needs device memory, it becomes `client.alloc_memory(..)`
//! here. The client is deliberately dumb: one in-flight request per
//! connection, strict request→response alternation, no pipelining. Real
//! drivers pipeline aggressively; we buy a protocol whose correctness is
//! checkable by eye first, and the framing already supports pipelining
//! when a milestone needs it (frames are self-delimiting).

use std::io;
use std::net::{TcpStream, ToSocketAddrs};

use vgpu_core::cmd::{ChannelExport, Command};
use vgpu_core::metrics::NodeMetrics;
use vgpu_core::types::{ChannelId, GpuVirtAddr, VgpuError, VgpuId};
use vgpu_core::vgpu::{VgpuProfile, VgpuState};

use crate::msg::{NodeInfo, Request, Response, TickSummary};
use crate::wire::{read_frame, write_frame, WireError};

/// Everything a client call can fail with, in three distinct layers —
/// keeping them separate matters because the right reaction differs:
/// retry/reconnect for transport, bug report for protocol, and normal
/// error handling for device errors.
#[derive(Debug)]
pub enum ClientError {
    /// Transport failed (socket died, daemon gone).
    Io(io::Error),
    /// Bytes arrived but didn't decode, or the response variant didn't
    /// match the request — a version skew or a bug, never normal.
    Protocol(WireError),
    /// The daemon answered with a shape that doesn't answer our request.
    UnexpectedResponse(&'static str),
    /// The device model refused the operation (the same `VgpuError` a
    /// local `GpuNode` call would have returned — full fidelity).
    Device(VgpuError),
}

impl std::fmt::Display for ClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "transport error: {e}"),
            Self::Protocol(e) => write!(f, "protocol error: {e}"),
            Self::UnexpectedResponse(what) => write!(f, "unexpected response to {what}"),
            Self::Device(e) => write!(f, "device error: {e}"),
        }
    }
}

impl std::error::Error for ClientError {}

impl From<io::Error> for ClientError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<WireError> for ClientError {
    fn from(e: WireError) -> Self {
        Self::Protocol(e)
    }
}

/// Client-side result alias.
pub type ClientResult<T> = Result<T, ClientError>;

/// A connection to one `vgpud`.
pub struct VgpuClient {
    stream: TcpStream,
}

impl VgpuClient {
    /// Connect to a daemon.
    pub fn connect(addr: impl ToSocketAddrs) -> io::Result<Self> {
        let stream = TcpStream::connect(addr)?;
        // A request/response protocol sends tiny frames and waits; Nagle's
        // algorithm would add 40ms stalls for nothing.
        stream.set_nodelay(true)?;
        Ok(Self { stream })
    }

    /// One request→response exchange. `Response::Error` is lifted into
    /// `ClientError::Device` here so every typed method below only has to
    /// match its own success shape.
    fn call(&mut self, req: &Request) -> ClientResult<Response> {
        write_frame(&mut self.stream, &req.encode())?;
        let body = read_frame(&mut self.stream)?.ok_or_else(|| {
            ClientError::Io(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "daemon closed the connection",
            ))
        })?;
        match Response::decode(&body)? {
            Response::Error(e) => Err(ClientError::Device(e)),
            ok => Ok(ok),
        }
    }

    /// Admit a vGPU.
    pub fn create_vgpu(&mut self, profile: VgpuProfile) -> ClientResult<VgpuId> {
        match self.call(&Request::CreateVgpu(profile))? {
            Response::VgpuCreated(id) => Ok(id),
            _ => Err(ClientError::UnexpectedResponse("CreateVgpu")),
        }
    }

    /// Start a created vGPU.
    pub fn start_vgpu(&mut self, id: VgpuId) -> ClientResult<()> {
        self.expect_done(Request::StartVgpu(id), "StartVgpu")
    }

    /// Suspend a running vGPU.
    pub fn suspend_vgpu(&mut self, id: VgpuId) -> ClientResult<()> {
        self.expect_done(Request::SuspendVgpu(id), "SuspendVgpu")
    }

    /// Resume a suspended vGPU.
    pub fn resume_vgpu(&mut self, id: VgpuId) -> ClientResult<()> {
        self.expect_done(Request::ResumeVgpu(id), "ResumeVgpu")
    }

    /// Destroy a vGPU.
    pub fn destroy_vgpu(&mut self, id: VgpuId) -> ClientResult<()> {
        self.expect_done(Request::DestroyVgpu(id), "DestroyVgpu")
    }

    /// Allocate device memory.
    pub fn alloc_memory(&mut self, vgpu: VgpuId, bytes: u64) -> ClientResult<GpuVirtAddr> {
        match self.call(&Request::AllocMemory { vgpu, bytes })? {
            Response::Memory(va) => Ok(va),
            _ => Err(ClientError::UnexpectedResponse("AllocMemory")),
        }
    }

    /// Free device memory by base VA.
    pub fn free_memory(&mut self, vgpu: VgpuId, base: GpuVirtAddr) -> ClientResult<()> {
        self.expect_done(Request::FreeMemory { vgpu, base }, "FreeMemory")
    }

    /// Create a channel.
    pub fn create_channel(&mut self, vgpu: VgpuId) -> ClientResult<ChannelId> {
        match self.call(&Request::CreateChannel(vgpu))? {
            Response::ChannelCreated(ch) => Ok(ch),
            _ => Err(ClientError::UnexpectedResponse("CreateChannel")),
        }
    }

    /// Submit one command (the doorbell).
    pub fn submit(
        &mut self,
        vgpu: VgpuId,
        channel: ChannelId,
        command: Command,
    ) -> ClientResult<()> {
        self.expect_done(
            Request::Submit {
                vgpu,
                channel,
                command,
            },
            "Submit",
        )
    }

    /// Poll a channel's completed fence.
    pub fn fence_value(&mut self, vgpu: VgpuId, channel: ChannelId) -> ClientResult<u64> {
        match self.call(&Request::FenceValue { vgpu, channel })? {
            Response::Fence(v) => Ok(v),
            _ => Err(ClientError::UnexpectedResponse("FenceValue")),
        }
    }

    /// Host→device DMA.
    pub fn dma_write(&mut self, vgpu: VgpuId, dst: GpuVirtAddr, data: &[u8]) -> ClientResult<()> {
        self.expect_done(
            Request::DmaWrite {
                vgpu,
                dst,
                data: data.to_vec(),
            },
            "DmaWrite",
        )
    }

    /// Device→host DMA.
    pub fn dma_read(&mut self, vgpu: VgpuId, src: GpuVirtAddr, len: u64) -> ClientResult<Vec<u8>> {
        match self.call(&Request::DmaRead { vgpu, src, len })? {
            Response::Data(d) => Ok(d),
            _ => Err(ClientError::UnexpectedResponse("DmaRead")),
        }
    }

    /// Query lifecycle state.
    pub fn vgpu_state(&mut self, id: VgpuId) -> ClientResult<VgpuState> {
        match self.call(&Request::VgpuState(id))? {
            Response::State(s) => Ok(s),
            _ => Err(ClientError::UnexpectedResponse("VgpuState")),
        }
    }

    /// Drive the GPU for up to `budget` cycles.
    pub fn tick(&mut self, budget: u64) -> ClientResult<TickSummary> {
        match self.call(&Request::Tick { budget })? {
            Response::Ticked(t) => Ok(t),
            _ => Err(ClientError::UnexpectedResponse("Tick")),
        }
    }

    /// Telemetry: a per-tenant and per-node metrics snapshot.
    pub fn metrics(&mut self) -> ClientResult<NodeMetrics> {
        match self.call(&Request::GetMetrics)? {
            Response::Metrics(m) => Ok(m),
            _ => Err(ClientError::UnexpectedResponse("GetMetrics")),
        }
    }

    /// Migration: read a tenant's QoS window spend.
    pub fn qos_window(&mut self, id: VgpuId) -> ClientResult<u64> {
        match self.call(&Request::GetQosWindow(id))? {
            Response::QosWindow(v) => Ok(v),
            _ => Err(ClientError::UnexpectedResponse("GetQosWindow")),
        }
    }

    /// Migration: carry a QoS window spend onto the destination.
    pub fn adopt_qos_window(&mut self, vgpu: VgpuId, consumed: u64) -> ClientResult<()> {
        self.expect_done(Request::AdoptQosWindow { vgpu, consumed }, "AdoptQosWindow")
    }

    /// Migration: allocate at a specific guest VA (heap-shape replay).
    pub fn alloc_memory_at(
        &mut self,
        vgpu: VgpuId,
        base: GpuVirtAddr,
        bytes: u64,
    ) -> ClientResult<GpuVirtAddr> {
        match self.call(&Request::AllocMemoryAt { vgpu, base, bytes })? {
            Response::Memory(va) => Ok(va),
            _ => Err(ClientError::UnexpectedResponse("AllocMemoryAt")),
        }
    }

    /// Migration: the profile a vGPU was admitted under.
    pub fn vgpu_profile(&mut self, id: VgpuId) -> ClientResult<VgpuProfile> {
        match self.call(&Request::GetProfile(id))? {
            Response::Profile(p) => Ok(p),
            _ => Err(ClientError::UnexpectedResponse("GetProfile")),
        }
    }

    /// Migration: live allocations `(base VA, bytes)` in creation order.
    pub fn list_allocations(&mut self, id: VgpuId) -> ClientResult<Vec<(GpuVirtAddr, u64)>> {
        match self.call(&Request::ListAllocations(id))? {
            Response::Allocations(a) => Ok(a),
            _ => Err(ClientError::UnexpectedResponse("ListAllocations")),
        }
    }

    /// Migration: harvest and clear the dirty-page set.
    pub fn take_dirty(&mut self, id: VgpuId) -> ClientResult<Vec<GpuVirtAddr>> {
        match self.call(&Request::TakeDirty(id))? {
            Response::DirtyPages(p) => Ok(p),
            _ => Err(ClientError::UnexpectedResponse("TakeDirty")),
        }
    }

    /// Migration: export a suspended vGPU's channel state.
    pub fn export_channels(&mut self, id: VgpuId) -> ClientResult<Vec<ChannelExport>> {
        match self.call(&Request::ExportChannels(id))? {
            Response::Channels(c) => Ok(c),
            _ => Err(ClientError::UnexpectedResponse("ExportChannels")),
        }
    }

    /// Migration: import channel state into a fresh vGPU.
    pub fn import_channels(
        &mut self,
        vgpu: VgpuId,
        channels: Vec<ChannelExport>,
    ) -> ClientResult<()> {
        self.expect_done(Request::ImportChannels { vgpu, channels }, "ImportChannels")
    }

    /// Describe the node.
    pub fn node_info(&mut self) -> ClientResult<NodeInfo> {
        match self.call(&Request::NodeInfo)? {
            Response::NodeInfo(n) => Ok(n),
            _ => Err(ClientError::UnexpectedResponse("NodeInfo")),
        }
    }

    fn expect_done(&mut self, req: Request, what: &'static str) -> ClientResult<()> {
        match self.call(&req)? {
            Response::Done => Ok(()),
            _ => Err(ClientError::UnexpectedResponse(what)),
        }
    }
}
