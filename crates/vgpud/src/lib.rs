//! # vgpud — the VGPU-X node daemon
//!
//! Serves one `GpuNode` to many clients over TCP. This crate is where the
//! project's concurrency story is decided, so the decision is documented
//! at the point of implementation:
//!
//! # The single-owner device thread
//!
//! `GpuNode` is `!Sync` by design — milestone 0 deliberately did not
//! sprinkle locks through the device model. The daemon keeps it that way:
//! **one thread owns the node outright**, and every connection thread
//! sends it messages over a channel. No `Mutex<GpuNode>`, no lock
//! ordering, no poisoning, no convoy.
//!
//! This is not a workaround; it is the architecture that matches the
//! domain. A physical GPU has *one* command front-end — mediation is
//! serialization. In real mediated passthrough, doorbell traps from every
//! guest funnel into one mediator context; our mpsc channel plays exactly
//! that role, and the channel's FIFO order plays the PCIe write-ordering
//! of doorbells. The alternative (`Mutex<GpuNode>`) would express the
//! same serialization *implicitly* while scattering lock-acquisition
//! points across every handler and inviting hold-across-I/O bugs. The
//! ownership thread makes the serialization point a *place you can read*.
//!
//! # Where the wall clock lives
//!
//! The core is a deterministic simulation and stays that way. Wall-clock
//! time enters the system in exactly one spot: the optional auto-tick,
//! where the device thread's `recv_timeout` deadline converts "no
//! requests for N ms" into "run the GPU for a budget of cycles". Tests
//! configure `auto_tick: None` and drive time via explicit `Tick`
//! requests, keeping every integration test as deterministic as the unit
//! tests underneath — the daemon's edge is the only place determinism is
//! traded, and only when asked.

use std::io;
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use vgpu_core::node::{GpuNode, PhysGpuConfig};
use vgpu_proto::msg::{NodeInfo, Request, Response, TickSummary};
use vgpu_proto::wire::{read_frame, write_frame};

/// Daemon configuration.
#[derive(Debug, Clone)]
pub struct DaemonConfig {
    /// The simulated card.
    pub gpu: PhysGpuConfig,
    /// If set, the device thread ticks the GPU with this budget whenever
    /// `interval` elapses without a request. `None` = manual ticks only
    /// (deterministic; what the tests use).
    pub auto_tick: Option<AutoTick>,
}

/// Auto-tick policy: the one wall-clock dial in the system.
#[derive(Debug, Clone)]
pub struct AutoTick {
    /// How long the device thread waits for requests before ticking.
    pub interval: Duration,
    /// Cycle budget per automatic tick.
    pub budget: u64,
}

/// One message from a connection thread to the device thread: the request
/// plus the sender's reply slot. The reply channel is per-request (a
/// rendezvous), which keeps responses impossible to cross-deliver even
/// with many connections in flight.
struct Job {
    request: Request,
    reply: mpsc::Sender<Response>,
}

/// A running daemon. Dropping the handle does NOT stop it; call
/// [`ServerHandle::shutdown`].
pub struct ServerHandle {
    /// The address actually bound (useful with port 0).
    pub addr: SocketAddr,
    stop: Arc<AtomicBool>,
    acceptor: Option<JoinHandle<()>>,
    device: Option<JoinHandle<()>>,
}

/// Bind `addr` and serve `config` until shutdown. Returns once the
/// listener is bound and both threads are running — after this, a client
/// `connect` cannot race the daemon coming up.
pub fn serve(addr: SocketAddr, config: DaemonConfig) -> io::Result<ServerHandle> {
    let listener = TcpListener::bind(addr)?;
    let bound = listener.local_addr()?;
    let stop = Arc::new(AtomicBool::new(false));

    let (job_tx, job_rx) = mpsc::channel::<Job>();

    // The device thread: sole owner of the GpuNode for its whole life.
    let device_cfg = config.clone();
    let device_stop = stop.clone();
    let device = std::thread::Builder::new()
        .name("vgpud-device".into())
        .spawn(move || device_loop(device_cfg, job_rx, device_stop))?;

    // The acceptor: one thread per connection. Thread-per-connection is
    // the right call at this scale (a node serves tens of tenants, not
    // ten thousand), and it keeps every connection's read loop linear.
    let acceptor_stop = stop.clone();
    let acceptor = std::thread::Builder::new()
        .name("vgpud-accept".into())
        .spawn(move || {
            for conn in listener.incoming() {
                if acceptor_stop.load(Ordering::SeqCst) {
                    break;
                }
                match conn {
                    Ok(stream) => {
                        let tx = job_tx.clone();
                        // Connection threads are detached: they exit when
                        // the peer hangs up, and at shutdown the device
                        // thread's channel disconnect unblocks any of
                        // their in-flight calls.
                        let _ = std::thread::Builder::new()
                            .name("vgpud-conn".into())
                            .spawn(move || connection_loop(stream, tx));
                    }
                    Err(_) => continue,
                }
            }
            // job_tx (and its clones in finished connections) dropping is
            // what lets the device thread's recv() disconnect and exit.
        })?;

    Ok(ServerHandle {
        addr: bound,
        stop,
        acceptor: Some(acceptor),
        device: Some(device),
    })
}

impl ServerHandle {
    /// Stop accepting, wind down the device thread, join both.
    pub fn shutdown(mut self) {
        self.stop.store(true, Ordering::SeqCst);
        // The acceptor is blocked in accept(); poke it awake with a
        // throwaway connection so it observes the stop flag.
        let _ = TcpStream::connect(self.addr);
        if let Some(h) = self.acceptor.take() {
            let _ = h.join();
        }
        if let Some(h) = self.device.take() {
            let _ = h.join();
        }
    }
}

/// How often the device thread re-checks the stop flag while idle in
/// manual-tick mode. Purely a shutdown-latency dial; nothing ticks here.
const STOP_POLL: Duration = Duration::from_millis(25);

/// The device thread body: own the node, drain jobs, optionally auto-tick.
///
/// The receive always uses a timeout, even in manual-tick mode: waiting
/// on channel *disconnect* alone would deadlock shutdown, because idle
/// connection threads each hold a live sender clone while blocked reading
/// their sockets. The stop flag, checked on every timeout, is what makes
/// `ServerHandle::shutdown` reliably terminal.
fn device_loop(config: DaemonConfig, jobs: mpsc::Receiver<Job>, stop: Arc<AtomicBool>) {
    let mut node = GpuNode::new(config.gpu);
    loop {
        let wait = config
            .auto_tick
            .as_ref()
            .map_or(STOP_POLL, |auto| auto.interval);
        let job = match jobs.recv_timeout(wait) {
            Ok(j) => j,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if stop.load(Ordering::SeqCst) {
                    return;
                }
                if let Some(auto) = &config.auto_tick {
                    // The one wall-clock entry point (see module docs).
                    node.tick(auto.budget);
                }
                continue;
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => return,
        };
        let response = handle(&mut node, job.request);
        // A dead reply receiver just means the client hung up mid-call;
        // the device carries on serving everyone else.
        let _ = job.reply.send(response);
    }
}

/// Map one request onto the node. Pure dispatch: every arm is a
/// one-to-one translation, so the network surface provably adds no
/// operations beyond `GpuNode`'s own API.
fn handle(node: &mut GpuNode, req: Request) -> Response {
    /// Collapse `Result<T>` + a success-shaping closure into a Response.
    fn map<T>(r: vgpu_core::types::Result<T>, ok: impl FnOnce(T) -> Response) -> Response {
        match r {
            Ok(v) => ok(v),
            Err(e) => Response::Error(e),
        }
    }

    match req {
        Request::CreateVgpu(profile) => map(node.create_vgpu(profile), Response::VgpuCreated),
        Request::StartVgpu(id) => map(node.start_vgpu(id), |()| Response::Done),
        Request::SuspendVgpu(id) => map(node.suspend_vgpu(id), |()| Response::Done),
        Request::ResumeVgpu(id) => map(node.resume_vgpu(id), |()| Response::Done),
        Request::DestroyVgpu(id) => map(node.destroy_vgpu(id), |()| Response::Done),
        Request::AllocMemory { vgpu, bytes } => {
            map(node.alloc_memory(vgpu, bytes), Response::Memory)
        }
        Request::FreeMemory { vgpu, base } => {
            map(node.free_memory(vgpu, base), |()| Response::Done)
        }
        Request::CreateChannel(id) => map(node.create_channel(id), Response::ChannelCreated),
        Request::Submit {
            vgpu,
            channel,
            command,
        } => map(node.submit(vgpu, channel, command), |()| Response::Done),
        Request::FenceValue { vgpu, channel } => {
            map(node.fence_value(vgpu, channel), Response::Fence)
        }
        Request::DmaWrite { vgpu, dst, data } => {
            map(node.dma_write(vgpu, dst, &data), |()| Response::Done)
        }
        Request::DmaRead { vgpu, src, len } => {
            let mut buf = vec![0u8; len as usize];
            map(node.dma_read(vgpu, src, &mut buf), move |()| {
                Response::Data(buf)
            })
        }
        Request::VgpuState(id) => map(node.vgpu_state(id), Response::State),
        Request::Tick { budget } => {
            let report = node.tick(budget);
            Response::Ticked(TickSummary::from(&report))
        }
        Request::NodeInfo => Response::NodeInfo(NodeInfo {
            name: node.config().name.clone(),
            vram_bytes: node.config().vram_bytes,
            uncommitted_vram: node.uncommitted_vram(),
            clock: node.clock(),
        }),
    }
}

/// One connection's read→dispatch→write loop. Malformed frames close the
/// connection (there is no way to resynchronize a byte stream with a peer
/// whose encoder you don't trust); device errors, by contrast, are normal
/// traffic and flow back inside `Response::Error`.
fn connection_loop(mut stream: TcpStream, jobs: mpsc::Sender<Job>) {
    let _ = stream.set_nodelay(true);
    loop {
        let body = match read_frame(&mut stream) {
            Ok(Some(b)) => b,
            Ok(None) | Err(_) => return, // peer closed / transport died
        };
        let request = match Request::decode(&body) {
            Ok(r) => r,
            Err(_) => return, // unparseable peer: hang up
        };
        let (reply_tx, reply_rx) = mpsc::channel();
        if jobs
            .send(Job {
                request,
                reply: reply_tx,
            })
            .is_err()
        {
            return; // device thread gone: daemon is shutting down
        }
        let Ok(response) = reply_rx.recv() else {
            return;
        };
        if write_frame(&mut stream, &response.encode()).is_err() {
            return;
        }
    }
}
