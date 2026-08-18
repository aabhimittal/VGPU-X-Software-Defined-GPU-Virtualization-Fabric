//! The message vocabulary: one request per `GpuNode` operation, one
//! response family, and lossless codecs for both — including the full
//! `VgpuError` taxonomy, so a device error crosses the wire with every
//! field intact. (This is the promise `types.rs` made in milestone 0:
//! "a single closed set keeps the wire mapping honest." Here the wire
//! arrives and the mapping is written — `error_roundtrips_losslessly`
//! holds it to exhaustiveness.)
//!
//! Layout of every payload: `[VERSION u8][tag u8][fields…]`. Tags are
//! stable protocol surface: append new ones, never renumber.

use vgpu_core::cmd::{ChannelExport, Command};
use vgpu_core::isa::Instr;
use vgpu_core::metrics::{NodeMetrics, TenantMetrics};
use vgpu_core::node::{FaultRecord, TickReport};
use vgpu_core::sched::QosLimits;
use vgpu_core::types::{AccessKind, ChannelId, GpuVirtAddr, VgpuError, VgpuId};
use vgpu_core::vgpu::{VgpuProfile, VgpuState};

use crate::wire::{Dec, Enc, WireError, VERSION};

/// Client → daemon. Mirrors `GpuNode`'s public surface 1:1 — the daemon
/// adds no operations of its own, so the security argument from the core
/// (guest VAs only, budgets enforced per vGPU) transfers to the network
/// boundary unchanged.
#[derive(Debug, Clone, PartialEq)]
pub enum Request {
    /// Admit a vGPU under a profile.
    CreateVgpu(VgpuProfile),
    /// Created → Running.
    StartVgpu(VgpuId),
    /// Running → Suspended.
    SuspendVgpu(VgpuId),
    /// Suspended → Running.
    ResumeVgpu(VgpuId),
    /// Tear down and release everything.
    DestroyVgpu(VgpuId),
    /// Allocate device memory; replies with the guest VA.
    AllocMemory {
        /// Owning vGPU.
        vgpu: VgpuId,
        /// Bytes requested.
        bytes: u64,
    },
    /// Free an allocation by base VA.
    FreeMemory {
        /// Owning vGPU.
        vgpu: VgpuId,
        /// Base VA returned by AllocMemory.
        base: GpuVirtAddr,
    },
    /// Create a command channel.
    CreateChannel(VgpuId),
    /// The doorbell: enqueue one command.
    Submit {
        /// Owning vGPU.
        vgpu: VgpuId,
        /// Target channel.
        channel: ChannelId,
        /// The command (guest VAs only).
        command: Command,
    },
    /// Poll a channel's completed fence value.
    FenceValue {
        /// Owning vGPU.
        vgpu: VgpuId,
        /// Channel to poll.
        channel: ChannelId,
    },
    /// Host→device DMA.
    DmaWrite {
        /// Owning vGPU.
        vgpu: VgpuId,
        /// Destination guest VA.
        dst: GpuVirtAddr,
        /// Payload.
        data: Vec<u8>,
    },
    /// Device→host DMA; replies with the bytes.
    DmaRead {
        /// Owning vGPU.
        vgpu: VgpuId,
        /// Source guest VA.
        src: GpuVirtAddr,
        /// Bytes to read.
        len: u64,
    },
    /// Query lifecycle state.
    VgpuState(VgpuId),
    /// Drive the GPU for up to `budget` cycles (manual-tick mode; with
    /// auto-tick the daemon also calls this on its own timer).
    Tick {
        /// Cycle budget for this tick.
        budget: u64,
    },
    /// Describe the node (card name, VRAM, clock).
    NodeInfo,
    /// Migration: fetch the profile a vGPU was admitted under.
    GetProfile(VgpuId),
    /// Migration: list live allocations `(base VA, bytes)` in creation
    /// order, for deterministic replay on the destination.
    ListAllocations(VgpuId),
    /// Migration: harvest and clear the dirty-page set (page-aligned
    /// guest VAs). The pre-copy loop's read-and-reset primitive.
    TakeDirty(VgpuId),
    /// Migration: export channel state (requires the vGPU be Suspended).
    ExportChannels(VgpuId),
    /// Migration: import channel state into a fresh, unstarted vGPU.
    ImportChannels {
        /// Destination vGPU (Created, no channels yet).
        vgpu: VgpuId,
        /// The exported channels, in order.
        channels: Vec<ChannelExport>,
    },
    /// Telemetry: a full per-tenant + per-node metrics snapshot.
    GetMetrics,
    /// Migration: allocate at a specific guest VA (replay preserves the
    /// source heap's exact shape, holes included).
    AllocMemoryAt {
        /// Owning vGPU.
        vgpu: VgpuId,
        /// Required page-aligned base VA.
        base: GpuVirtAddr,
        /// Bytes requested.
        bytes: u64,
    },
}

/// Daemon → client.
#[derive(Debug, Clone, PartialEq)]
pub enum Response {
    /// CreateVgpu succeeded.
    VgpuCreated(VgpuId),
    /// A fire-and-forget operation succeeded.
    Done,
    /// AllocMemory succeeded.
    Memory(GpuVirtAddr),
    /// CreateChannel succeeded.
    ChannelCreated(ChannelId),
    /// FenceValue reply.
    Fence(u64),
    /// DmaRead reply.
    Data(Vec<u8>),
    /// VgpuState reply.
    State(VgpuState),
    /// Tick reply: what the mediation loop did.
    Ticked(TickSummary),
    /// NodeInfo reply.
    NodeInfo(NodeInfo),
    /// GetProfile reply.
    Profile(VgpuProfile),
    /// ListAllocations reply: `(base VA, bytes)` in creation order.
    Allocations(Vec<(GpuVirtAddr, u64)>),
    /// TakeDirty reply: page-aligned guest VAs.
    DirtyPages(Vec<GpuVirtAddr>),
    /// ExportChannels reply.
    Channels(Vec<ChannelExport>),
    /// GetMetrics reply.
    Metrics(NodeMetrics),
    /// The device model refused the operation. Full fidelity: the client
    /// re-raises exactly the `VgpuError` the core produced.
    Error(VgpuError),
}

/// `TickReport`, flattened for the wire (`FaultRecord` in the core is not
/// `Clone`; the summary owns its data).
#[derive(Debug, Clone, PartialEq)]
pub struct TickSummary {
    /// Cycles consumed.
    pub cycles: u64,
    /// Commands completed.
    pub commands: u64,
    /// Channels killed this tick.
    pub faults: Vec<FaultSummary>,
}

/// One channel-killing fault, wire form.
#[derive(Debug, Clone, PartialEq)]
pub struct FaultSummary {
    /// Offending vGPU.
    pub vgpu: VgpuId,
    /// Killed channel.
    pub channel: ChannelId,
    /// The fault.
    pub error: VgpuError,
}

impl From<&TickReport> for TickSummary {
    fn from(r: &TickReport) -> Self {
        Self {
            cycles: r.cycles,
            commands: r.commands,
            faults: r
                .faults
                .iter()
                .map(|f: &FaultRecord| FaultSummary {
                    vgpu: f.vgpu,
                    channel: f.channel,
                    error: f.error.clone(),
                })
                .collect(),
        }
    }
}

/// Node description for inventory/telemetry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeInfo {
    /// Card name.
    pub name: String,
    /// Total VRAM bytes.
    pub vram_bytes: u64,
    /// VRAM not yet committed to profiles.
    pub uncommitted_vram: u64,
    /// Logical clock.
    pub clock: u64,
}

// ---------------------------------------------------------------------------
// Codecs
// ---------------------------------------------------------------------------

pub(crate) fn enc_profile(e: &mut Enc, p: &VgpuProfile) {
    e.str(&p.name);
    e.u64(p.vram_bytes);
    e.u32(p.compute_weight);
    e.u32(p.max_channels);
    e.u64(p.ring_slots as u64);
    // Optional QoS limits: 0 encodes "unset", since a share of 0% is
    // rejected by profile validation anyway.
    e.u32(p.qos.max_share_pct.unwrap_or(0));
    e.u32(p.qos.min_share_pct.unwrap_or(0));
}

fn opt_pct(v: u32) -> Option<u32> {
    (v > 0).then_some(v)
}

pub(crate) fn dec_profile(d: &mut Dec) -> Result<VgpuProfile, WireError> {
    Ok(VgpuProfile {
        name: d.str()?,
        vram_bytes: d.u64()?,
        compute_weight: d.u32()?,
        max_channels: d.u32()?,
        ring_slots: d.u64()? as usize,
        qos: QosLimits {
            max_share_pct: opt_pct(d.u32()?),
            min_share_pct: opt_pct(d.u32()?),
        },
    })
}

fn enc_command(e: &mut Enc, c: &Command) {
    match c {
        Command::MemFill { dst, len, value } => {
            e.u8(1);
            e.u64(dst.0);
            e.u64(*len);
            e.u8(*value);
        }
        Command::MemCopy { src, dst, len } => {
            e.u8(2);
            e.u64(src.0);
            e.u64(dst.0);
            e.u64(*len);
        }
        Command::KernelLaunch {
            name,
            threads,
            args,
            program,
        } => {
            e.u8(3);
            e.str(name);
            e.u32(*threads);
            e.u32(args.len() as u32);
            for a in args {
                e.u64(*a);
            }
            e.u32(program.len() as u32);
            for i in program {
                enc_instr(e, i);
            }
        }
        Command::FenceSignal { value } => {
            e.u8(4);
            e.u64(*value);
        }
    }
}

/// Instruction codec. One tag byte per opcode; register operands are the
/// raw `u8` indices (the *daemon-side* `isa::validate` at submit is the
/// authority on their validity — the wire only preserves them).
fn enc_instr(e: &mut Enc, i: &Instr) {
    match i {
        Instr::Imm { dst, value } => {
            e.u8(1);
            e.u8(*dst);
            e.u64(*value);
        }
        Instr::Mov { dst, src } => {
            e.u8(2);
            e.u8(*dst);
            e.u8(*src);
        }
        Instr::Add { dst, a, b } => {
            e.u8(3);
            e.u8(*dst);
            e.u8(*a);
            e.u8(*b);
        }
        Instr::Sub { dst, a, b } => {
            e.u8(4);
            e.u8(*dst);
            e.u8(*a);
            e.u8(*b);
        }
        Instr::Mul { dst, a, b } => {
            e.u8(5);
            e.u8(*dst);
            e.u8(*a);
            e.u8(*b);
        }
        Instr::Ld { dst, addr, offset } => {
            e.u8(6);
            e.u8(*dst);
            e.u8(*addr);
            e.u64(*offset);
        }
        Instr::St { src, addr, offset } => {
            e.u8(7);
            e.u8(*src);
            e.u8(*addr);
            e.u64(*offset);
        }
        Instr::Bnz { cond, target } => {
            e.u8(8);
            e.u8(*cond);
            e.u32(*target as u32);
        }
        Instr::Halt => e.u8(9),
    }
}

fn dec_instr(d: &mut Dec) -> Result<Instr, WireError> {
    Ok(match d.u8()? {
        1 => Instr::Imm {
            dst: d.u8()?,
            value: d.u64()?,
        },
        2 => Instr::Mov {
            dst: d.u8()?,
            src: d.u8()?,
        },
        3 => Instr::Add {
            dst: d.u8()?,
            a: d.u8()?,
            b: d.u8()?,
        },
        4 => Instr::Sub {
            dst: d.u8()?,
            a: d.u8()?,
            b: d.u8()?,
        },
        5 => Instr::Mul {
            dst: d.u8()?,
            a: d.u8()?,
            b: d.u8()?,
        },
        6 => Instr::Ld {
            dst: d.u8()?,
            addr: d.u8()?,
            offset: d.u64()?,
        },
        7 => Instr::St {
            src: d.u8()?,
            addr: d.u8()?,
            offset: d.u64()?,
        },
        8 => Instr::Bnz {
            cond: d.u8()?,
            target: d.u32()? as u16,
        },
        9 => Instr::Halt,
        tag => {
            return Err(WireError::BadTag {
                context: "Instr",
                tag,
            })
        }
    })
}

fn dec_command(d: &mut Dec) -> Result<Command, WireError> {
    Ok(match d.u8()? {
        1 => Command::MemFill {
            dst: GpuVirtAddr(d.u64()?),
            len: d.u64()?,
            value: d.u8()?,
        },
        2 => Command::MemCopy {
            src: GpuVirtAddr(d.u64()?),
            dst: GpuVirtAddr(d.u64()?),
            len: d.u64()?,
        },
        3 => {
            let name = d.str()?;
            let threads = d.u32()?;
            let n_args = d.u32()?;
            let mut args = Vec::with_capacity(n_args as usize);
            for _ in 0..n_args {
                args.push(d.u64()?);
            }
            let n_instr = d.u32()?;
            let mut program = Vec::with_capacity(n_instr as usize);
            for _ in 0..n_instr {
                program.push(dec_instr(d)?);
            }
            Command::KernelLaunch {
                name,
                threads,
                args,
                program,
            }
        }
        4 => Command::FenceSignal { value: d.u64()? },
        tag => {
            return Err(WireError::BadTag {
                context: "Command",
                tag,
            })
        }
    })
}

pub(crate) fn enc_channel_export(e: &mut Enc, ch: &ChannelExport) {
    e.u32(ch.pending.len() as u32);
    for cmd in &ch.pending {
        enc_command(e, cmd);
    }
    e.u64(ch.completed_fence);
    e.u64(ch.submitted_fence);
    e.bool(ch.faulted);
}

pub(crate) fn dec_channel_export(d: &mut Dec) -> Result<ChannelExport, WireError> {
    let n = d.u32()?;
    let mut pending = Vec::with_capacity(n as usize);
    for _ in 0..n {
        pending.push(dec_command(d)?);
    }
    Ok(ChannelExport {
        pending,
        completed_fence: d.u64()?,
        submitted_fence: d.u64()?,
        faulted: d.bool()?,
    })
}

fn enc_metrics(e: &mut Enc, m: &NodeMetrics) {
    e.str(&m.name);
    e.u64(m.clock);
    e.u64(m.busy_cycles);
    e.u64(m.capped_idle_cycles);
    e.u64(m.vram_bytes);
    e.u64(m.uncommitted_vram);
    e.u64(m.faults);
    e.u32(m.tenants.len() as u32);
    for t in &m.tenants {
        e.u32(t.vgpu.0);
        e.str(&t.profile_name);
        enc_state(e, t.state);
        e.u64(t.cycles_consumed);
        e.u64(t.commands_completed);
        e.u64(t.faults);
        e.u64(t.bytes_dma_in);
        e.u64(t.bytes_dma_out);
        e.u64(t.kernel_launches);
        e.u64(t.vram_used);
        e.u64(t.vram_budget);
        e.u64(t.queued_commands);
        e.u32(t.channels);
        e.u64(t.window_consumed);
        e.bool(t.capped_out);
    }
}

fn dec_metrics(d: &mut Dec) -> Result<NodeMetrics, WireError> {
    let name = d.str()?;
    let clock = d.u64()?;
    let busy_cycles = d.u64()?;
    let capped_idle_cycles = d.u64()?;
    let vram_bytes = d.u64()?;
    let uncommitted_vram = d.u64()?;
    let faults = d.u64()?;
    let n = d.u32()?;
    let mut tenants = Vec::with_capacity(n as usize);
    for _ in 0..n {
        tenants.push(TenantMetrics {
            vgpu: VgpuId(d.u32()?),
            profile_name: d.str()?,
            state: dec_state(d)?,
            cycles_consumed: d.u64()?,
            commands_completed: d.u64()?,
            faults: d.u64()?,
            bytes_dma_in: d.u64()?,
            bytes_dma_out: d.u64()?,
            kernel_launches: d.u64()?,
            vram_used: d.u64()?,
            vram_budget: d.u64()?,
            queued_commands: d.u64()?,
            channels: d.u32()?,
            window_consumed: d.u64()?,
            capped_out: d.bool()?,
        });
    }
    Ok(NodeMetrics {
        name,
        clock,
        busy_cycles,
        capped_idle_cycles,
        vram_bytes,
        uncommitted_vram,
        faults,
        tenants,
    })
}

fn enc_access(e: &mut Enc, a: AccessKind) {
    e.u8(match a {
        AccessKind::Read => 0,
        AccessKind::Write => 1,
    });
}

fn dec_access(d: &mut Dec) -> Result<AccessKind, WireError> {
    match d.u8()? {
        0 => Ok(AccessKind::Read),
        1 => Ok(AccessKind::Write),
        tag => Err(WireError::BadTag {
            context: "AccessKind",
            tag,
        }),
    }
}

fn enc_error(e: &mut Enc, err: &VgpuError) {
    match err {
        VgpuError::OutOfVram {
            requested_frames,
            free_frames,
        } => {
            e.u8(1);
            e.u64(*requested_frames);
            e.u64(*free_frames);
        }
        VgpuError::VramBudgetExceeded {
            requested,
            budget_left,
        } => {
            e.u8(2);
            e.u64(*requested);
            e.u64(*budget_left);
        }
        VgpuError::PageFault { addr, access } => {
            e.u8(3);
            e.u64(addr.0);
            enc_access(e, *access);
        }
        VgpuError::AlreadyMapped { addr } => {
            e.u8(4);
            e.u64(addr.0);
        }
        VgpuError::NotMapped { addr } => {
            e.u8(5);
            e.u64(addr.0);
        }
        VgpuError::BadAddress { addr, why } => {
            e.u8(6);
            e.u64(addr.0);
            e.str(why);
        }
        VgpuError::RingFull => e.u8(7),
        VgpuError::InvalidState { actual, wanted } => {
            e.u8(8);
            e.str(actual);
            e.str(wanted);
        }
        VgpuError::NoSuchVgpu(id) => {
            e.u8(9);
            e.u32(id.0);
        }
        VgpuError::NoSuchChannel(id) => {
            e.u8(10);
            e.u32(id.0);
        }
        VgpuError::ProfileUnsatisfiable { why } => {
            e.u8(11);
            e.str(why);
        }
        VgpuError::ChannelFaulted(id) => {
            e.u8(12);
            e.u32(id.0);
        }
        VgpuError::BadProgram { why } => {
            e.u8(13);
            e.str(why);
        }
        VgpuError::KernelTimeout { executed } => {
            e.u8(14);
            e.u64(*executed);
        }
        VgpuError::TransferTooLarge { requested, limit } => {
            e.u8(15);
            e.u64(*requested);
            e.u64(*limit);
        }
        VgpuError::FenceRegression { last, attempted } => {
            e.u8(16);
            e.u64(*last);
            e.u64(*attempted);
        }
    }
}

fn dec_error(d: &mut Dec) -> Result<VgpuError, WireError> {
    Ok(match d.u8()? {
        1 => VgpuError::OutOfVram {
            requested_frames: d.u64()?,
            free_frames: d.u64()?,
        },
        2 => VgpuError::VramBudgetExceeded {
            requested: d.u64()?,
            budget_left: d.u64()?,
        },
        3 => VgpuError::PageFault {
            addr: GpuVirtAddr(d.u64()?),
            access: dec_access(d)?,
        },
        4 => VgpuError::AlreadyMapped {
            addr: GpuVirtAddr(d.u64()?),
        },
        5 => VgpuError::NotMapped {
            addr: GpuVirtAddr(d.u64()?),
        },
        6 => VgpuError::BadAddress {
            addr: GpuVirtAddr(d.u64()?),
            why: d.str()?,
        },
        7 => VgpuError::RingFull,
        8 => VgpuError::InvalidState {
            actual: d.str()?,
            wanted: d.str()?,
        },
        9 => VgpuError::NoSuchVgpu(VgpuId(d.u32()?)),
        10 => VgpuError::NoSuchChannel(ChannelId(d.u32()?)),
        11 => VgpuError::ProfileUnsatisfiable { why: d.str()? },
        12 => VgpuError::ChannelFaulted(ChannelId(d.u32()?)),
        13 => VgpuError::BadProgram { why: d.str()? },
        14 => VgpuError::KernelTimeout { executed: d.u64()? },
        15 => VgpuError::TransferTooLarge {
            requested: d.u64()?,
            limit: d.u64()?,
        },
        16 => VgpuError::FenceRegression {
            last: d.u64()?,
            attempted: d.u64()?,
        },
        tag => {
            return Err(WireError::BadTag {
                context: "VgpuError",
                tag,
            })
        }
    })
}

fn enc_state(e: &mut Enc, s: VgpuState) {
    e.u8(match s {
        VgpuState::Created => 0,
        VgpuState::Running => 1,
        VgpuState::Suspended => 2,
        VgpuState::Destroyed => 3,
    });
}

fn dec_state(d: &mut Dec) -> Result<VgpuState, WireError> {
    Ok(match d.u8()? {
        0 => VgpuState::Created,
        1 => VgpuState::Running,
        2 => VgpuState::Suspended,
        3 => VgpuState::Destroyed,
        tag => {
            return Err(WireError::BadTag {
                context: "VgpuState",
                tag,
            })
        }
    })
}

/// Check the leading version byte.
fn check_version(d: &mut Dec) -> Result<(), WireError> {
    let theirs = d.u8()?;
    if theirs != VERSION {
        return Err(WireError::VersionMismatch {
            ours: VERSION,
            theirs,
        });
    }
    Ok(())
}

impl Request {
    /// Encode to a frame body.
    pub fn encode(&self) -> Vec<u8> {
        let mut e = Enc::new();
        e.u8(VERSION);
        match self {
            Request::CreateVgpu(p) => {
                e.u8(1);
                enc_profile(&mut e, p);
            }
            Request::StartVgpu(id) => {
                e.u8(2);
                e.u32(id.0);
            }
            Request::SuspendVgpu(id) => {
                e.u8(3);
                e.u32(id.0);
            }
            Request::ResumeVgpu(id) => {
                e.u8(4);
                e.u32(id.0);
            }
            Request::DestroyVgpu(id) => {
                e.u8(5);
                e.u32(id.0);
            }
            Request::AllocMemory { vgpu, bytes } => {
                e.u8(6);
                e.u32(vgpu.0);
                e.u64(*bytes);
            }
            Request::FreeMemory { vgpu, base } => {
                e.u8(7);
                e.u32(vgpu.0);
                e.u64(base.0);
            }
            Request::CreateChannel(id) => {
                e.u8(8);
                e.u32(id.0);
            }
            Request::Submit {
                vgpu,
                channel,
                command,
            } => {
                e.u8(9);
                e.u32(vgpu.0);
                e.u32(channel.0);
                enc_command(&mut e, command);
            }
            Request::FenceValue { vgpu, channel } => {
                e.u8(10);
                e.u32(vgpu.0);
                e.u32(channel.0);
            }
            Request::DmaWrite { vgpu, dst, data } => {
                e.u8(11);
                e.u32(vgpu.0);
                e.u64(dst.0);
                e.bytes(data);
            }
            Request::DmaRead { vgpu, src, len } => {
                e.u8(12);
                e.u32(vgpu.0);
                e.u64(src.0);
                e.u64(*len);
            }
            Request::VgpuState(id) => {
                e.u8(13);
                e.u32(id.0);
            }
            Request::Tick { budget } => {
                e.u8(14);
                e.u64(*budget);
            }
            Request::NodeInfo => e.u8(15),
            Request::GetProfile(id) => {
                e.u8(16);
                e.u32(id.0);
            }
            Request::ListAllocations(id) => {
                e.u8(17);
                e.u32(id.0);
            }
            Request::TakeDirty(id) => {
                e.u8(18);
                e.u32(id.0);
            }
            Request::ExportChannels(id) => {
                e.u8(19);
                e.u32(id.0);
            }
            Request::ImportChannels { vgpu, channels } => {
                e.u8(20);
                e.u32(vgpu.0);
                e.u32(channels.len() as u32);
                for ch in channels {
                    enc_channel_export(&mut e, ch);
                }
            }
            Request::GetMetrics => e.u8(22),
            Request::AllocMemoryAt { vgpu, base, bytes } => {
                e.u8(21);
                e.u32(vgpu.0);
                e.u64(base.0);
                e.u64(*bytes);
            }
        }
        e.into_bytes()
    }

    /// Decode from a frame body (total: bad bytes → `WireError`).
    pub fn decode(buf: &[u8]) -> Result<Self, WireError> {
        let mut d = Dec::new(buf);
        check_version(&mut d)?;
        let req = match d.u8()? {
            1 => Request::CreateVgpu(dec_profile(&mut d)?),
            2 => Request::StartVgpu(VgpuId(d.u32()?)),
            3 => Request::SuspendVgpu(VgpuId(d.u32()?)),
            4 => Request::ResumeVgpu(VgpuId(d.u32()?)),
            5 => Request::DestroyVgpu(VgpuId(d.u32()?)),
            6 => Request::AllocMemory {
                vgpu: VgpuId(d.u32()?),
                bytes: d.u64()?,
            },
            7 => Request::FreeMemory {
                vgpu: VgpuId(d.u32()?),
                base: GpuVirtAddr(d.u64()?),
            },
            8 => Request::CreateChannel(VgpuId(d.u32()?)),
            9 => Request::Submit {
                vgpu: VgpuId(d.u32()?),
                channel: ChannelId(d.u32()?),
                command: dec_command(&mut d)?,
            },
            10 => Request::FenceValue {
                vgpu: VgpuId(d.u32()?),
                channel: ChannelId(d.u32()?),
            },
            11 => Request::DmaWrite {
                vgpu: VgpuId(d.u32()?),
                dst: GpuVirtAddr(d.u64()?),
                data: d.bytes()?,
            },
            12 => Request::DmaRead {
                vgpu: VgpuId(d.u32()?),
                src: GpuVirtAddr(d.u64()?),
                len: d.u64()?,
            },
            13 => Request::VgpuState(VgpuId(d.u32()?)),
            14 => Request::Tick { budget: d.u64()? },
            15 => Request::NodeInfo,
            16 => Request::GetProfile(VgpuId(d.u32()?)),
            17 => Request::ListAllocations(VgpuId(d.u32()?)),
            18 => Request::TakeDirty(VgpuId(d.u32()?)),
            19 => Request::ExportChannels(VgpuId(d.u32()?)),
            20 => {
                let vgpu = VgpuId(d.u32()?);
                let n = d.u32()?;
                let mut channels = Vec::with_capacity(n as usize);
                for _ in 0..n {
                    channels.push(dec_channel_export(&mut d)?);
                }
                Request::ImportChannels { vgpu, channels }
            }
            22 => Request::GetMetrics,
            21 => Request::AllocMemoryAt {
                vgpu: VgpuId(d.u32()?),
                base: GpuVirtAddr(d.u64()?),
                bytes: d.u64()?,
            },
            tag => {
                return Err(WireError::BadTag {
                    context: "Request",
                    tag,
                })
            }
        };
        d.finish()?;
        Ok(req)
    }
}

impl Response {
    /// Encode to a frame body.
    pub fn encode(&self) -> Vec<u8> {
        let mut e = Enc::new();
        e.u8(VERSION);
        match self {
            Response::VgpuCreated(id) => {
                e.u8(1);
                e.u32(id.0);
            }
            Response::Done => e.u8(2),
            Response::Memory(va) => {
                e.u8(3);
                e.u64(va.0);
            }
            Response::ChannelCreated(ch) => {
                e.u8(4);
                e.u32(ch.0);
            }
            Response::Fence(v) => {
                e.u8(5);
                e.u64(*v);
            }
            Response::Data(bytes) => {
                e.u8(6);
                e.bytes(bytes);
            }
            Response::State(s) => {
                e.u8(7);
                enc_state(&mut e, *s);
            }
            Response::Ticked(t) => {
                e.u8(8);
                e.u64(t.cycles);
                e.u64(t.commands);
                e.u32(t.faults.len() as u32);
                for f in &t.faults {
                    e.u32(f.vgpu.0);
                    e.u32(f.channel.0);
                    enc_error(&mut e, &f.error);
                }
            }
            Response::NodeInfo(n) => {
                e.u8(9);
                e.str(&n.name);
                e.u64(n.vram_bytes);
                e.u64(n.uncommitted_vram);
                e.u64(n.clock);
            }
            Response::Error(err) => {
                e.u8(10);
                enc_error(&mut e, err);
            }
            Response::Profile(p) => {
                e.u8(11);
                enc_profile(&mut e, p);
            }
            Response::Allocations(list) => {
                e.u8(12);
                e.u32(list.len() as u32);
                for (base, bytes) in list {
                    e.u64(base.0);
                    e.u64(*bytes);
                }
            }
            Response::DirtyPages(pages) => {
                e.u8(13);
                e.u32(pages.len() as u32);
                for p in pages {
                    e.u64(p.0);
                }
            }
            Response::Channels(chans) => {
                e.u8(14);
                e.u32(chans.len() as u32);
                for ch in chans {
                    enc_channel_export(&mut e, ch);
                }
            }
            Response::Metrics(m) => {
                e.u8(15);
                enc_metrics(&mut e, m);
            }
        }
        e.into_bytes()
    }

    /// Decode from a frame body.
    pub fn decode(buf: &[u8]) -> Result<Self, WireError> {
        let mut d = Dec::new(buf);
        check_version(&mut d)?;
        let resp = match d.u8()? {
            1 => Response::VgpuCreated(VgpuId(d.u32()?)),
            2 => Response::Done,
            3 => Response::Memory(GpuVirtAddr(d.u64()?)),
            4 => Response::ChannelCreated(ChannelId(d.u32()?)),
            5 => Response::Fence(d.u64()?),
            6 => Response::Data(d.bytes()?),
            7 => Response::State(dec_state(&mut d)?),
            8 => {
                let cycles = d.u64()?;
                let commands = d.u64()?;
                let n = d.u32()?;
                let mut faults = Vec::with_capacity(n as usize);
                for _ in 0..n {
                    faults.push(FaultSummary {
                        vgpu: VgpuId(d.u32()?),
                        channel: ChannelId(d.u32()?),
                        error: dec_error(&mut d)?,
                    });
                }
                Response::Ticked(TickSummary {
                    cycles,
                    commands,
                    faults,
                })
            }
            9 => Response::NodeInfo(NodeInfo {
                name: d.str()?,
                vram_bytes: d.u64()?,
                uncommitted_vram: d.u64()?,
                clock: d.u64()?,
            }),
            10 => Response::Error(dec_error(&mut d)?),
            11 => Response::Profile(dec_profile(&mut d)?),
            12 => {
                let n = d.u32()?;
                let mut list = Vec::with_capacity(n as usize);
                for _ in 0..n {
                    list.push((GpuVirtAddr(d.u64()?), d.u64()?));
                }
                Response::Allocations(list)
            }
            13 => {
                let n = d.u32()?;
                let mut pages = Vec::with_capacity(n as usize);
                for _ in 0..n {
                    pages.push(GpuVirtAddr(d.u64()?));
                }
                Response::DirtyPages(pages)
            }
            14 => {
                let n = d.u32()?;
                let mut chans = Vec::with_capacity(n as usize);
                for _ in 0..n {
                    chans.push(dec_channel_export(&mut d)?);
                }
                Response::Channels(chans)
            }
            15 => Response::Metrics(dec_metrics(&mut d)?),
            tag => {
                return Err(WireError::BadTag {
                    context: "Response",
                    tag,
                })
            }
        };
        d.finish()?;
        Ok(resp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip_req(r: Request) {
        let bytes = r.encode();
        assert_eq!(Request::decode(&bytes).unwrap(), r);
    }

    fn roundtrip_resp(r: Response) {
        let bytes = r.encode();
        assert_eq!(Response::decode(&bytes).unwrap(), r);
    }

    #[test]
    fn every_request_roundtrips() {
        roundtrip_req(Request::CreateVgpu(VgpuProfile {
            name: "sim-2g.25c".into(),
            vram_bytes: 1 << 30,
            compute_weight: 3,
            max_channels: 8,
            ring_slots: 256,
            qos: QosLimits::default(),
        }));
        roundtrip_req(Request::StartVgpu(VgpuId(7)));
        roundtrip_req(Request::SuspendVgpu(VgpuId(7)));
        roundtrip_req(Request::ResumeVgpu(VgpuId(7)));
        roundtrip_req(Request::DestroyVgpu(VgpuId(7)));
        roundtrip_req(Request::AllocMemory {
            vgpu: VgpuId(1),
            bytes: 4096,
        });
        roundtrip_req(Request::FreeMemory {
            vgpu: VgpuId(1),
            base: GpuVirtAddr(0x0400_0000),
        });
        roundtrip_req(Request::CreateChannel(VgpuId(1)));
        for command in [
            Command::MemFill {
                dst: GpuVirtAddr(64),
                len: 128,
                value: 0xAB,
            },
            Command::MemCopy {
                src: GpuVirtAddr(0),
                dst: GpuVirtAddr(64),
                len: 32,
            },
            // One launch exercising every instruction kind in the codec.
            Command::KernelLaunch {
                name: "gemm".into(),
                threads: 64,
                args: vec![0x0400_0000, 0x0500_0000, 0x0600_0000],
                program: vec![
                    Instr::Imm { dst: 4, value: 8 },
                    Instr::Mov { dst: 5, src: 0 },
                    Instr::Add { dst: 5, a: 5, b: 4 },
                    Instr::Sub { dst: 6, a: 5, b: 4 },
                    Instr::Mul { dst: 5, a: 0, b: 4 },
                    Instr::Ld {
                        dst: 7,
                        addr: 1,
                        offset: 16,
                    },
                    Instr::St {
                        src: 7,
                        addr: 3,
                        offset: 24,
                    },
                    Instr::Bnz { cond: 6, target: 2 },
                    Instr::Halt,
                ],
            },
            Command::FenceSignal { value: 9 },
        ] {
            roundtrip_req(Request::Submit {
                vgpu: VgpuId(1),
                channel: ChannelId(0),
                command,
            });
        }
        roundtrip_req(Request::FenceValue {
            vgpu: VgpuId(1),
            channel: ChannelId(0),
        });
        roundtrip_req(Request::DmaWrite {
            vgpu: VgpuId(1),
            dst: GpuVirtAddr(0x0400_0000),
            data: vec![1, 2, 3],
        });
        roundtrip_req(Request::DmaRead {
            vgpu: VgpuId(1),
            src: GpuVirtAddr(0x0400_0000),
            len: 3,
        });
        roundtrip_req(Request::VgpuState(VgpuId(1)));
        roundtrip_req(Request::Tick { budget: 10_000 });
        roundtrip_req(Request::NodeInfo);
        roundtrip_req(Request::GetProfile(VgpuId(2)));
        roundtrip_req(Request::ListAllocations(VgpuId(2)));
        roundtrip_req(Request::TakeDirty(VgpuId(2)));
        roundtrip_req(Request::ExportChannels(VgpuId(2)));
        roundtrip_req(Request::ImportChannels {
            vgpu: VgpuId(2),
            channels: vec![ChannelExport {
                pending: vec![
                    Command::MemFill {
                        dst: GpuVirtAddr(0x0400_0000),
                        len: 64,
                        value: 3,
                    },
                    Command::FenceSignal { value: 5 },
                ],
                completed_fence: 4,
                submitted_fence: 5,
                faulted: false,
            }],
        });
        roundtrip_req(Request::AllocMemoryAt {
            vgpu: VgpuId(2),
            base: GpuVirtAddr(0x0400_0000),
            bytes: 1 << 20,
        });
    }

    #[test]
    fn every_response_roundtrips() {
        roundtrip_resp(Response::VgpuCreated(VgpuId(3)));
        roundtrip_resp(Response::Done);
        roundtrip_resp(Response::Memory(GpuVirtAddr(0x0400_0000)));
        roundtrip_resp(Response::ChannelCreated(ChannelId(2)));
        roundtrip_resp(Response::Fence(41));
        roundtrip_resp(Response::Data(vec![9; 100]));
        for s in [
            VgpuState::Created,
            VgpuState::Running,
            VgpuState::Suspended,
            VgpuState::Destroyed,
        ] {
            roundtrip_resp(Response::State(s));
        }
        roundtrip_resp(Response::Ticked(TickSummary {
            cycles: 1000,
            commands: 7,
            faults: vec![FaultSummary {
                vgpu: VgpuId(0),
                channel: ChannelId(1),
                error: VgpuError::PageFault {
                    addr: GpuVirtAddr(0xdead_0000),
                    access: AccessKind::Write,
                },
            }],
        }));
        roundtrip_resp(Response::NodeInfo(NodeInfo {
            name: "sim-a".into(),
            vram_bytes: 1 << 33,
            uncommitted_vram: 1 << 32,
            clock: 123456,
        }));
        roundtrip_resp(Response::Profile(VgpuProfile {
            name: "mig".into(),
            vram_bytes: 1 << 30,
            compute_weight: 2,
            max_channels: 4,
            ring_slots: 128,
            // QoS must survive the wire, or a migrated tenant would land
            // on its new node with its contract silently erased.
            qos: QosLimits {
                max_share_pct: Some(25),
                min_share_pct: Some(10),
            },
        }));
        roundtrip_resp(Response::Allocations(vec![
            (GpuVirtAddr(0x0400_0000), 1 << 20),
            (GpuVirtAddr(0x0500_0000), 1 << 16),
        ]));
        roundtrip_resp(Response::DirtyPages(vec![
            GpuVirtAddr(0x0400_0000),
            GpuVirtAddr(0x0401_0000),
        ]));
        roundtrip_resp(Response::Channels(vec![ChannelExport {
            pending: vec![Command::FenceSignal { value: 9 }],
            completed_fence: 8,
            submitted_fence: 8,
            faulted: true,
        }]));
    }

    /// Exhaustive over the error enum: adding a `VgpuError` variant
    /// without a codec arm fails to compile (match exhaustiveness in
    /// `enc_error`), and this test pins the decode side byte-for-byte.
    #[test]
    fn error_roundtrips_losslessly() {
        let all = [
            VgpuError::OutOfVram {
                requested_frames: 8,
                free_frames: 3,
            },
            VgpuError::VramBudgetExceeded {
                requested: 4096,
                budget_left: 0,
            },
            VgpuError::PageFault {
                addr: GpuVirtAddr(0x1_0000),
                access: AccessKind::Read,
            },
            VgpuError::AlreadyMapped {
                addr: GpuVirtAddr(0x2_0000),
            },
            VgpuError::NotMapped {
                addr: GpuVirtAddr(0x3_0000),
            },
            VgpuError::BadAddress {
                addr: GpuVirtAddr(0),
                why: "zero-byte allocation".into(),
            },
            VgpuError::RingFull,
            VgpuError::InvalidState {
                actual: "Suspended".into(),
                wanted: "submit".into(),
            },
            VgpuError::NoSuchVgpu(VgpuId(9)),
            VgpuError::NoSuchChannel(ChannelId(9)),
            VgpuError::ProfileUnsatisfiable {
                why: "channel limit reached".into(),
            },
            VgpuError::ChannelFaulted(ChannelId(4)),
            VgpuError::BadProgram {
                why: "branch target 9 outside program at pc 3".into(),
            },
            VgpuError::KernelTimeout {
                executed: 1_000_000,
            },
        ];
        for err in all {
            roundtrip_resp(Response::Error(err));
        }
    }

    #[test]
    fn version_mismatch_is_refused() {
        let mut bytes = Request::NodeInfo.encode();
        bytes[0] = 99;
        assert_eq!(
            Request::decode(&bytes),
            Err(WireError::VersionMismatch {
                ours: VERSION,
                theirs: 99
            })
        );
    }

    #[test]
    fn unknown_tag_is_refused() {
        let bytes = vec![VERSION, 0xEE];
        assert!(matches!(
            Request::decode(&bytes),
            Err(WireError::BadTag {
                context: "Request",
                ..
            })
        ));
    }
}
