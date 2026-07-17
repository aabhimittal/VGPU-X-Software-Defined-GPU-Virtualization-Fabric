//! Foundational value types shared by every subsystem.
//!
//! Everything here is a *newtype*: a thin, zero-cost wrapper around a
//! primitive integer. The point is not runtime behavior — it is that the
//! compiler refuses to let a guest virtual address flow into a slot that
//! expects a physical frame number. In a device model, mixing up those two
//! integer spaces is the classic catastrophic bug (it is exactly the class
//! of bug an IOMMU exists to catch in hardware), so we make it a *type
//! error* instead of a security incident.

use core::fmt;

// ---------------------------------------------------------------------------
// Identifiers
// ---------------------------------------------------------------------------

/// Identifies one virtual GPU within a node.
///
/// Real mediated-passthrough stacks (NVIDIA vGPU, Intel GVT-g) use a UUID
/// here; we use a dense `u32` because the node allocates them and dense IDs
/// index directly into slab-style tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VgpuId(pub u32);

/// Identifies a command channel (a guest-visible submission queue) within a
/// single vGPU. Analogous to a CUDA stream's underlying hardware channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ChannelId(pub u32);

impl fmt::Display for VgpuId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "vgpu{}", self.0)
    }
}

impl fmt::Display for ChannelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ch{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// Address spaces
// ---------------------------------------------------------------------------

/// An address in a vGPU's *guest virtual* address space.
///
/// Guests only ever see these. They are meaningless outside the owning
/// vGPU's page tables — two vGPUs can hold the same `GpuVirtAddr` value and
/// refer to completely different (or no) physical memory. That property IS
/// the isolation model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GpuVirtAddr(pub u64);

/// A byte address into the physical GPU's VRAM, produced only by the GMMU's
/// translation step. Nothing outside the memory subsystem can fabricate one
/// except by going through translation — the constructor is crate-private
/// by convention (the field is public for tests, but guests never see it).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VramAddr(pub u64);

/// Index of one VRAM frame (a `FRAME_SIZE`-byte aligned block).
/// `VramAddr = FrameNum * FRAME_SIZE + offset`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FrameNum(pub u64);

/// Size of one VRAM frame: 64 KiB.
///
/// Why 64 KiB and not the CPU-style 4 KiB? GPUs prefer "big pages":
/// NVIDIA's GMMU natively supports 64 KiB pages and drivers use them for
/// VRAM because (a) VRAM allocations are large and contiguous-ish, so the
/// finer granularity buys nothing, and (b) bigger pages mean fewer TLB
/// entries per buffer, and GPU TLB misses stall thousands of threads at
/// once, not one.
pub const FRAME_SIZE: u64 = 64 * 1024;

impl GpuVirtAddr {
    /// Byte offset within the containing frame (the low 16 bits for 64 KiB
    /// frames).
    pub fn frame_offset(self) -> u64 {
        self.0 % FRAME_SIZE
    }
}

impl FrameNum {
    /// The physical byte address where this frame starts.
    pub fn base_addr(self) -> VramAddr {
        VramAddr(self.0 * FRAME_SIZE)
    }
}

impl fmt::Display for GpuVirtAddr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "gva:{:#014x}", self.0)
    }
}

impl fmt::Display for VramAddr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "vram:{:#014x}", self.0)
    }
}

// ---------------------------------------------------------------------------
// Time
// ---------------------------------------------------------------------------

/// Logical GPU time, in "cycles".
///
/// The whole device model is a *deterministic simulation*: no wall clock
/// anywhere. Command costs are charged in cycles, the scheduler accounts in
/// cycles, and tests can assert exact fairness ratios because replaying the
/// same inputs yields the same clock. Real hypervisor schedulers are tested
/// exactly this way (deterministic simulation testing) because wall-clock
/// tests of fairness are flaky by construction.
pub type Cycles = u64;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Every way the device model can refuse an operation.
///
/// One flat enum instead of per-module error types: the fabric's control
/// plane (a later milestone) must serialize these across a wire, and a
/// single closed set keeps that mapping honest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VgpuError {
    /// VRAM allocator cannot satisfy the request (capacity or fragmentation).
    OutOfVram {
        /// Frames requested.
        requested_frames: u64,
        /// Frames still free (may be non-contiguous — buddy allocators can
        /// fail with free space remaining; see `vram.rs`).
        free_frames: u64,
    },
    /// The vGPU would exceed the VRAM budget of its profile.
    VramBudgetExceeded {
        /// Bytes requested.
        requested: u64,
        /// Bytes remaining in the profile budget.
        budget_left: u64,
    },
    /// Translation failed: no valid mapping for this guest virtual address.
    /// This is the GPU equivalent of a segfault, and in real hardware it
    /// surfaces as an engine fault interrupt (NVIDIA Xid 31).
    PageFault {
        /// The faulting guest virtual address.
        addr: GpuVirtAddr,
        /// What the access was trying to do.
        access: AccessKind,
    },
    /// A mapping request overlapped an existing valid mapping.
    AlreadyMapped {
        /// First conflicting guest virtual address.
        addr: GpuVirtAddr,
    },
    /// An unmap request named a page that was not mapped.
    NotMapped {
        /// The offending guest virtual address.
        addr: GpuVirtAddr,
    },
    /// Address arithmetic overflowed or violated alignment rules.
    BadAddress {
        /// The offending guest virtual address.
        addr: GpuVirtAddr,
        /// Human-readable constraint that was violated.
        why: String,
    },
    /// The command ring is full; the guest must wait for the device to
    /// drain it (real drivers spin or sleep on exactly this condition).
    RingFull,
    /// Operation is illegal in the vGPU's current lifecycle state.
    InvalidState {
        /// State the vGPU was actually in.
        actual: String,
        /// Operation that was attempted.
        wanted: String,
    },
    /// Referenced vGPU does not exist on this node.
    NoSuchVgpu(VgpuId),
    /// Referenced channel does not exist on this vGPU.
    NoSuchChannel(ChannelId),
    /// The physical GPU cannot host another vGPU of this profile.
    ProfileUnsatisfiable {
        /// Why placement failed.
        why: String,
    },
    /// The channel previously faulted and was killed; it accepts no more
    /// work until torn down (mirrors real channel-error semantics, where
    /// the driver "RC recovers" the channel).
    ChannelFaulted(ChannelId),
}

/// Whether a faulting access was a read or a write — reported in the fault
/// record because it changes both debugging and (later) copy-on-write
/// migration logic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessKind {
    /// Read access.
    Read,
    /// Write access.
    Write,
}

impl fmt::Display for VgpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OutOfVram {
                requested_frames,
                free_frames,
            } => write!(
                f,
                "out of VRAM: requested {requested_frames} frames, {free_frames} free"
            ),
            Self::VramBudgetExceeded {
                requested,
                budget_left,
            } => write!(
                f,
                "vGPU VRAM budget exceeded: requested {requested} B, {budget_left} B left"
            ),
            Self::PageFault { addr, access } => {
                write!(f, "page fault on {access:?} at {addr}")
            }
            Self::AlreadyMapped { addr } => write!(f, "{addr} is already mapped"),
            Self::NotMapped { addr } => write!(f, "{addr} is not mapped"),
            Self::BadAddress { addr, why } => write!(f, "bad address {addr}: {why}"),
            Self::RingFull => write!(f, "command ring is full"),
            Self::InvalidState { actual, wanted } => {
                write!(f, "cannot {wanted} while vGPU is {actual}")
            }
            Self::NoSuchVgpu(id) => write!(f, "no such vGPU: {id}"),
            Self::NoSuchChannel(id) => write!(f, "no such channel: {id}"),
            Self::ProfileUnsatisfiable { why } => write!(f, "profile unsatisfiable: {why}"),
            Self::ChannelFaulted(id) => write!(f, "channel {id} is faulted"),
        }
    }
}

impl std::error::Error for VgpuError {}

/// Crate-wide result alias.
pub type Result<T> = core::result::Result<T, VgpuError>;
