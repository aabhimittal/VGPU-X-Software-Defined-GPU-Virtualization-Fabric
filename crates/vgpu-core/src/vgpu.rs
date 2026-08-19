//! The vGPU: a profile, a lifecycle, an address space, and channels.
//!
//! A vGPU owns no physical resources directly. It holds *receipts*
//! (`FrameRange`s) for VRAM the node's allocator lent it, and its page
//! tables reference those frames — but the allocator and the backing store
//! stay with the node. Every method that touches physical memory therefore
//! takes the allocator/store as parameters: the borrow checker is enforcing
//! the mediation boundary that a hypervisor enforces with privilege levels.
//! A `Vgpu` value simply has no way to reach VRAM behind the node's back.

use std::collections::BTreeMap;

use crate::cmd::{Channel, ChannelState, Command};
use crate::gmmu::{AddressSpace, VA_LIMIT};
use crate::sched::QosLimits;
use crate::types::{ChannelId, GpuVirtAddr, Result, VgpuError, VgpuId, FRAME_SIZE};
use crate::vram::{FrameRange, FrameStore, VramAllocator};

/// A vGPU profile: the resource contract for one tenant.
///
/// This mirrors NVIDIA's vGPU "types" (e.g. `A100-2-10C` = 2/7 of compute,
/// 10 GiB of VRAM): a named, fixed bundle of limits chosen at creation
/// time. Fixed bundles — rather than arbitrary per-resource dials — are
/// what make placement decidable for a fabric scheduler: it can pack
/// profiles onto cards like Tetris pieces instead of solving a knapsack
/// problem per request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VgpuProfile {
    /// Human-readable profile name, e.g. "sim-2g.25c".
    pub name: String,
    /// Hard VRAM budget in bytes (must be a frame multiple).
    pub vram_bytes: u64,
    /// Scheduler weight: this vGPU's share of compute is
    /// `weight / sum(weights of runnable vGPUs)`. See `sched.rs`.
    pub compute_weight: u32,
    /// Maximum concurrent channels.
    pub max_channels: u32,
    /// Slots per channel ring.
    pub ring_slots: usize,
    /// Optional hard ceiling and guaranteed floor on compute share.
    /// Defaults to neither, which is pure proportional share — the
    /// behaviour every profile had before QoS existed.
    pub qos: QosLimits,
}

impl Default for VgpuProfile {
    /// A profile skeleton whose resource fields are deliberately *not*
    /// usable (`validate` rejects a zero VRAM budget). It exists so
    /// callers can write `..Default::default()` and pick up the optional
    /// tail — today just `qos` — without restating it. Optional policy
    /// should cost nothing to ignore.
    fn default() -> Self {
        Self {
            name: String::new(),
            vram_bytes: 0,
            compute_weight: 1,
            max_channels: 1,
            ring_slots: 2,
            qos: QosLimits::default(),
        }
    }
}

impl VgpuProfile {
    /// Validate profile invariants; call at creation so a bad profile
    /// fails loudly at admission, not at first allocation.
    pub fn validate(&self) -> Result<()> {
        if self.vram_bytes == 0 || !self.vram_bytes.is_multiple_of(FRAME_SIZE) {
            return Err(VgpuError::ProfileUnsatisfiable {
                why: "vram_bytes must be a positive multiple of FRAME_SIZE".to_string(),
            });
        }
        if self.vram_bytes > VA_LIMIT {
            return Err(VgpuError::ProfileUnsatisfiable {
                why: "vram_bytes exceeds the 64 GiB per-vGPU VA space".to_string(),
            });
        }
        if self.compute_weight == 0 {
            return Err(VgpuError::ProfileUnsatisfiable {
                why: "compute_weight must be > 0".to_string(),
            });
        }
        if self.max_channels == 0 || self.ring_slots < 2 {
            return Err(VgpuError::ProfileUnsatisfiable {
                why: "need at least 1 channel and 2 ring slots".to_string(),
            });
        }
        for (pct, what) in [
            (self.qos.max_share_pct, "max_share_pct"),
            (self.qos.min_share_pct, "min_share_pct"),
        ] {
            if let Some(p) = pct {
                if p == 0 || p > 100 {
                    return Err(VgpuError::ProfileUnsatisfiable {
                        why: format!("{what} must be in 1..=100, got {p}"),
                    });
                }
            }
        }
        // A floor above the ceiling is unsatisfiable by construction, and
        // silently clamping it would hand the operator a contract the
        // node quietly rewrote.
        if let (Some(min), Some(max)) = (self.qos.min_share_pct, self.qos.max_share_pct) {
            if min > max {
                return Err(VgpuError::ProfileUnsatisfiable {
                    why: format!("min_share_pct {min} exceeds max_share_pct {max}"),
                });
            }
        }
        Ok(())
    }
}

/// Lifecycle states. Transitions are a strict diamond:
///
/// ```text
///   Created ──start──▶ Running ◀─resume── Suspended
///                        │  └────suspend────▲
///                        └──────destroy──────┴──▶ Destroyed
/// ```
///
/// `Suspended` exists now (rather than with the migration milestone that
/// needs it) because retrofitting a state machine is how real device
/// models grow their "impossible state" bugs — the states are the spec,
/// so they arrive with the spec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VgpuState {
    /// Configured but never started; owns no channels yet.
    Created,
    /// Schedulable: accepts submissions, receives time slices.
    Running,
    /// Frozen: memory intact, channels intact, but not schedulable and
    /// not accepting submissions. The precondition for live migration.
    Suspended,
    /// Torn down: all VRAM returned and scrubbed. Terminal.
    Destroyed,
}

impl VgpuState {
    fn name(self) -> &'static str {
        match self {
            VgpuState::Created => "Created",
            VgpuState::Running => "Running",
            VgpuState::Suspended => "Suspended",
            VgpuState::Destroyed => "Destroyed",
        }
    }
}

/// One guest-visible memory allocation: a contiguous VA range backed by
/// one or more physical blocks.
#[derive(Debug)]
struct Allocation {
    /// Pages in the VA range.
    pages: u64,
    /// The physical receipts backing it (binary decomposition, so a
    /// 5-page allocation holds a 4-frame block and a 1-frame block).
    ranges: Vec<FrameRange>,
}

/// A virtual GPU instance.
pub struct Vgpu {
    /// Node-assigned identity.
    pub id: VgpuId,
    /// The resource contract this instance was admitted under.
    pub profile: VgpuProfile,
    state: VgpuState,
    /// This tenant's private address space.
    pub(crate) aspace: AddressSpace,
    /// Live allocations keyed by base VA.
    allocations: BTreeMap<u64, Allocation>,
    /// Bytes currently allocated against the profile budget.
    vram_used: u64,
    /// Bump pointer for the guest heap. VA 0 is left unmapped on purpose:
    /// a null guest pointer must fault, not silently read allocation #1.
    next_va: u64,
    /// Channels, indexed by `ChannelId.0`. Slots are never reused within
    /// a vGPU's lifetime so a stale ChannelId can never alias a new channel.
    pub(crate) channels: Vec<Channel>,
}

/// Guest heap starts at 64 MiB, mirroring real drivers which reserve low
/// VA for system structures (and making the null-page guard obvious).
const HEAP_BASE: u64 = 64 * 1024 * 1024;

impl Vgpu {
    /// Create a vGPU under `profile`. No physical resources are consumed
    /// until the first allocation — admission control (does the card have
    /// room for this profile *at all*) is the node's job.
    pub fn new(id: VgpuId, profile: VgpuProfile) -> Result<Self> {
        profile.validate()?;
        Ok(Self {
            id,
            profile,
            state: VgpuState::Created,
            aspace: AddressSpace::new(),
            allocations: BTreeMap::new(),
            vram_used: 0,
            next_va: HEAP_BASE,
            channels: Vec::new(),
        })
    }

    /// Current lifecycle state.
    pub fn state(&self) -> VgpuState {
        self.state
    }

    /// Bytes allocated against the profile budget.
    pub fn vram_used(&self) -> u64 {
        self.vram_used
    }

    /// Commands sitting in this tenant's rings right now (telemetry).
    /// Sustained depth means the tenant is limited by the GPU; zero means
    /// it is limited by itself, and the two want opposite responses.
    pub fn queued_commands(&self) -> u64 {
        self.channels.iter().map(|c| c.ring.len() as u64).sum()
    }

    /// Live channel count (telemetry).
    pub fn channel_count(&self) -> u32 {
        self.channels.len() as u32
    }

    fn expect_state(&self, wanted_op: &'static str, ok: &[VgpuState]) -> Result<()> {
        if ok.contains(&self.state) {
            Ok(())
        } else {
            Err(VgpuError::InvalidState {
                actual: self.state.name().to_string(),
                wanted: wanted_op.to_string(),
            })
        }
    }

    /// Created → Running.
    pub fn start(&mut self) -> Result<()> {
        self.expect_state("start", &[VgpuState::Created])?;
        self.state = VgpuState::Running;
        Ok(())
    }

    /// Running → Suspended.
    pub fn suspend(&mut self) -> Result<()> {
        self.expect_state("suspend", &[VgpuState::Running])?;
        self.state = VgpuState::Suspended;
        Ok(())
    }

    /// Suspended → Running.
    pub fn resume(&mut self) -> Result<()> {
        self.expect_state("resume", &[VgpuState::Suspended])?;
        self.state = VgpuState::Running;
        Ok(())
    }

    /// Allocate `bytes` of device memory; returns the guest VA.
    ///
    /// Order of operations is the whole story here:
    /// 1. **Budget check first** — the tenant-facing limit, independent of
    ///    physical availability.
    /// 2. **Physical allocation by binary decomposition** — a 5-page
    ///    request becomes a 4-frame + a 1-frame buddy block instead of a
    ///    rounded-up 8-frame block. The GMMU makes them one contiguous VA
    ///    range anyway, so buddy round-up waste is paid per *power-of-two
    ///    chunk*, not per allocation. This is exactly why paging plus a
    ///    buddy allocator compose so well.
    /// 3. **Rollback on partial failure** — if any chunk fails, every
    ///    chunk already taken is returned before the error propagates.
    ///    The allocator's state must be identical to before the call.
    /// 4. **Map last** — page tables only ever point at frames we own.
    pub fn alloc_memory(&mut self, vram: &mut VramAllocator, bytes: u64) -> Result<GpuVirtAddr> {
        let base = GpuVirtAddr(self.next_va);
        self.alloc_memory_at(vram, base, bytes)
    }

    /// Allocate `bytes` at a *specific* page-aligned guest VA.
    ///
    /// This is the migration replay primitive. A source heap that has
    /// seen frees is a bump heap with holes — replaying "allocate N
    /// bytes" in order would compact those holes and shift every later
    /// VA, silently invalidating the guest's live pointers. Replaying
    /// "allocate N bytes *at* VA" reproduces the exact shape, holes
    /// included. Overlap with an existing mapping fails atomically
    /// (`AlreadyMapped` from the GMMU's two-pass map, after rollback).
    pub fn alloc_memory_at(
        &mut self,
        vram: &mut VramAllocator,
        base: GpuVirtAddr,
        bytes: u64,
    ) -> Result<GpuVirtAddr> {
        self.expect_state("alloc_memory", &[VgpuState::Running, VgpuState::Created])?;
        if bytes == 0 {
            return Err(VgpuError::BadAddress {
                addr: GpuVirtAddr(0),
                why: "zero-byte allocation".to_string(),
            });
        }
        if !base.0.is_multiple_of(FRAME_SIZE) {
            return Err(VgpuError::BadAddress {
                addr: base,
                why: "allocation base not page-aligned".to_string(),
            });
        }
        let pages = bytes.div_ceil(FRAME_SIZE);
        let charged = pages * FRAME_SIZE; // budget is charged in whole frames
        if self.vram_used + charged > self.profile.vram_bytes {
            return Err(VgpuError::VramBudgetExceeded {
                requested: charged,
                budget_left: self.profile.vram_bytes - self.vram_used,
            });
        }

        // Binary decomposition: one buddy block per set bit of `pages`,
        // largest first so VA layout is deterministic.
        let mut ranges: Vec<FrameRange> = Vec::new();
        let mut remaining = pages;
        while remaining > 0 {
            let chunk = 1u64 << (63 - remaining.leading_zeros()); // highest set bit
            match vram.alloc(chunk) {
                Ok(r) => ranges.push(r),
                Err(e) => {
                    for r in ranges {
                        vram.free(r); // rollback: leave no orphaned frames
                    }
                    return Err(e);
                }
            }
            remaining -= chunk;
        }

        // Map the scattered physical blocks as one contiguous VA range.
        // On overlap (`AlreadyMapped`), roll the frames back — the map
        // itself is two-pass atomic, so nothing was half-installed.
        let frames = ranges.iter().flat_map(|r| r.frames()).collect::<Vec<_>>();
        if let Err(e) = self.aspace.map(base, frames.into_iter(), true) {
            for r in ranges {
                vram.free(r);
            }
            return Err(e);
        }

        // The bump pointer only ever moves forward, past any explicitly
        // placed allocation, so future implicit allocations never collide.
        self.next_va = self.next_va.max(base.0 + pages * FRAME_SIZE);
        self.vram_used += charged;
        self.allocations
            .insert(base.0, Allocation { pages, ranges });
        Ok(base)
    }

    /// Free the allocation whose base VA is `base` (exact-match, like
    /// `free()`): unmap, scrub every frame, return blocks to the allocator.
    ///
    /// Scrubbing before returning frames is the multi-tenant hygiene rule:
    /// the allocator may hand these frames to a different vGPU next, and
    /// VRAM is not zeroed by hardware on reallocation.
    pub fn free_memory(
        &mut self,
        vram: &mut VramAllocator,
        store: &mut FrameStore,
        base: GpuVirtAddr,
    ) -> Result<()> {
        self.expect_state("free_memory", &[VgpuState::Running, VgpuState::Created])?;
        let alloc = self
            .allocations
            .remove(&base.0)
            .ok_or(VgpuError::NotMapped { addr: base })?;
        let freed = self.aspace.unmap(base, alloc.pages)?;
        for frame in freed {
            store.scrub(frame);
        }
        for range in alloc.ranges {
            vram.free(range);
        }
        self.vram_used -= alloc.pages * FRAME_SIZE;
        Ok(())
    }

    /// Create a channel; fails past the profile's channel limit.
    pub fn create_channel(&mut self) -> Result<ChannelId> {
        self.expect_state("create_channel", &[VgpuState::Running, VgpuState::Created])?;
        if self.channels.len() as u32 >= self.profile.max_channels {
            return Err(VgpuError::ProfileUnsatisfiable {
                why: "channel limit reached".to_string(),
            });
        }
        let id = ChannelId(self.channels.len() as u32);
        self.channels
            .push(Channel::new(id, self.profile.ring_slots));
        Ok(id)
    }

    /// Submit a command to one of this vGPU's channels. Only legal while
    /// Running: a suspended vGPU's rings must be a *closed set* so a
    /// migration pass can copy them without chasing a moving target.
    ///
    /// Kernel programs are statically validated *here*, at the doorbell:
    /// a malformed program is a typed error to the submitter, never a
    /// runtime surprise in the engine — and the interpreter's hot loop
    /// gets to index registers and branch targets unchecked.
    pub fn submit(&mut self, ch: ChannelId, cmd: Command) -> Result<()> {
        self.expect_state("submit", &[VgpuState::Running])?;
        if let Command::KernelLaunch { args, program, .. } = &cmd {
            crate::isa::validate(program, args.len())?;
        }
        let channel = self
            .channels
            .get_mut(ch.0 as usize)
            .ok_or(VgpuError::NoSuchChannel(ch))?;
        channel.submit(cmd)
    }

    /// Last signaled fence value on a channel (the guest's completion poll).
    pub fn fence_value(&self, ch: ChannelId) -> Result<u64> {
        let channel = self
            .channels
            .get(ch.0 as usize)
            .ok_or(VgpuError::NoSuchChannel(ch))?;
        Ok(channel.completed_fence)
    }

    /// True if any channel has queued work — the scheduler's runnability
    /// predicate.
    pub fn has_pending_work(&self) -> bool {
        self.state == VgpuState::Running && self.channels.iter().any(|c| !c.ring.is_empty())
    }

    // -- migration primitives ------------------------------------------------

    /// The live allocations as `(base VA, bytes)`, in creation order.
    ///
    /// Creation order matters: the heap is a monotonic bump allocator, so
    /// replaying these `alloc_memory` calls in order on a *fresh* vGPU
    /// reproduces the same guest VAs deterministically. That replay is how
    /// a migration destination rebuilds the address-space shape without
    /// ever seeing (or needing) the source's physical frame numbers —
    /// `BTreeMap` iteration is ascending-by-base, which for a bump heap
    /// *is* creation order.
    pub fn list_allocations(&self) -> Vec<(GpuVirtAddr, u64)> {
        self.allocations
            .iter()
            .map(|(base, alloc)| (GpuVirtAddr(*base), alloc.pages * FRAME_SIZE))
            .collect()
    }

    /// Harvest and clear the dirty-page set (page-aligned guest VAs
    /// written or newly mapped since the last call). Legal while Running —
    /// that is the whole point of *live* pre-copy.
    pub fn take_dirty(&mut self) -> Vec<GpuVirtAddr> {
        self.aspace.take_dirty()
    }

    /// Export all channels' migratable state.
    ///
    /// The precondition is *the rings cannot change*, and two states give
    /// that: `Suspended` (frozen mid-life) and `Created` (never started,
    /// so it has no channels and cannot accept submissions). Naming the
    /// property rather than one state that implies it is what keeps a
    /// merely-placed tenant movable — the alternative silently makes the
    /// tenants an operator is most likely to relocate the ones that
    /// cannot be.
    pub fn export_channels(&self) -> Result<Vec<crate::cmd::ChannelExport>> {
        self.expect_state(
            "export_channels",
            &[VgpuState::Suspended, VgpuState::Created],
        )?;
        Ok(self
            .channels
            .iter()
            .map(|ch| crate::cmd::ChannelExport {
                pending: ch.ring.iter_pending().cloned().collect(),
                completed_fence: ch.completed_fence,
                submitted_fence: ch.submitted_fence,
                faulted: ch.state == ChannelState::Faulted,
            })
            .collect())
    }

    /// Reconstruct channels from an export, on a *fresh* (Created, no
    /// channels yet) vGPU. Pending programs are re-validated — migrated
    /// state gets no trust discount — and dead channels arrive dead.
    pub fn import_channels(&mut self, exports: Vec<crate::cmd::ChannelExport>) -> Result<()> {
        self.expect_state("import_channels", &[VgpuState::Created])?;
        if !self.channels.is_empty() {
            return Err(VgpuError::InvalidState {
                actual: "has channels".to_string(),
                wanted: "import_channels into a fresh vGPU".to_string(),
            });
        }
        if exports.len() as u32 > self.profile.max_channels {
            return Err(VgpuError::ProfileUnsatisfiable {
                why: "import exceeds channel limit".to_string(),
            });
        }
        for export in exports {
            let id = ChannelId(self.channels.len() as u32);
            let mut channel = Channel::new(id, self.profile.ring_slots);
            for cmd in export.pending {
                if let Command::KernelLaunch { args, program, .. } = &cmd {
                    crate::isa::validate(program, args.len())?;
                }
                channel.ring.push(cmd)?;
            }
            channel.completed_fence = export.completed_fence;
            // Restore the fence clock *after* replaying pending commands:
            // pushing them directly onto the ring above bypasses submit's
            // monotonicity check (they already passed it on the source),
            // so the high-water mark is set from the export, not derived.
            channel.submitted_fence = export.submitted_fence.max(export.completed_fence);
            if export.faulted {
                channel.state = ChannelState::Faulted;
            }
            self.channels.push(channel);
        }
        Ok(())
    }

    /// Tear everything down: unmap and scrub all memory, return all
    /// frames, mark Destroyed. Idempotence (destroying a Destroyed vGPU
    /// is a no-op) makes control-plane retries safe.
    pub fn destroy(&mut self, vram: &mut VramAllocator, store: &mut FrameStore) {
        if self.state == VgpuState::Destroyed {
            return;
        }
        let bases: Vec<u64> = self.allocations.keys().copied().collect();
        for base in bases {
            let alloc = self.allocations.remove(&base).expect("key from iteration");
            if let Ok(freed) = self.aspace.unmap(GpuVirtAddr(base), alloc.pages) {
                for frame in freed {
                    store.scrub(frame);
                }
            }
            for range in alloc.ranges {
                vram.free(range);
            }
        }
        self.vram_used = 0;
        self.channels.clear();
        self.state = VgpuState::Destroyed;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile() -> VgpuProfile {
        VgpuProfile {
            name: "test-1g".to_string(),
            vram_bytes: 16 * FRAME_SIZE,
            compute_weight: 1,
            max_channels: 2,
            ring_slots: 8,
            qos: QosLimits::default(),
        }
    }

    #[test]
    fn lifecycle_transitions_are_enforced() {
        let mut v = Vgpu::new(VgpuId(0), profile()).unwrap();
        assert!(v.resume().is_err()); // Created cannot resume
        v.start().unwrap();
        assert!(v.start().is_err()); // double-start rejected
        v.suspend().unwrap();
        assert!(v
            .submit(ChannelId(0), Command::FenceSignal { value: 1 })
            .is_err());
        v.resume().unwrap();
    }

    #[test]
    fn budget_is_enforced_before_physical_allocation() {
        let mut vram = VramAllocator::new(1024 * FRAME_SIZE); // card is huge
        let mut v = Vgpu::new(VgpuId(0), profile()).unwrap(); // budget: 16 frames
        v.start().unwrap();
        v.alloc_memory(&mut vram, 16 * FRAME_SIZE).unwrap();
        let err = v.alloc_memory(&mut vram, FRAME_SIZE).unwrap_err();
        assert!(matches!(err, VgpuError::VramBudgetExceeded { .. }));
        // The card still has plenty free — the *profile* said no.
        assert_eq!(vram.free_frames(), 1024 - 16);
    }

    #[test]
    fn binary_decomposition_avoids_buddy_roundup_waste() {
        let mut vram = VramAllocator::new(1024 * FRAME_SIZE);
        let mut v = Vgpu::new(VgpuId(0), profile()).unwrap();
        v.start().unwrap();
        // 5 pages should consume exactly 5 frames (4+1), not 8.
        v.alloc_memory(&mut vram, 5 * FRAME_SIZE).unwrap();
        assert_eq!(vram.free_frames(), 1024 - 5);
        assert_eq!(v.vram_used(), 5 * FRAME_SIZE);
    }

    #[test]
    fn failed_allocation_rolls_back_cleanly() {
        // Card with 4 frames; ask for 6 (as 4+2): the 4-block succeeds,
        // the 2-block fails, and the 4-block must be returned.
        let mut vram = VramAllocator::new(4 * FRAME_SIZE);
        let mut v = Vgpu::new(
            VgpuId(0),
            VgpuProfile {
                vram_bytes: 64 * FRAME_SIZE,
                ..profile()
            },
        )
        .unwrap();
        v.start().unwrap();
        let err = v.alloc_memory(&mut vram, 6 * FRAME_SIZE).unwrap_err();
        assert!(matches!(err, VgpuError::OutOfVram { .. }));
        assert_eq!(vram.free_frames(), 4, "rollback must return every frame");
        assert_eq!(v.vram_used(), 0);
    }

    #[test]
    fn free_returns_frames_and_budget() {
        let mut vram = VramAllocator::new(64 * FRAME_SIZE);
        let mut store = FrameStore::new();
        let mut v = Vgpu::new(VgpuId(0), profile()).unwrap();
        v.start().unwrap();
        let a = v.alloc_memory(&mut vram, 3 * FRAME_SIZE).unwrap();
        v.free_memory(&mut vram, &mut store, a).unwrap();
        assert_eq!(vram.free_frames(), 64);
        assert_eq!(v.vram_used(), 0);
        // Double-free is rejected.
        assert!(v.free_memory(&mut vram, &mut store, a).is_err());
    }

    #[test]
    fn destroy_releases_everything_and_is_idempotent() {
        let mut vram = VramAllocator::new(64 * FRAME_SIZE);
        let mut store = FrameStore::new();
        let mut v = Vgpu::new(VgpuId(0), profile()).unwrap();
        v.start().unwrap();
        v.alloc_memory(&mut vram, 4 * FRAME_SIZE).unwrap();
        v.alloc_memory(&mut vram, 2 * FRAME_SIZE).unwrap();
        v.destroy(&mut vram, &mut store);
        assert_eq!(vram.free_frames(), 64);
        assert_eq!(v.state(), VgpuState::Destroyed);
        v.destroy(&mut vram, &mut store); // idempotent
        assert_eq!(vram.free_frames(), 64);
    }

    #[test]
    fn channel_limit_is_enforced() {
        let mut v = Vgpu::new(VgpuId(0), profile()).unwrap();
        v.start().unwrap();
        v.create_channel().unwrap();
        v.create_channel().unwrap();
        assert!(v.create_channel().is_err());
    }
}
