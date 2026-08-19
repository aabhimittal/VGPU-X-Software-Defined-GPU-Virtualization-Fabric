//! Checkpoint, restore, and clone: a tenant as a portable byte string.
//!
//! Milestone 3 moved a vGPU from one live node to another. This module
//! moves one *out of time*: freeze a tenant into bytes you can write to
//! disk, ship somewhere, keep for a week, and restore — possibly more
//! than once.
//!
//! # Why this is a different capability, not a variation on migration
//!
//! Migration's destination is a running peer, so the source can always
//! be asked another question. A checkpoint has no peer: the bytes must
//! be *self-describing and complete*, because the thing that reads them
//! may run next month against a node that shares nothing with this one.
//! That single difference is what makes the three new uses possible:
//!
//! * **Suspend to disk.** Evacuate a node for maintenance when there is
//!   nowhere to evacuate *to* — the fleet is full, or it is 3am and you
//!   would rather not wake a second machine. Migration cannot help;
//!   capacity is the whole problem it needs solved first.
//! * **Forensics.** A tenant that faults mysteriously can be frozen
//!   exactly as it died and restored later, repeatedly, on a debugging
//!   node. The device model is deterministic, so a restored checkpoint
//!   replays identically — a bug report that is a *file* rather than a
//!   story.
//! * **Clone.** Restore the same checkpoint N times and get N identical
//!   warm tenants. This is the one with real production value: an ML
//!   inference worker spends its first minutes loading weights into VRAM,
//!   and cloning a warmed-up tenant skips that for every replica. Fork,
//!   for GPUs.
//!
//! # Composed, not built
//!
//! There are no new wire messages here. A checkpoint is
//! `list_allocations` + `dma_read` + `export_channels`; a restore is
//! `create_vgpu` + `alloc_memory_at` + `dma_write` + `import_channels`.
//! Milestone 3 argued that small verbs beat one snapshot blob because a
//! caller can recombine them; this file is that claim being cashed —
//! an entire feature at the client layer, with the device unchanged.
//!
//! # What a checkpoint deliberately omits
//!
//! Physical frame numbers (meaningless elsewhere), scheduler vruntime
//! (relative to a node's other tenants), and the vGPU id (the restore
//! gets a fresh one — an id from a dead node is a lie, and clones must
//! not share one). What remains is exactly the guest-observable state,
//! which is the same principle migration follows.

use vgpu_core::cmd::ChannelExport;
use vgpu_core::types::{GpuVirtAddr, VgpuId, FRAME_SIZE, MAX_DMA_BYTES};
use vgpu_core::vgpu::{VgpuProfile, VgpuState};

use crate::client::VgpuClient;
use crate::migrate::MigrateError;
use crate::msg::{dec_channel_export, dec_profile, enc_channel_export, enc_profile};
use crate::wire::{Dec, Enc, WireError, VERSION};

/// A frozen tenant: everything needed to reconstitute it anywhere.
#[derive(Debug, Clone, PartialEq)]
pub struct Checkpoint {
    /// The contract the tenant was admitted under. Restoring re-admits
    /// under the identical profile, so a checkpoint cannot be used to
    /// smuggle a tenant onto a node at a better QoS tier than it bought.
    pub profile: VgpuProfile,
    /// Allocations as `(base VA, bytes)`, in creation order — replayed
    /// with `alloc_memory_at` so the heap's exact shape, holes included,
    /// comes back.
    pub allocations: Vec<(GpuVirtAddr, u64)>,
    /// Page contents, as `(page VA, bytes)`. Sparse: only mapped pages
    /// appear, so a tenant with a 1 GiB budget and one live page is a
    /// small file.
    pub pages: Vec<(GpuVirtAddr, Vec<u8>)>,
    /// Channel state: pending work, fence values, faulted flags.
    pub channels: Vec<ChannelExport>,
}

impl Checkpoint {
    /// Serialize to a self-describing byte string, version-tagged like
    /// every other payload in this crate. A checkpoint outlives the
    /// process that wrote it, so the version byte matters *more* here
    /// than on the wire: a peer with the wrong version is a connection
    /// error you see immediately, while a file with the wrong version is
    /// a corruption you see in six months.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut e = Enc::new();
        e.u8(VERSION);
        enc_profile(&mut e, &self.profile);
        e.u32(self.allocations.len() as u32);
        for (base, bytes) in &self.allocations {
            e.u64(base.0);
            e.u64(*bytes);
        }
        e.u32(self.pages.len() as u32);
        for (va, data) in &self.pages {
            e.u64(va.0);
            e.bytes(data);
        }
        e.u32(self.channels.len() as u32);
        for ch in &self.channels {
            enc_channel_export(&mut e, ch);
        }
        e.into_bytes()
    }

    /// Parse a checkpoint. Total, like every decoder here: arbitrary
    /// bytes yield a `WireError`, never a panic. Checkpoints come from
    /// files, and a file is exactly as trustworthy as a socket.
    pub fn from_bytes(buf: &[u8]) -> Result<Self, WireError> {
        let mut d = Dec::new(buf);
        let version = d.u8()?;
        if version != VERSION {
            return Err(WireError::VersionMismatch {
                ours: VERSION,
                theirs: version,
            });
        }
        let profile = dec_profile(&mut d)?;
        let n = d.u32()?;
        let mut allocations = Vec::with_capacity(n as usize);
        for _ in 0..n {
            allocations.push((GpuVirtAddr(d.u64()?), d.u64()?));
        }
        let n = d.u32()?;
        let mut pages = Vec::with_capacity(n as usize);
        for _ in 0..n {
            pages.push((GpuVirtAddr(d.u64()?), d.bytes()?));
        }
        let n = d.u32()?;
        let mut channels = Vec::with_capacity(n as usize);
        for _ in 0..n {
            channels.push(dec_channel_export(&mut d)?);
        }
        d.finish()?;
        Ok(Self {
            profile,
            allocations,
            pages,
            channels,
        })
    }

    /// Total bytes of guest memory captured (what the file will cost).
    pub fn memory_bytes(&self) -> u64 {
        self.pages.iter().map(|(_, d)| d.len() as u64).sum()
    }
}

/// Freeze `vgpu` into a checkpoint.
///
/// Suspends the tenant if it is running, and **leaves it suspended**:
/// resuming automatically would make the checkpoint a lie the instant it
/// was taken, since the guest would start writing to memory the bytes
/// claim to describe. The caller decides — `resume_vgpu` to carry on,
/// `destroy_vgpu` to hand the capacity back now that the state is safe
/// on disk.
pub fn checkpoint(src: &mut VgpuClient, vgpu: VgpuId) -> Result<Checkpoint, MigrateError> {
    if src.vgpu_state(vgpu)? == VgpuState::Running {
        src.suspend_vgpu(vgpu)?;
    }
    let profile = src.vgpu_profile(vgpu)?;
    let allocations = src.list_allocations(vgpu)?;

    let mut pages = Vec::new();
    for (base, bytes) in &allocations {
        for i in 0..bytes / FRAME_SIZE {
            let va = GpuVirtAddr(base.0 + i * FRAME_SIZE);
            pages.push((va, src.dma_read(vgpu, va, FRAME_SIZE)?));
        }
    }
    let channels = src.export_channels(vgpu)?;
    Ok(Checkpoint {
        profile,
        allocations,
        pages,
        channels,
    })
}

/// Rebuild a tenant from a checkpoint on `dst` and start it.
///
/// The restored tenant is a *new* tenant: fresh vGPU id, fresh physical
/// frames, no scheduler history. Everything the guest can observe —
/// pointers, memory, queued work, fence values — is identical, which is
/// the same standard migration is held to.
///
/// Restoring the same checkpoint twice yields two independent tenants,
/// which is the clone story (see [`clone_tenant`]).
pub fn restore(dst: &mut VgpuClient, ckpt: &Checkpoint) -> Result<VgpuId, MigrateError> {
    let twin = dst.create_vgpu(ckpt.profile.clone())?;
    match restore_inner(dst, ckpt, twin) {
        Ok(()) => Ok(twin),
        Err(e) => {
            // Never leave a half-built tenant holding a budget.
            let _ = dst.destroy_vgpu(twin);
            Err(e)
        }
    }
}

fn restore_inner(
    dst: &mut VgpuClient,
    ckpt: &Checkpoint,
    twin: VgpuId,
) -> Result<(), MigrateError> {
    for (base, bytes) in &ckpt.allocations {
        dst.alloc_memory_at(twin, *base, *bytes)?;
    }
    for (va, data) in &ckpt.pages {
        // Page-sized writes are already under the transfer ceiling, but
        // chunk anyway: a checkpoint may have been written by a build
        // with a different FRAME_SIZE, and silently exceeding a device
        // limit is exactly the failure this codebase keeps refusing.
        for (i, part) in data.chunks(MAX_DMA_BYTES as usize).enumerate() {
            let at = GpuVirtAddr(va.0 + (i as u64) * MAX_DMA_BYTES);
            dst.dma_write(twin, at, part)?;
        }
    }
    dst.import_channels(twin, ckpt.channels.clone())?;
    dst.start_vgpu(twin)?;
    Ok(())
}

/// Fork a tenant: checkpoint it and restore the copy alongside it.
///
/// The source is left **suspended**, not resumed, for the same reason
/// `checkpoint` leaves it suspended — and here there is a second reason
/// worth stating: the moment of the fork is the one instant at which the
/// two tenants are known to be identical, and resuming the parent before
/// the child exists would silently make the clone a copy of a *past*
/// state rather than the present one. Resume the source when the caller
/// is ready.
///
/// The intended use is warm-start replication: bring one tenant up,
/// load its weights, then clone it as many times as you have capacity —
/// each replica skipping the load entirely.
pub fn clone_tenant(
    src: &mut VgpuClient,
    dst: &mut VgpuClient,
    vgpu: VgpuId,
) -> Result<VgpuId, MigrateError> {
    let ckpt = checkpoint(src, vgpu)?;
    restore(dst, &ckpt)
}
