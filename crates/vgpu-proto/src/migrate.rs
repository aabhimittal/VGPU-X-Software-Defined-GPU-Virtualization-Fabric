//! The migration driver: move a vGPU between two nodes, live.
//!
//! This is the classic **pre-copy** algorithm (the one QEMU/vMotion made
//! standard), composed entirely from wire primitives that each already
//! exist and are tested — the driver adds *sequencing*, not mechanism:
//!
//! ```text
//! 1. twin     = dst.create_vgpu(src profile)         admission on dst
//! 2. replay     src allocations on the twin          same guest VAs
//! 3. LOOP (source still RUNNING):
//!      dirty = src.take_dirty()                      round 1 = all pages
//!      copy dirty pages src → dst                    guest keeps working
//!    until dirty set is small or round budget spent
//! 4. src.suspend()                                   the brownout begins
//! 5. copy the final dirty set                        bounded: it's small
//! 6. move channel state (pending work + fences)
//! 7. dst.start(twin); src.destroy()                  the brownout ends
//! ```
//!
//! # Why pre-copy converges (and when it doesn't)
//!
//! Each round copies the pages dirtied *during* the previous round's
//! copy. If the guest dirties pages slower than the wire can move them,
//! successive dirty sets shrink geometrically and step 4's downtime is
//! bounded by one small final copy. A guest that dirties memory faster
//! than the wire forever (`memtest` as a tenant) never converges — which
//! is why the loop has a round budget and falls back to stop-and-copy:
//! guaranteed termination, longer brownout. Real systems make the same
//! choice (or switch to post-copy, a different trade not modeled here).
//!
//! # What migrates and what deliberately does not
//!
//! Guest VAs, memory contents, pending commands, fence values: migrate.
//! Physical frame numbers: do NOT — the destination's buddy allocator
//! hands the twin whatever frames it has, and the GMMU maps them at the
//! *same guest VAs*. The tenant cannot tell. This is the entire payoff
//! of the address-space indirection built in milestone 0.
//! Scheduler vruntime also does not migrate: fairness is relative to a
//! node's *other* tenants, so a migrated vGPU joins the destination at
//! its high-water mark like any new arrival (see `sched::register`).
//!
//! # Structure drift
//!
//! The guest may `malloc`/`free` *during* step 3 — the allocation list
//! replayed in step 2 can be stale by step 4. After suspending, the
//! driver re-lists allocations; on drift it rebuilds the twin's structure
//! from the frozen truth and full-copies (the source is suspended, so
//! this is final and correct — it just forfeits pre-copy's shorter
//! brownout for that unlucky migration).

use vgpu_core::types::{GpuVirtAddr, VgpuId, FRAME_SIZE};

use crate::client::{ClientError, VgpuClient};

/// Tuning knobs for the pre-copy loop.
#[derive(Debug, Clone)]
pub struct MigrateOptions {
    /// Maximum live copy rounds before falling back to stop-and-copy.
    pub max_precopy_rounds: u32,
    /// Stop iterating early once a round's dirty set is this small —
    /// the remaining pages are cheaper to copy during the brownout than
    /// to chase for another round.
    pub dirty_threshold: usize,
}

impl Default for MigrateOptions {
    fn default() -> Self {
        Self {
            max_precopy_rounds: 3,
            dirty_threshold: 8,
        }
    }
}

/// Migration failures.
#[derive(Debug)]
pub enum MigrateError {
    /// An underlying RPC failed (either side).
    Client(ClientError),
}

impl std::fmt::Display for MigrateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Client(e) => write!(f, "migration rpc failed: {e}"),
        }
    }
}

impl std::error::Error for MigrateError {}

impl From<ClientError> for MigrateError {
    fn from(e: ClientError) -> Self {
        Self::Client(e)
    }
}

/// Move `vgpu` from the node behind `src` to the node behind `dst`.
/// Returns the twin's id on the destination. On success the source vGPU
/// is destroyed; on error the source is left as-is (possibly Suspended —
/// the caller can resume it) and the twin, if created, is torn down.
pub fn migrate(
    src: &mut VgpuClient,
    dst: &mut VgpuClient,
    vgpu: VgpuId,
    opts: &MigrateOptions,
) -> Result<VgpuId, MigrateError> {
    let twin = dst.create_vgpu(src.vgpu_profile(vgpu)?)?;
    match migrate_inner(src, dst, vgpu, twin, opts) {
        Ok(twin) => Ok(twin),
        Err(e) => {
            // Best-effort cleanup: never leak a half-built twin.
            let _ = dst.destroy_vgpu(twin);
            Err(e)
        }
    }
}

fn migrate_inner(
    src: &mut VgpuClient,
    dst: &mut VgpuClient,
    vgpu: VgpuId,
    mut twin: VgpuId,
    opts: &MigrateOptions,
) -> Result<VgpuId, MigrateError> {
    // (2) Structure replay: same allocation sequence → same guest VAs.
    let allocs = src.list_allocations(vgpu)?;
    replay_allocations(dst, twin, &allocs)?;

    // (3) Live pre-copy rounds. Round 1's take_dirty returns every mapped
    // page (pages are born dirty), so the first round IS the bulk copy.
    for _ in 0..opts.max_precopy_rounds {
        let dirty = src.take_dirty(vgpu)?;
        if dirty.is_empty() {
            break;
        }
        copy_pages(src, dst, vgpu, twin, &dirty)?;
        if dirty.len() <= opts.dirty_threshold {
            break; // diminishing returns: finish during the brownout
        }
    }

    // (4) Brownout begins: freeze the source.
    src.suspend_vgpu(vgpu)?;

    // (4b) Structure drift check against the now-frozen truth.
    let frozen = src.list_allocations(vgpu)?;
    if frozen != allocs {
        // Rebuild from scratch; source is frozen so this pass is final.
        dst.destroy_vgpu(twin)?;
        twin = dst.create_vgpu(src.vgpu_profile(vgpu)?)?;
        replay_allocations(dst, twin, &frozen)?;
        let _ = src.take_dirty(vgpu)?; // discard: we copy everything anyway
        let all_pages: Vec<GpuVirtAddr> = frozen
            .iter()
            .flat_map(|(base, bytes)| {
                (0..bytes / FRAME_SIZE).map(move |i| GpuVirtAddr(base.0 + i * FRAME_SIZE))
            })
            .collect();
        copy_pages(src, dst, vgpu, twin, &all_pages)?;
    } else {
        // (5) Final copy: exactly what changed during the last round.
        let dirty = src.take_dirty(vgpu)?;
        copy_pages(src, dst, vgpu, twin, &dirty)?;
    }

    // (6) In-flight work and guest-visible completion state.
    let channels = src.export_channels(vgpu)?;
    dst.import_channels(twin, channels)?;

    // (7) Flip: twin goes live, source ceases to exist.
    dst.start_vgpu(twin)?;
    src.destroy_vgpu(vgpu)?;
    Ok(twin)
}

/// Reproduce the source heap's exact shape on the twin — placed
/// allocations, so a heap with freed holes replays faithfully (naive
/// "alloc N bytes" replay would compact holes and shift every later VA,
/// invalidating the guest's live pointers).
fn replay_allocations(
    dst: &mut VgpuClient,
    twin: VgpuId,
    allocs: &[(GpuVirtAddr, u64)],
) -> Result<(), MigrateError> {
    for (base, bytes) in allocs {
        dst.alloc_memory_at(twin, *base, *bytes)?;
    }
    Ok(())
}

fn copy_pages(
    src: &mut VgpuClient,
    dst: &mut VgpuClient,
    vgpu: VgpuId,
    twin: VgpuId,
    pages: &[GpuVirtAddr],
) -> Result<(), MigrateError> {
    for page in pages {
        let data = src.dma_read(vgpu, *page, FRAME_SIZE)?;
        dst.dma_write(twin, *page, &data)?;
    }
    Ok(())
}
