//! Integration tests: the multi-tenant scenarios the whole crate exists
//! to make true. Each test is a security or fairness *claim* stated as
//! executable code; `docs/03-architecture.md` cites them by name.

use vgpu_core::isa::busy;
use vgpu_core::prelude::*;

fn node() -> GpuNode {
    GpuNode::new(PhysGpuConfig {
        name: "sim-256f".to_string(),
        vram_bytes: 256 * FRAME_SIZE,
        slice_cycles: 100,
    })
}

fn profile(name: &'static str, frames: u64, weight: u32) -> VgpuProfile {
    VgpuProfile {
        name: name.to_string(),
        vram_bytes: frames * FRAME_SIZE,
        compute_weight: weight,
        max_channels: 4,
        ring_slots: 256,
    }
}

/// Two tenants can hold the *same numeric* guest address and see their own
/// data — and neither can reach the other's memory at all, because there
/// is no cross-vGPU addressing in the entire API surface.
#[test]
fn tenants_are_isolated_by_construction() {
    let mut node = node();
    let a = node.create_vgpu(profile("a", 32, 1)).unwrap();
    let b = node.create_vgpu(profile("b", 32, 1)).unwrap();
    node.start_vgpu(a).unwrap();
    node.start_vgpu(b).unwrap();

    let va_a = node.alloc_memory(a, FRAME_SIZE).unwrap();
    let va_b = node.alloc_memory(b, FRAME_SIZE).unwrap();
    // Same guest VA in both spaces (deterministic heap base), different data.
    assert_eq!(va_a, va_b);
    node.dma_write(a, va_a, b"tenant-a-secret!").unwrap();
    node.dma_write(b, va_b, b"tenant-b-payload").unwrap();

    let mut buf = [0u8; 16];
    node.dma_read(a, va_a, &mut buf).unwrap();
    assert_eq!(&buf, b"tenant-a-secret!");
    node.dma_read(b, va_b, &mut buf).unwrap();
    assert_eq!(&buf, b"tenant-b-payload");
}

/// A wild pointer kills the offending channel — and nothing else. The
/// same tenant's other channel and the other tenant keep working.
#[test]
fn fault_blast_radius_is_one_channel() {
    let mut node = node();
    let a = node.create_vgpu(profile("a", 32, 1)).unwrap();
    let b = node.create_vgpu(profile("b", 32, 1)).unwrap();
    node.start_vgpu(a).unwrap();
    node.start_vgpu(b).unwrap();

    let ch_bad = node.create_channel(a).unwrap();
    let ch_ok = node.create_channel(a).unwrap();
    let ch_b = node.create_channel(b).unwrap();

    // a's first channel dereferences unmapped memory.
    node.submit(
        a,
        ch_bad,
        Command::MemFill {
            dst: GpuVirtAddr(0),
            len: 16,
            value: 7,
        },
    )
    .unwrap();
    node.submit(a, ch_bad, Command::FenceSignal { value: 1 })
        .unwrap();
    // a's second channel and b's channel do legitimate work.
    node.submit(a, ch_ok, Command::FenceSignal { value: 1 })
        .unwrap();
    node.submit(b, ch_b, Command::FenceSignal { value: 1 })
        .unwrap();

    let report = node.tick(1_000_000);

    assert_eq!(report.faults.len(), 1);
    assert_eq!(report.faults[0].vgpu, a);
    assert_eq!(report.faults[0].channel, ch_bad);
    assert!(matches!(
        report.faults[0].error,
        VgpuError::PageFault { .. }
    ));

    // The dead channel never signaled; the healthy ones did.
    assert_eq!(node.fence_value(a, ch_bad).unwrap(), 0);
    assert_eq!(node.fence_value(a, ch_ok).unwrap(), 1);
    assert_eq!(node.fence_value(b, ch_b).unwrap(), 1);

    // And the dead channel rejects further work.
    assert!(matches!(
        node.submit(a, ch_bad, Command::FenceSignal { value: 2 }),
        Err(VgpuError::ChannelFaulted(_))
    ));
}

/// Freed VRAM is scrubbed before reuse: the next tenant to receive the
/// frames reads zeros, not the previous tenant's data.
#[test]
fn freed_vram_is_scrubbed_before_reuse() {
    let mut node = node();
    let a = node.create_vgpu(profile("a", 32, 1)).unwrap();
    node.start_vgpu(a).unwrap();

    let va = node.alloc_memory(a, FRAME_SIZE).unwrap();
    node.dma_write(a, va, b"residual secret").unwrap();
    node.free_memory(a, va).unwrap();
    node.destroy_vgpu(a).unwrap();

    // New tenant lands on the same (lowest-address) frames — the buddy
    // allocator's determinism guarantees reuse here.
    let b = node.create_vgpu(profile("b", 32, 1)).unwrap();
    node.start_vgpu(b).unwrap();
    let vb = node.alloc_memory(b, FRAME_SIZE).unwrap();
    let mut buf = [0u8; 15];
    node.dma_read(b, vb, &mut buf).unwrap();
    assert_eq!(
        buf, [0u8; 15],
        "scrubbing must prevent cross-tenant data leaks"
    );
}

/// The weighted fair scheduler delivers compute in proportion to profile
/// weights over a saturated window: 3:1 weights → 3:1 cycles, exactly,
/// because the simulation is deterministic.
#[test]
fn compute_shares_follow_profile_weights() {
    let mut node = node();
    let heavy = node.create_vgpu(profile("heavy", 32, 3)).unwrap();
    let light = node.create_vgpu(profile("light", 32, 1)).unwrap();
    node.start_vgpu(heavy).unwrap();
    node.start_vgpu(light).unwrap();
    let ch_h = node.create_channel(heavy).unwrap();
    let ch_l = node.create_channel(light).unwrap();

    // Keep both saturated: 200 kernels of 100 cycles each, per tenant.
    for _ in 0..200 {
        node.submit(
            heavy,
            ch_h,
            Command::KernelLaunch {
                name: "h".to_string(),
                threads: 1,
                args: vec![],
                program: busy(100),
            },
        )
        .unwrap();
        node.submit(
            light,
            ch_l,
            Command::KernelLaunch {
                name: "l".to_string(),
                threads: 1,
                args: vec![],
                program: busy(100),
            },
        )
        .unwrap();
    }

    // Run a window that leaves both queues non-empty (contention holds
    // throughout): 200 slices of 100 cycles = 20k cycles of demand per
    // tenant, window of 16k total.
    node.tick(16_000);

    let (h, l) = (node.consumed(heavy), node.consumed(light));
    assert_eq!(h + l, 16_000, "the GPU was saturated the whole window");
    assert_eq!(h, 12_000, "weight-3 tenant gets exactly 3/4");
    assert_eq!(l, 4_000, "weight-1 tenant gets exactly 1/4");
}

/// End-to-end data flow: upload → device-side copy across an allocation
/// boundary → fence → download. Exercises DMA, rings, translation with
/// scattered frames, and fence ordering in one scenario.
#[test]
fn end_to_end_upload_copy_download() {
    let mut node = node();
    let t = node.create_vgpu(profile("t", 64, 1)).unwrap();
    node.start_vgpu(t).unwrap();
    let ch = node.create_channel(t).unwrap();

    // Two separate allocations -> almost certainly non-adjacent frames.
    let src = node.alloc_memory(t, 3 * FRAME_SIZE).unwrap();
    let dst = node.alloc_memory(t, 3 * FRAME_SIZE).unwrap();

    // A recognizable payload spanning two pages of src.
    let payload: Vec<u8> = (0..(FRAME_SIZE + 1000) as usize)
        .map(|i| (i % 251) as u8)
        .collect();
    let src_off = GpuVirtAddr(src.0 + 500); // deliberately unaligned
    node.dma_write(t, src_off, &payload).unwrap();

    node.submit(
        t,
        ch,
        Command::MemCopy {
            src: src_off,
            dst,
            len: payload.len() as u64,
        },
    )
    .unwrap();
    node.submit(t, ch, Command::FenceSignal { value: 1 })
        .unwrap();

    let report = node.tick(100_000);
    assert!(report.faults.is_empty());
    assert_eq!(node.fence_value(t, ch).unwrap(), 1);

    let mut out = vec![0u8; payload.len()];
    node.dma_read(t, dst, &mut out).unwrap();
    assert_eq!(out, payload, "bytes must survive the round trip intact");
}

/// A suspended vGPU stops receiving both submissions and GPU time, and
/// picks up exactly where it left off on resume.
#[test]
fn suspend_freezes_execution_and_resume_continues() {
    let mut node = node();
    let t = node.create_vgpu(profile("t", 32, 1)).unwrap();
    node.start_vgpu(t).unwrap();
    let ch = node.create_channel(t).unwrap();

    for i in 1..=3u64 {
        node.submit(
            t,
            ch,
            Command::KernelLaunch {
                name: "k".to_string(),
                threads: 1,
                args: vec![],
                program: busy(1000),
            },
        )
        .unwrap();
        node.submit(t, ch, Command::FenceSignal { value: i })
            .unwrap();
    }

    node.suspend_vgpu(t).unwrap();
    let frozen = node.tick(1_000_000);
    assert_eq!(
        frozen.cycles, 0,
        "suspended tenant must receive no GPU time"
    );
    assert!(node
        .submit(t, ch, Command::FenceSignal { value: 9 })
        .is_err());

    node.resume_vgpu(t).unwrap();
    node.tick(1_000_000);
    assert_eq!(
        node.fence_value(t, ch).unwrap(),
        3,
        "queued work completes after resume"
    );
}

/// Every write path — host DMA, fill, copy, kernel store — must set the
/// dirty bit on the page it wrote. A missed mark is silent
/// post-migration corruption, so this test is the safety net under the
/// "every write marks" invariant in `gmmu::mark_dirty_range`.
#[test]
fn every_write_path_marks_dirty() {
    let mut node = node();
    let t = node.create_vgpu(profile("t", 32, 1)).unwrap();
    node.start_vgpu(t).unwrap();
    let buf = node.alloc_memory(t, 4 * FRAME_SIZE).unwrap();
    let page = |i: u64| GpuVirtAddr(buf.0 + i * FRAME_SIZE);

    // Fresh allocation: every page is born dirty (never copied anywhere).
    let initial = node.take_dirty(t).unwrap();
    assert_eq!(initial.len(), 4);

    // 1. Host DMA write.
    node.dma_write(t, page(0), b"dma").unwrap();
    // 2-4. Device-side fill, copy (dst = page 2), kernel store (page 3).
    let ch = node.create_channel(t).unwrap();
    node.submit(
        t,
        ch,
        Command::MemFill {
            dst: page(1),
            len: 64,
            value: 7,
        },
    )
    .unwrap();
    node.submit(
        t,
        ch,
        Command::MemCopy {
            src: page(1),
            dst: page(2),
            len: 64,
        },
    )
    .unwrap();
    node.submit(
        t,
        ch,
        Command::KernelLaunch {
            name: "store".to_string(),
            threads: 1,
            args: vec![page(3).0],
            program: vec![
                vgpu_core::isa::Instr::Imm { dst: 2, value: 42 },
                vgpu_core::isa::Instr::St {
                    src: 2,
                    addr: 1,
                    offset: 0,
                },
                vgpu_core::isa::Instr::Halt,
            ],
        },
    )
    .unwrap();
    node.tick(1_000_000);

    let mut dirty = node.take_dirty(t).unwrap();
    dirty.sort();
    assert_eq!(
        dirty,
        vec![page(0), page(1), page(2), page(3)],
        "dma, fill, copy-dst, and kernel-store pages must all be dirty"
    );
    // And the set was cleared by the harvest: nothing new -> nothing dirty.
    assert!(node.take_dirty(t).unwrap().is_empty());
    // Copy *source* (page 1) was re-read, not re-written: after the
    // harvest above, reading must not re-dirty anything.
    node.submit(
        t,
        ch,
        Command::MemCopy {
            src: page(1),
            dst: page(2),
            len: 64,
        },
    )
    .unwrap();
    node.tick(1_000_000);
    assert_eq!(node.take_dirty(t).unwrap(), vec![page(2)]);
}

/// The milestone-3 claim, in-core: a suspended vGPU moves to a different
/// node — different physical frames, same guest VAs, same memory
/// contents, same fence state — and its *pending, unexecuted* work
/// completes correctly on the destination.
#[test]
fn stop_and_copy_migration_between_nodes() {
    let mut src = node();
    let mut dst = GpuNode::new(PhysGpuConfig {
        name: "sim-dst".to_string(),
        vram_bytes: 256 * FRAME_SIZE,
        slice_cycles: 100,
    });
    // Occupy dst's low frames so the twin lands on *different* physical
    // frames than the source used — proving frames don't migrate, VAs do.
    let squatter = dst.create_vgpu(profile("squat", 8, 1)).unwrap();
    dst.alloc_memory(squatter, 8 * FRAME_SIZE).unwrap();

    // Source tenant: data + completed work + PENDING work.
    let t = src.create_vgpu(profile("t", 32, 1)).unwrap();
    src.start_vgpu(t).unwrap();
    let a = src.alloc_memory(t, 2 * FRAME_SIZE).unwrap();
    let b = src.alloc_memory(t, FRAME_SIZE).unwrap();
    src.dma_write(t, a, b"payload before migration").unwrap();
    let ch = src.create_channel(t).unwrap();
    src.submit(t, ch, Command::FenceSignal { value: 1 })
        .unwrap();
    src.tick(1_000); // fence 1 completes on the source...
    assert_eq!(src.fence_value(t, ch).unwrap(), 1);
    // ...and this copy + fence 2 stay PENDING across the migration.
    src.submit(
        t,
        ch,
        Command::MemCopy {
            src: a,
            dst: b,
            len: 24,
        },
    )
    .unwrap();
    src.submit(t, ch, Command::FenceSignal { value: 2 })
        .unwrap();

    // ---- the migration, from primitives ----
    src.suspend_vgpu(t).unwrap();
    let twin = dst.create_vgpu(src.vgpu_profile(t).unwrap()).unwrap();
    for (base, bytes) in src.list_allocations(t).unwrap() {
        let got = dst.alloc_memory(twin, bytes).unwrap();
        assert_eq!(got, base, "allocation replay must reproduce guest VAs");
    }
    for page in src.take_dirty(t).unwrap() {
        let mut data = vec![0u8; FRAME_SIZE as usize];
        src.dma_read(t, page, &mut data).unwrap();
        dst.dma_write(twin, page, &data).unwrap();
    }
    dst.import_channels(twin, src.export_channels(t).unwrap())
        .unwrap();
    dst.start_vgpu(twin).unwrap();
    src.destroy_vgpu(t).unwrap();
    // ---- end migration ----

    // Fence state carried over; pending work has NOT run yet.
    assert_eq!(dst.fence_value(twin, ch).unwrap(), 1);
    // Data landed at the same guest VA.
    let mut check = [0u8; 24];
    dst.dma_read(twin, a, &mut check).unwrap();
    assert_eq!(&check, b"payload before migration");
    // The pending copy executes on the destination and signals fence 2.
    let report = dst.tick(1_000_000);
    assert!(report.faults.is_empty());
    assert_eq!(dst.fence_value(twin, ch).unwrap(), 2);
    dst.dma_read(twin, b, &mut check).unwrap();
    assert_eq!(&check, b"payload before migration");
}
