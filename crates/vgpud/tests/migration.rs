//! Live migration, end to end: two real daemons on TCP, one tenant
//! moving between them with data, completed work, and *pending* work in
//! flight. The milestone-3 acceptance suite.

use std::net::SocketAddr;

use vgpu_core::cmd::Command;
use vgpu_core::node::PhysGpuConfig;
use vgpu_core::sched::QosLimits;
use vgpu_core::types::{GpuVirtAddr, VgpuError, FRAME_SIZE};
use vgpu_core::vgpu::{VgpuProfile, VgpuState};
use vgpu_proto::{migrate, ClientError, MigrateOptions, VgpuClient};
use vgpud::{serve, DaemonConfig, ServerHandle};

fn spawn_daemon(name: &str) -> ServerHandle {
    let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    serve(
        addr,
        DaemonConfig {
            gpu: PhysGpuConfig {
                name: name.to_string(),
                vram_bytes: 256 * FRAME_SIZE,
                slice_cycles: 100,
            },
            auto_tick: None,
        },
    )
    .expect("daemon binds")
}

fn profile(frames: u64) -> VgpuProfile {
    VgpuProfile {
        name: "mig".to_string(),
        vram_bytes: frames * FRAME_SIZE,
        compute_weight: 1,
        max_channels: 4,
        ring_slots: 256,
        qos: QosLimits::default(),
    }
}

/// The full story: a tenant with completed work, live data, a freed
/// allocation (heap hole), and pending unexecuted commands moves from
/// node A to node B. Everything the guest can observe is preserved; the
/// pending work completes on B.
#[test]
fn live_migration_between_two_daemons() {
    let node_a = spawn_daemon("node-a");
    let node_b = spawn_daemon("node-b");
    let mut a = VgpuClient::connect(node_a.addr).unwrap();
    let mut b = VgpuClient::connect(node_b.addr).unwrap();

    // Occupy node B's low frames so the twin's physical layout provably
    // differs from the source's.
    let squatter = b.create_vgpu(profile(8)).unwrap();
    b.alloc_memory(squatter, 8 * FRAME_SIZE).unwrap();

    // --- build a guest with history on node A ---
    let t = a.create_vgpu(profile(32)).unwrap();
    a.start_vgpu(t).unwrap();
    let hole = a.alloc_memory(t, FRAME_SIZE).unwrap(); // will be freed
    let data = a.alloc_memory(t, 2 * FRAME_SIZE).unwrap();
    let dst = a.alloc_memory(t, FRAME_SIZE).unwrap();
    a.free_memory(t, hole).unwrap(); // heap now has a hole at the front

    a.dma_write(t, data, b"survives the journey").unwrap();
    let ch = a.create_channel(t).unwrap();
    a.submit(t, ch, Command::FenceSignal { value: 1 }).unwrap();
    a.tick(1_000).unwrap(); // fence 1 completes on A
    assert_eq!(a.fence_value(t, ch).unwrap(), 1);
    // Pending across the migration: a copy and fence 2.
    a.submit(
        t,
        ch,
        Command::MemCopy {
            src: data,
            dst,
            len: 20,
        },
    )
    .unwrap();
    a.submit(t, ch, Command::FenceSignal { value: 2 }).unwrap();

    // --- the migration ---
    let twin = migrate(&mut a, &mut b, t, &MigrateOptions::default()).unwrap();

    // Source is gone; its node no longer knows the vGPU.
    assert!(matches!(
        a.vgpu_state(t),
        Err(ClientError::Device(VgpuError::NoSuchVgpu(_)))
    ));

    // Twin is running on B with identical guest-visible state.
    assert_eq!(b.vgpu_state(twin).unwrap(), VgpuState::Running);
    assert_eq!(b.fence_value(twin, ch).unwrap(), 1, "fence state migrated");
    assert_eq!(
        b.dma_read(twin, data, 20).unwrap(),
        b"survives the journey".to_vec()
    );
    // The heap hole migrated too: the freed VA is unmapped on the twin.
    assert!(matches!(
        b.dma_read(twin, hole, 8),
        Err(ClientError::Device(VgpuError::PageFault { .. }))
    ));

    // Pending work executes on B and signals fence 2.
    let report = b.tick(1_000_000).unwrap();
    assert!(report.faults.is_empty());
    assert_eq!(b.fence_value(twin, ch).unwrap(), 2);
    assert_eq!(
        b.dma_read(twin, dst, 20).unwrap(),
        b"survives the journey".to_vec()
    );

    // And the twin is fully alive: new work after migration.
    a.tick(1).unwrap(); // node A idles on
    b.submit(twin, ch, Command::FenceSignal { value: 3 })
        .unwrap();
    b.tick(1_000).unwrap();
    assert_eq!(b.fence_value(twin, ch).unwrap(), 3);

    node_a.shutdown();
    node_b.shutdown();
}

/// Pre-copy really iterates: writes landed between rounds still arrive.
/// We interleave guest writes with the migration by driving the rounds
/// manually through the same primitives `migrate()` uses.
#[test]
fn writes_between_precopy_rounds_are_not_lost() {
    let node_a = spawn_daemon("node-a");
    let node_b = spawn_daemon("node-b");
    let mut a = VgpuClient::connect(node_a.addr).unwrap();
    let mut b = VgpuClient::connect(node_b.addr).unwrap();

    let t = a.create_vgpu(profile(8)).unwrap();
    a.start_vgpu(t).unwrap();
    let buf = a.alloc_memory(t, 4 * FRAME_SIZE).unwrap();
    a.dma_write(t, buf, &[0x11; 128]).unwrap();

    // Round 1 by hand: twin + structure + bulk copy.
    let twin = b.create_vgpu(a.vgpu_profile(t).unwrap()).unwrap();
    for (base, bytes) in a.list_allocations(t).unwrap() {
        b.alloc_memory_at(twin, base, bytes).unwrap();
    }
    for page in a.take_dirty(t).unwrap() {
        let bytes = a.dma_read(t, page, FRAME_SIZE).unwrap();
        b.dma_write(twin, page, &bytes).unwrap();
    }

    // The guest writes AFTER the bulk copy — round 1's twin is stale.
    let late = GpuVirtAddr(buf.0 + 2 * FRAME_SIZE);
    a.dma_write(t, late, b"late write").unwrap();

    // Final round: suspend, copy exactly the dirty delta, move channels.
    a.suspend_vgpu(t).unwrap();
    let delta = a.take_dirty(t).unwrap();
    assert_eq!(delta, vec![late], "only the late-written page is dirty");
    for page in delta {
        let bytes = a.dma_read(t, page, FRAME_SIZE).unwrap();
        b.dma_write(twin, page, &bytes).unwrap();
    }
    b.import_channels(twin, a.export_channels(t).unwrap())
        .unwrap();
    b.start_vgpu(twin).unwrap();
    a.destroy_vgpu(t).unwrap();

    assert_eq!(b.dma_read(twin, late, 10).unwrap(), b"late write".to_vec());
    assert_eq!(b.dma_read(twin, buf, 4).unwrap(), vec![0x11; 4]);

    node_a.shutdown();
    node_b.shutdown();
}

/// Migration honors admission control: a destination without room
/// refuses the twin, the error surfaces typed, and the source tenant is
/// left intact and running.
#[test]
fn migration_to_a_full_node_fails_cleanly() {
    let node_a = spawn_daemon("node-a");
    let node_b = spawn_daemon("node-b");
    let mut a = VgpuClient::connect(node_a.addr).unwrap();
    let mut b = VgpuClient::connect(node_b.addr).unwrap();

    // Fill B completely.
    b.create_vgpu(profile(256)).unwrap();

    let t = a.create_vgpu(profile(32)).unwrap();
    a.start_vgpu(t).unwrap();
    let buf = a.alloc_memory(t, FRAME_SIZE).unwrap();
    a.dma_write(t, buf, b"stays home").unwrap();

    match migrate(&mut a, &mut b, t, &MigrateOptions::default()) {
        Err(vgpu_proto::MigrateError::Client(ClientError::Device(
            VgpuError::ProfileUnsatisfiable { .. },
        ))) => {}
        other => panic!("expected typed admission failure, got {other:?}"),
    }

    // Source untouched: still running, data intact.
    assert_eq!(a.vgpu_state(t).unwrap(), VgpuState::Running);
    assert_eq!(a.dma_read(t, buf, 10).unwrap(), b"stays home".to_vec());

    node_a.shutdown();
    node_b.shutdown();
}
