//! Checkpoint, restore, and clone over real daemons. Migration moves a
//! tenant between live nodes; these move one *out of time* — to bytes you
//! can keep, and back again more than once.

use std::net::SocketAddr;

use vgpu_core::cmd::Command;
use vgpu_core::node::PhysGpuConfig;
use vgpu_core::sched::QosLimits;
use vgpu_core::types::FRAME_SIZE;
use vgpu_core::vgpu::{VgpuProfile, VgpuState};
use vgpu_proto::wire::WireError;
use vgpu_proto::{checkpoint, clone_tenant, restore, Checkpoint, VgpuClient};
use vgpud::{serve, DaemonConfig, ServerHandle};

fn spawn() -> ServerHandle {
    let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    serve(
        addr,
        DaemonConfig {
            gpu: PhysGpuConfig {
                name: "ckpt".to_string(),
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
        name: "ckpt".to_string(),
        vram_bytes: frames * FRAME_SIZE,
        compute_weight: 1,
        max_channels: 4,
        ring_slots: 64,
        qos: QosLimits::default(),
    }
}

/// A tenant survives a round trip through raw bytes: memory, heap shape
/// (holes included), fence state, and *pending unexecuted work* all come
/// back, and the pending work then runs on the restored tenant.
///
/// This is the suspend-to-disk story: an operator can free a card when
/// there is nowhere to migrate *to*, which is precisely when capacity is
/// the problem migration cannot solve.
#[test]
fn a_tenant_survives_a_round_trip_through_bytes() {
    let node = spawn();
    let mut c = VgpuClient::connect(node.addr).unwrap();

    let t = c.create_vgpu(profile(32)).unwrap();
    c.start_vgpu(t).unwrap();
    let hole = c.alloc_memory(t, FRAME_SIZE).unwrap();
    let data = c.alloc_memory(t, 2 * FRAME_SIZE).unwrap();
    let dst = c.alloc_memory(t, FRAME_SIZE).unwrap();
    c.free_memory(t, hole).unwrap(); // a heap hole must survive too

    c.dma_write(t, data, b"frozen in amber").unwrap();
    let ch = c.create_channel(t).unwrap();
    c.submit(t, ch, Command::FenceSignal { value: 1 }).unwrap();
    c.tick(10_000).unwrap();
    // Queued but NOT executed at checkpoint time:
    c.submit(
        t,
        ch,
        Command::MemCopy {
            src: data,
            dst,
            len: 15,
        },
    )
    .unwrap();
    c.submit(t, ch, Command::FenceSignal { value: 2 }).unwrap();

    // Freeze to bytes, then hand the capacity back entirely.
    let ckpt = checkpoint(&mut c, t).unwrap();
    let bytes = ckpt.to_bytes();
    c.destroy_vgpu(t).unwrap();
    assert_eq!(
        c.node_info().unwrap().uncommitted_vram,
        256 * FRAME_SIZE,
        "the card is fully free while the tenant lives in a byte string"
    );

    // Later — possibly on another machine — parse and restore.
    let parsed = Checkpoint::from_bytes(&bytes).unwrap();
    assert_eq!(parsed, ckpt, "bytes round-trip losslessly");
    let back = restore(&mut c, &parsed).unwrap();

    assert_eq!(c.vgpu_state(back).unwrap(), VgpuState::Running);
    assert_eq!(c.fence_value(back, ch).unwrap(), 1, "fence state restored");
    assert_eq!(
        c.dma_read(back, data, 15).unwrap(),
        b"frozen in amber".to_vec()
    );
    // The heap hole came back as a hole.
    assert!(c.dma_read(back, hole, 8).is_err());

    // And the work that was still queued when we froze now runs.
    c.tick(100_000).unwrap();
    assert_eq!(c.fence_value(back, ch).unwrap(), 2);
    assert_eq!(
        c.dma_read(back, dst, 15).unwrap(),
        b"frozen in amber".to_vec()
    );
    node.shutdown();
}

/// **Clone: fork a warm tenant.** Restoring one checkpoint twice yields
/// two independent tenants with identical memory. The production use is
/// warm-start replication — an inference worker spends its first minutes
/// loading weights into VRAM, and every replica after the first can skip
/// that entirely.
///
/// "Independent" is the load-bearing word: the clones must share nothing,
/// or this is a distributed aliasing bug rather than a feature.
#[test]
fn cloning_a_warm_tenant_yields_independent_copies() {
    let node = spawn();
    let mut c = VgpuClient::connect(node.addr).unwrap();

    // A tenant that has done expensive warm-up work.
    let warm = c.create_vgpu(profile(32)).unwrap();
    c.start_vgpu(warm).unwrap();
    let weights = c.alloc_memory(warm, 2 * FRAME_SIZE).unwrap();
    c.dma_write(warm, weights, b"expensively loaded weights")
        .unwrap();

    let ckpt = checkpoint(&mut c, warm).unwrap();
    assert!(ckpt.memory_bytes() > 0);

    // Three replicas from one warm-up.
    let replicas: Vec<_> = (0..3).map(|_| restore(&mut c, &ckpt).unwrap()).collect();
    for r in &replicas {
        assert_eq!(
            c.dma_read(*r, weights, 26).unwrap(),
            b"expensively loaded weights".to_vec(),
            "every replica starts warm"
        );
    }

    // Distinct identities, and writes do not bleed between them.
    assert_eq!(
        replicas
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len(),
        3,
        "clones must not share an identity"
    );
    c.dma_write(replicas[0], weights, b"replica-0 diverged")
        .unwrap();
    assert_eq!(
        c.dma_read(replicas[1], weights, 26).unwrap(),
        b"expensively loaded weights".to_vec(),
        "clones must not share memory"
    );
    // Including the original, which is merely suspended, not consumed.
    assert_eq!(c.vgpu_state(warm).unwrap(), VgpuState::Suspended);
    c.resume_vgpu(warm).unwrap();
    assert_eq!(
        c.dma_read(warm, weights, 26).unwrap(),
        b"expensively loaded weights".to_vec()
    );
    node.shutdown();
}

/// A clone can land on a *different* node than its parent — the same
/// property migration has, for the same reason: nothing physical travels,
/// only guest-observable state.
#[test]
fn a_clone_can_be_restored_onto_another_node() {
    let (na, nb) = (spawn(), spawn());
    let mut a = VgpuClient::connect(na.addr).unwrap();
    let mut b = VgpuClient::connect(nb.addr).unwrap();

    // Occupy node b's low frames so the clone provably lands on
    // different physical memory.
    let squatter = b.create_vgpu(profile(8)).unwrap();
    b.alloc_memory(squatter, 8 * FRAME_SIZE).unwrap();

    let parent = a.create_vgpu(profile(16)).unwrap();
    a.start_vgpu(parent).unwrap();
    let buf = a.alloc_memory(parent, FRAME_SIZE).unwrap();
    a.dma_write(parent, buf, b"crossed the wire twice").unwrap();

    let child = clone_tenant(&mut a, &mut b, parent).unwrap();
    assert_eq!(
        b.dma_read(child, buf, 22).unwrap(),
        b"crossed the wire twice".to_vec()
    );
    assert_eq!(b.vgpu_state(child).unwrap(), VgpuState::Running);
    // The parent is left frozen for the caller to resume or destroy —
    // never silently resumed, since the fork instant is the only moment
    // the two are known identical.
    assert_eq!(a.vgpu_state(parent).unwrap(), VgpuState::Suspended);
    na.shutdown();
    nb.shutdown();
}

/// A checkpoint is a file, and a file is exactly as trustworthy as a
/// socket: garbage decodes to a typed error, never a panic. The version
/// byte matters more here than on the wire — a peer with the wrong
/// version is an error you see immediately; a *file* with the wrong
/// version is a corruption you see in six months.
#[test]
fn corrupt_and_stale_checkpoints_are_refused_not_trusted() {
    let node = spawn();
    let mut c = VgpuClient::connect(node.addr).unwrap();
    let t = c.create_vgpu(profile(8)).unwrap();
    c.start_vgpu(t).unwrap();
    c.alloc_memory(t, FRAME_SIZE).unwrap();
    let good = checkpoint(&mut c, t).unwrap().to_bytes();

    // Truncation at every length: none may panic.
    for cut in 0..good.len().min(400) {
        let _ = Checkpoint::from_bytes(&good[..cut]);
    }
    // Trailing junk is caught rather than ignored.
    let mut extra = good.clone();
    extra.push(0xFF);
    assert!(matches!(
        Checkpoint::from_bytes(&extra),
        Err(WireError::TrailingBytes(_))
    ));
    // A checkpoint from another protocol version is refused loudly.
    let mut stale = good.clone();
    stale[0] = 1;
    assert!(matches!(
        Checkpoint::from_bytes(&stale),
        Err(WireError::VersionMismatch { theirs: 1, .. })
    ));
    node.shutdown();
}

/// Restoring must not smuggle a tenant onto a node past admission
/// control: the checkpoint carries its profile, and the node re-admits
/// under it like any other request. A card without room says no.
#[test]
fn restoring_still_obeys_admission_control() {
    let (na, nb) = (spawn(), spawn());
    let mut a = VgpuClient::connect(na.addr).unwrap();
    let mut b = VgpuClient::connect(nb.addr).unwrap();

    let t = a.create_vgpu(profile(32)).unwrap();
    a.start_vgpu(t).unwrap();
    a.alloc_memory(t, FRAME_SIZE).unwrap();
    let ckpt = checkpoint(&mut a, t).unwrap();

    // Fill node b completely, then try to restore into it.
    b.create_vgpu(profile(256)).unwrap();
    assert!(
        restore(&mut b, &ckpt).is_err(),
        "a full node must refuse a restore like any other admission"
    );
    // And the failed restore left nothing behind.
    assert_eq!(b.node_info().unwrap().uncommitted_vram, 0);
    na.shutdown();
    nb.shutdown();
}
