//! The fleet, end to end: three real daemons, one control plane. Every
//! placement assertion is exact because best-fit over deterministic
//! nodes is itself deterministic — the milestone-4 acceptance suite.

use std::net::SocketAddr;

use vgpu_core::cmd::Command;
use vgpu_core::node::PhysGpuConfig;
use vgpu_core::types::FRAME_SIZE;
use vgpu_core::vgpu::VgpuProfile;
use vgpu_fabric::{Fabric, FabricError, NodeId};
use vgpu_proto::VgpuClient;
use vgpud::{serve, DaemonConfig, ServerHandle};

fn spawn_node(name: &str, frames: u64) -> ServerHandle {
    let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    serve(
        addr,
        DaemonConfig {
            gpu: PhysGpuConfig {
                name: name.to_string(),
                vram_bytes: frames * FRAME_SIZE,
                slice_cycles: 100,
            },
            auto_tick: None,
        },
    )
    .expect("daemon binds")
}

fn profile(frames: u64) -> VgpuProfile {
    VgpuProfile {
        name: format!("p{frames}"),
        vram_bytes: frames * FRAME_SIZE,
        compute_weight: 1,
        max_channels: 2,
        ring_slots: 64,
    }
}

/// Best-fit is exact and deterministic: each placement goes to the node
/// with the least sufficient slack, ties to the lower NodeId.
#[test]
fn placement_is_best_fit_and_deterministic() {
    let (na, nb, nc) = (
        spawn_node("a", 256),
        spawn_node("b", 128),
        spawn_node("c", 64),
    );
    let mut fabric = Fabric::new();
    let a = fabric.add_node(na.addr).unwrap();
    let b = fabric.add_node(nb.addr).unwrap();
    let c = fabric.add_node(nc.addr).unwrap();

    // free: a=256 b=128 c=64.
    // 100 frames: fits a (slack 156) and b (slack 28) -> b.
    assert_eq!(fabric.place(profile(100)).unwrap().node, b);
    // free: a=256 b=28 c=64.
    // 100 frames: only a fits -> a.
    assert_eq!(fabric.place(profile(100)).unwrap().node, a);
    // free: a=156 b=28 c=64.
    // 60 frames: fits a (slack 96) and c (slack 4) -> c.
    assert_eq!(fabric.place(profile(60)).unwrap().node, c);
    // free: a=156 b=28 c=4.
    // 28 frames: fits a (slack 128) and b (slack 0, exact fit) -> b.
    assert_eq!(fabric.place(profile(28)).unwrap().node, b);

    // Inventory agrees with the arithmetic above.
    let inv = fabric.inventory().unwrap();
    let free: Vec<u64> = inv
        .iter()
        .map(|r| r.uncommitted_vram / FRAME_SIZE)
        .collect();
    assert_eq!(free, vec![156, 0, 4]);
    assert_eq!(inv[1].tenants.len(), 2, "node b hosts two tenants");

    na.shutdown();
    nb.shutdown();
    nc.shutdown();
}

/// Overflow is a typed, actionable error carrying the numbers a caller
/// needs (queue, shed, or buy hardware).
#[test]
fn placement_overflow_is_typed() {
    let na = spawn_node("a", 64);
    let mut fabric = Fabric::new();
    fabric.add_node(na.addr).unwrap();
    fabric.place(profile(60)).unwrap();

    match fabric.place(profile(32)) {
        Err(FabricError::NoCapacity {
            requested,
            best_available,
        }) => {
            assert_eq!(requested, 32 * FRAME_SIZE);
            assert_eq!(best_available, 4 * FRAME_SIZE);
        }
        other => panic!("expected NoCapacity, got {other:?}"),
    }
    na.shutdown();
}

/// Drain for maintenance: every tenant on the node live-migrates away,
/// with data and in-flight work intact; the drained node ends empty.
/// The fabric-stable TenantId survives while (node, vgpu) changes.
#[test]
fn evacuation_drains_a_node_with_tenants_intact() {
    let (na, nb) = (spawn_node("a", 128), spawn_node("b", 128));
    let mut fabric = Fabric::new();
    let a = fabric.add_node(na.addr).unwrap();
    let b = fabric.add_node(nb.addr).unwrap();

    // Two tenants; steer both onto node a by draining b's capacity
    // first with a placeholder... simpler: place, then check where they
    // landed and evacuate that node.
    let t1 = fabric.place(profile(100)).unwrap(); // slack: a 28 vs b 28 -> a (tie, lower id)
    assert_eq!(t1.node, a);
    let t2 = fabric.place(profile(20)).unwrap(); // a: 28-20=8 slack vs b: 108 -> a
    assert_eq!(t2.node, a);

    // Give tenant 1 state worth preserving: data + a pending fence.
    let mut guest = VgpuClient::connect(t1.addr).unwrap();
    let buf = guest.alloc_memory(t1.vgpu, FRAME_SIZE).unwrap();
    guest.dma_write(t1.vgpu, buf, b"drain me gently").unwrap();
    let ch = guest.create_channel(t1.vgpu).unwrap();
    guest
        .submit(t1.vgpu, ch, Command::FenceSignal { value: 1 })
        .unwrap();

    // Drain node a.
    let moved = fabric.evacuate(a).unwrap();
    assert_eq!(moved, 2);

    // Registry: both tenants now on b, same TenantIds.
    let h1 = fabric.handle(t1.tenant).unwrap();
    let h2 = fabric.handle(t2.tenant).unwrap();
    assert_eq!(h1.node, b);
    assert_eq!(h2.node, b);

    // Node a is empty; node b carries both budgets.
    let inv = fabric.inventory().unwrap();
    assert_eq!(inv[0].uncommitted_vram, 128 * FRAME_SIZE);
    assert_eq!(inv[0].tenants.len(), 0);
    assert_eq!(inv[1].uncommitted_vram, 8 * FRAME_SIZE);

    // Tenant 1's state made the trip; its pending fence completes on b.
    let mut guest = VgpuClient::connect(h1.addr).unwrap();
    assert_eq!(
        guest.dma_read(h1.vgpu, buf, 15).unwrap(),
        b"drain me gently".to_vec()
    );
    guest.tick(10_000).unwrap();
    assert_eq!(guest.fence_value(h1.vgpu, ch).unwrap(), 1);

    na.shutdown();
    nb.shutdown();
}

/// Explicit migration through the fabric API: registry follows the move,
/// stale handles die with the source, and destroy() releases capacity.
#[test]
fn migrate_tenant_and_destroy_keep_the_registry_true() {
    let (na, nb) = (spawn_node("a", 64), spawn_node("b", 64));
    let mut fabric = Fabric::new();
    let a = fabric.add_node(na.addr).unwrap();
    let b = fabric.add_node(nb.addr).unwrap();

    let t = fabric.place(profile(32)).unwrap();
    assert_eq!(t.node, a); // tie -> lower id

    let moved = fabric.migrate_tenant(t.tenant, b).unwrap();
    assert_eq!(moved.node, b);
    assert_eq!(fabric.handle(t.tenant).unwrap().node, b);

    // Migrating to a bogus node is refused before anything happens.
    assert!(matches!(
        fabric.migrate_tenant(t.tenant, NodeId(99)),
        Err(FabricError::NoSuchNode(_))
    ));

    fabric.destroy(t.tenant).unwrap();
    assert!(matches!(
        fabric.handle(t.tenant),
        Err(FabricError::NoSuchTenant(_))
    ));
    let inv = fabric.inventory().unwrap();
    assert!(inv.iter().all(|r| r.uncommitted_vram == 64 * FRAME_SIZE));

    na.shutdown();
    nb.shutdown();
}

/// **Registering a node is idempotent by address.** Two ids for one card
/// would make the fabric believe it has twice the VRAM it has, and
/// capacity that does not exist is worse than no capacity: every
/// downstream decision is computed against a fiction that only surfaces
/// as a mystifying admission failure at the node. Registration is exactly
/// what gets retried by an operator or a config reload, so repeating it
/// must be safe.
#[test]
fn registering_a_node_twice_does_not_invent_capacity() {
    let n = spawn_node("a", 64);
    let mut fabric = Fabric::new();
    let first = fabric.add_node(n.addr).unwrap();
    let again = fabric.add_node(n.addr).unwrap();
    assert_eq!(first, again, "the same address is the same node");

    let inv = fabric.inventory().unwrap();
    assert_eq!(inv.len(), 1, "one card, one entry");
    assert_eq!(inv[0].vram_bytes, 64 * FRAME_SIZE);

    // And the capacity arithmetic stays honest end to end.
    fabric.place(profile(32)).unwrap();
    fabric.place(profile(32)).unwrap();
    assert!(matches!(
        fabric.place(profile(32)),
        Err(FabricError::NoCapacity { .. })
    ));
    n.shutdown();
}

/// **A node's own refusal is not fleet-wide exhaustion.** The fabric
/// reads capacity and *then* admits, so a node can change in between —
/// another operator, a stale registry, a node restarted smaller. Whatever
/// the cause, placement consults the rest of the ranking before declaring
/// the fleet full, and only genuine exhaustion is reported as such.
///
/// (What this test can pin deterministically is the fall-through: the
/// top-ranked candidate cannot host, and the tenant still lands. The
/// concurrent-mutation race that motivates the retry is inherently
/// timing-dependent and is deliberately not simulated here — asserting a
/// race would make this suite flaky, which is worse than testing the
/// property one level down.)
#[test]
fn placement_falls_through_when_the_best_candidate_cannot_host() {
    let (na, nb) = (spawn_node("a", 64), spawn_node("b", 128));
    let mut fabric = Fabric::new();
    let a = fabric.add_node(na.addr).unwrap();
    let b = fabric.add_node(nb.addr).unwrap();

    // Someone outside the fabric consumes node a entirely, behind its back.
    let mut interloper = VgpuClient::connect(na.addr).unwrap();
    interloper.create_vgpu(profile(64)).unwrap();

    // a is the tighter fit on paper but cannot host; b takes the tenant.
    let placed = fabric.place(profile(64)).unwrap();
    assert_eq!(placed.node, b);
    assert_ne!(placed.node, a);

    // b has 64 frames left, so one more fits...
    assert_eq!(fabric.place(profile(64)).unwrap().node, b);
    // ...and now the fleet really is full, reported honestly.
    assert!(matches!(
        fabric.place(profile(64)),
        Err(FabricError::NoCapacity { .. })
    ));
    na.shutdown();
    nb.shutdown();
}
