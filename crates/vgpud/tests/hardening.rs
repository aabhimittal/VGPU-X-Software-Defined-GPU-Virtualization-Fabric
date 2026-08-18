//! Edge cases that only exist once the device is behind a socket and a
//! guest is genuinely *live*. Each test is a bug found by probing the
//! running system; `docs/10-industrial-edge-cases.md` tells the stories.

use std::net::SocketAddr;

use vgpu_core::cmd::Command;
use vgpu_core::node::PhysGpuConfig;
use vgpu_core::sched::QosLimits;
use vgpu_core::types::{VgpuError, FRAME_SIZE, MAX_DMA_BYTES};
use vgpu_core::vgpu::{VgpuProfile, VgpuState};
use vgpu_proto::wire::{write_frame, MAX_FRAME_LEN};
use vgpu_proto::{migrate, ClientError, MigrateOptions, VgpuClient};
use vgpud::{serve, DaemonConfig, ServerHandle};

fn spawn() -> ServerHandle {
    let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    serve(
        addr,
        DaemonConfig {
            gpu: PhysGpuConfig {
                name: "hardened".to_string(),
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
        name: "h".to_string(),
        vram_bytes: frames * FRAME_SIZE,
        compute_weight: 1,
        max_channels: 4,
        ring_slots: 64,
        qos: QosLimits::default(),
    }
}

/// **A client's number must never size the host's memory.** A hostile
/// `DmaRead` length is refused with a typed error, and — the part that
/// matters — the daemon is still serving everyone afterwards. Before the
/// fix the daemon sized a buffer from this number, so a large enough
/// value aborted the process hosting every tenant on the card.
#[test]
fn a_hostile_transfer_length_cannot_kill_the_daemon() {
    let node = spawn();
    let mut hostile = VgpuClient::connect(node.addr).unwrap();
    let mut neighbour = VgpuClient::connect(node.addr).unwrap();

    let victim = neighbour.create_vgpu(profile(8)).unwrap();
    neighbour.start_vgpu(victim).unwrap();
    let safe = neighbour.alloc_memory(victim, FRAME_SIZE).unwrap();
    neighbour.dma_write(victim, safe, b"still here").unwrap();

    let t = hostile.create_vgpu(profile(8)).unwrap();
    hostile.start_vgpu(t).unwrap();
    let buf = hostile.alloc_memory(t, FRAME_SIZE).unwrap();

    // Several flavours of "please allocate the world".
    for len in [MAX_DMA_BYTES + 1, 1 << 40, u64::MAX / 2, u64::MAX] {
        match hostile.dma_read(t, buf, len) {
            Err(ClientError::Device(VgpuError::TransferTooLarge { requested, limit })) => {
                assert_eq!(requested, len);
                assert_eq!(limit, MAX_DMA_BYTES);
            }
            other => panic!("len {len} should be refused, got {other:?}"),
        }
    }

    // The daemon survived, the connection is still usable, and the
    // neighbour's data is untouched.
    assert_eq!(hostile.vgpu_state(t).unwrap(), VgpuState::Running);
    assert_eq!(
        neighbour.dma_read(victim, safe, 10).unwrap(),
        b"still here".to_vec()
    );
    node.shutdown();
}

/// A transfer at exactly the limit is accepted by the size check (and
/// then judged on its merits by the page tables), so the ceiling is a
/// clean boundary rather than an off-by-one.
#[test]
fn transfers_at_the_limit_are_still_served() {
    let node = spawn();
    let mut c = VgpuClient::connect(node.addr).unwrap();
    let t = c.create_vgpu(profile(200)).unwrap();
    c.start_vgpu(t).unwrap();
    let buf = c.alloc_memory(t, MAX_DMA_BYTES).unwrap();

    let payload = vec![0x5Au8; MAX_DMA_BYTES as usize];
    c.dma_write(t, buf, &payload).unwrap();
    let back = c.dma_read(t, buf, MAX_DMA_BYTES).unwrap();
    assert_eq!(back.len(), payload.len());
    assert_eq!(&back[..16], &payload[..16]);
    assert_eq!(back.last(), payload.last());
    node.shutdown();
}

/// **The framing limit is symmetric.** A peer that emits a frame its own
/// reader would reject leaves the stream unrecoverable — the receiver
/// cannot skip a body it refused to size. So the writer refuses too.
#[test]
fn the_framer_refuses_to_emit_what_it_could_not_read() {
    let mut sink = Vec::new();
    let oversized = vec![0u8; MAX_FRAME_LEN as usize + 1];
    assert!(
        write_frame(&mut sink, &oversized).is_err(),
        "writing an unreadable frame must fail loudly"
    );
    assert!(sink.is_empty(), "and must not emit a partial frame");
    // At the limit it still writes: 4-byte prefix + body.
    let ok = vec![0u8; 1024];
    write_frame(&mut sink, &ok).unwrap();
    assert_eq!(sink.len(), 1028);
}

/// **Live migration must tolerate a live guest.** A guest that allocates
/// while pre-copy is running is the normal case, not an exotic one. Before
/// the fix, new pages (born dirty) were handed to a destination that had
/// no such allocation, and the copy page-faulted: calling `malloc` during
/// a migration failed the whole migration.
#[test]
fn migration_survives_a_guest_that_allocates_mid_flight() {
    let (na, nb) = (spawn(), spawn());
    let mut a = VgpuClient::connect(na.addr).unwrap();
    let mut b = VgpuClient::connect(nb.addr).unwrap();

    let t = a.create_vgpu(profile(64)).unwrap();
    a.start_vgpu(t).unwrap();
    let original = a.alloc_memory(t, 20 * FRAME_SIZE).unwrap();
    a.dma_write(t, original, b"present before the move")
        .unwrap();

    // A second connection acts as the guest, allocating and writing while
    // the migration runs — exactly what a live tenant does.
    let mut guest = VgpuClient::connect(na.addr).unwrap();
    let worker = std::thread::spawn(move || {
        let late = guest.alloc_memory(t, 2 * FRAME_SIZE).unwrap();
        guest
            .dma_write(t, late, b"allocated mid-migration")
            .unwrap();
        late
    });
    let late = worker.join().unwrap();

    let twin = migrate(&mut a, &mut b, t, &MigrateOptions::default())
        .expect("a guest allocating during pre-copy must not fail the migration");

    // Both the old and the mid-flight allocation arrived intact.
    assert_eq!(
        b.dma_read(twin, original, 23).unwrap(),
        b"present before the move".to_vec()
    );
    assert_eq!(
        b.dma_read(twin, late, 23).unwrap(),
        b"allocated mid-migration".to_vec()
    );
    assert_eq!(b.vgpu_state(twin).unwrap(), VgpuState::Running);
    na.shutdown();
    nb.shutdown();
}

/// A tenant that is merely *placed* (never started) or already frozen by
/// an operator is still movable. Migration cares that the source cannot
/// change, not how it came to be that way — demanding `Running` would
/// make exactly the tenants an operator is most likely to move unmovable.
#[test]
fn suspended_and_unstarted_tenants_can_be_migrated() {
    for state in ["created", "suspended"] {
        let (na, nb) = (spawn(), spawn());
        let mut a = VgpuClient::connect(na.addr).unwrap();
        let mut b = VgpuClient::connect(nb.addr).unwrap();

        let t = a.create_vgpu(profile(8)).unwrap();
        let buf = if state == "suspended" {
            a.start_vgpu(t).unwrap();
            let buf = a.alloc_memory(t, FRAME_SIZE).unwrap();
            a.dma_write(t, buf, b"frozen by an operator").unwrap();
            a.suspend_vgpu(t).unwrap();
            Some(buf)
        } else {
            None
        };

        let twin = migrate(&mut a, &mut b, t, &MigrateOptions::default())
            .unwrap_or_else(|e| panic!("{state} tenant should migrate: {e}"));
        assert_eq!(b.vgpu_state(twin).unwrap(), VgpuState::Running);
        if let Some(buf) = buf {
            assert_eq!(
                b.dma_read(twin, buf, 21).unwrap(),
                b"frozen by an operator".to_vec()
            );
        }
        na.shutdown();
        nb.shutdown();
    }
}

/// Fence regressions are refused across the wire with the numbers intact,
/// so a guest learns *why* rather than silently corrupting its own
/// completion clock.
#[test]
fn fence_regression_is_refused_over_the_wire() {
    let node = spawn();
    let mut c = VgpuClient::connect(node.addr).unwrap();
    let t = c.create_vgpu(profile(8)).unwrap();
    c.start_vgpu(t).unwrap();
    let ch = c.create_channel(t).unwrap();

    c.submit(t, ch, Command::FenceSignal { value: 7 }).unwrap();
    match c.submit(t, ch, Command::FenceSignal { value: 3 }) {
        Err(ClientError::Device(VgpuError::FenceRegression { last, attempted })) => {
            assert_eq!((last, attempted), (7, 3));
        }
        other => panic!("expected a typed FenceRegression, got {other:?}"),
    }
    // The channel is not poisoned by a rejected submit — it is a refusal,
    // not a fault, so ordinary work continues.
    c.submit(t, ch, Command::FenceSignal { value: 8 }).unwrap();
    c.tick(10_000).unwrap();
    assert_eq!(c.fence_value(t, ch).unwrap(), 8);
    node.shutdown();
}
