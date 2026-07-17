//! End-to-end tests: real daemon, real TCP sockets, real concurrent
//! clients. The milestone-0 isolation claims are re-proven here *through
//! the network boundary* — if the daemon layer had opened a hole (an
//! operation that bypasses per-vGPU confinement), these are the tests
//! that would catch it.
//!
//! All daemons run with `auto_tick: None` and time is driven by explicit
//! `Tick` requests, so every assertion is exact — the determinism story
//! survives the jump to a multi-threaded server because the device
//! thread serializes everything anyway.

use std::net::SocketAddr;
use std::time::Duration;

use vgpu_core::cmd::Command;
use vgpu_core::node::PhysGpuConfig;
use vgpu_core::types::{GpuVirtAddr, VgpuError, FRAME_SIZE};
use vgpu_core::vgpu::{VgpuProfile, VgpuState};
use vgpu_proto::{ClientError, VgpuClient};
use vgpud::{serve, AutoTick, DaemonConfig, ServerHandle};

fn spawn_daemon(auto_tick: Option<AutoTick>) -> ServerHandle {
    let addr: SocketAddr = "127.0.0.1:0".parse().unwrap(); // ephemeral port
    serve(
        addr,
        DaemonConfig {
            gpu: PhysGpuConfig {
                name: "sim-test".into(),
                vram_bytes: 256 * FRAME_SIZE,
                slice_cycles: 100,
            },
            auto_tick,
        },
    )
    .expect("daemon binds")
}

fn profile(name: &str, frames: u64, weight: u32) -> VgpuProfile {
    VgpuProfile {
        name: name.to_string(),
        vram_bytes: frames * FRAME_SIZE,
        compute_weight: weight,
        max_channels: 4,
        ring_slots: 256,
    }
}

/// The full guest workflow, over the wire: create → start → alloc →
/// upload → copy → fence → download → destroy.
#[test]
fn end_to_end_session_over_tcp() {
    let daemon = spawn_daemon(None);
    let mut c = VgpuClient::connect(daemon.addr).unwrap();

    let info = c.node_info().unwrap();
    assert_eq!(info.name, "sim-test");
    assert_eq!(info.vram_bytes, 256 * FRAME_SIZE);

    let t = c.create_vgpu(profile("t", 32, 1)).unwrap();
    c.start_vgpu(t).unwrap();
    assert_eq!(c.vgpu_state(t).unwrap(), VgpuState::Running);

    let src = c.alloc_memory(t, 2 * FRAME_SIZE).unwrap();
    let dst = c.alloc_memory(t, 2 * FRAME_SIZE).unwrap();
    let payload: Vec<u8> = (0..70_000).map(|i| (i % 251) as u8).collect();
    c.dma_write(t, src, &payload).unwrap();

    let ch = c.create_channel(t).unwrap();
    c.submit(
        t,
        ch,
        Command::MemCopy {
            src,
            dst,
            len: payload.len() as u64,
        },
    )
    .unwrap();
    c.submit(t, ch, Command::FenceSignal { value: 1 }).unwrap();

    let report = c.tick(1_000_000).unwrap();
    assert!(report.faults.is_empty());
    assert!(report.commands >= 2);
    assert_eq!(c.fence_value(t, ch).unwrap(), 1);

    let out = c.dma_read(t, dst, payload.len() as u64).unwrap();
    assert_eq!(out, payload);

    c.destroy_vgpu(t).unwrap();
    assert_eq!(c.node_info().unwrap().uncommitted_vram, 256 * FRAME_SIZE);
    daemon.shutdown();
}

/// Device errors cross the wire with full fidelity: a page fault comes
/// back as the same typed `VgpuError` a local call would return.
#[test]
fn device_errors_survive_the_wire() {
    let daemon = spawn_daemon(None);
    let mut c = VgpuClient::connect(daemon.addr).unwrap();

    let t = c.create_vgpu(profile("t", 8, 1)).unwrap();
    c.start_vgpu(t).unwrap();

    // DMA into unmapped memory → typed PageFault, not a stringly error.
    let err = c.dma_write(t, GpuVirtAddr(0), b"x").unwrap_err();
    match err {
        ClientError::Device(VgpuError::PageFault { addr, .. }) => {
            assert_eq!(addr, GpuVirtAddr(0));
        }
        other => panic!("expected typed PageFault, got {other:?}"),
    }

    // Budget violation carries its numbers across intact.
    let err = c.alloc_memory(t, 9 * FRAME_SIZE).unwrap_err();
    match err {
        ClientError::Device(VgpuError::VramBudgetExceeded {
            requested,
            budget_left,
        }) => {
            assert_eq!(requested, 9 * FRAME_SIZE);
            assert_eq!(budget_left, 8 * FRAME_SIZE);
        }
        other => panic!("expected VramBudgetExceeded, got {other:?}"),
    }
    daemon.shutdown();
}

/// Two tenants on two *concurrent connections*: the isolation claims from
/// milestone 0, now with the requests racing through real sockets. The
/// device thread serializes them; neither tenant can observe the other's
/// data, and both make progress.
#[test]
fn concurrent_tenants_stay_isolated() {
    let daemon = spawn_daemon(None);
    let addr = daemon.addr;

    // Admit both tenants up front on a control connection so the worker
    // threads race only on the data path.
    let mut control = VgpuClient::connect(addr).unwrap();
    let ta = control.create_vgpu(profile("a", 32, 1)).unwrap();
    let tb = control.create_vgpu(profile("b", 32, 1)).unwrap();
    control.start_vgpu(ta).unwrap();
    control.start_vgpu(tb).unwrap();

    let worker = |tenant, fill_byte: u8| {
        std::thread::spawn(move || {
            let mut c = VgpuClient::connect(addr).unwrap();
            let buf = c.alloc_memory(tenant, FRAME_SIZE).unwrap();
            let ch = c.create_channel(tenant).unwrap();
            // Interleave many small operations to maximize interleaving
            // through the device thread's queue.
            for round in 1..=50u64 {
                c.submit(
                    tenant,
                    ch,
                    Command::MemFill {
                        dst: buf,
                        len: 64,
                        value: fill_byte,
                    },
                )
                .unwrap();
                c.submit(tenant, ch, Command::FenceSignal { value: round })
                    .unwrap();
            }
            (c, tenant, buf, ch)
        })
    };

    let ha = worker(ta, 0xAA);
    let hb = worker(tb, 0xBB);
    let (mut ca, ta, buf_a, ch_a) = ha.join().unwrap();
    let (mut cb, tb, buf_b, ch_b) = hb.join().unwrap();

    // Both tenants allocated at the same guest VA (deterministic heap
    // base) — the classic isolation setup.
    assert_eq!(buf_a, buf_b);

    // Drive the GPU until both queues drain.
    control.tick(10_000_000).unwrap();
    assert_eq!(ca.fence_value(ta, ch_a).unwrap(), 50);
    assert_eq!(cb.fence_value(tb, ch_b).unwrap(), 50);

    // Same VA, different bytes: isolation held under concurrency.
    assert_eq!(ca.dma_read(ta, buf_a, 4).unwrap(), vec![0xAA; 4]);
    assert_eq!(cb.dma_read(tb, buf_b, 4).unwrap(), vec![0xBB; 4]);
    daemon.shutdown();
}

/// With auto-tick enabled, the GPU makes progress with no client driving
/// it — the daemon's own timer runs the mediation loop. (The one
/// deliberately non-deterministic test: it asserts progress, not exact
/// cycle counts.)
#[test]
fn auto_tick_drives_the_gpu() {
    let daemon = spawn_daemon(Some(AutoTick {
        interval: Duration::from_millis(1),
        budget: 100_000,
    }));
    let mut c = VgpuClient::connect(daemon.addr).unwrap();

    let t = c.create_vgpu(profile("t", 8, 1)).unwrap();
    c.start_vgpu(t).unwrap();
    let ch = c.create_channel(t).unwrap();
    c.submit(
        t,
        ch,
        Command::KernelLaunch {
            name: "bg".into(),
            cost: 10,
        },
    )
    .unwrap();
    c.submit(t, ch, Command::FenceSignal { value: 1 }).unwrap();

    // Poll the fence; the auto-ticker should complete the work without
    // any explicit Tick request. Bounded wait so a regression fails the
    // test instead of hanging it.
    let deadline = std::time::Instant::now() + Duration::from_secs(5);
    loop {
        if c.fence_value(t, ch).unwrap() == 1 {
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "auto-tick made no progress within 5s"
        );
        std::thread::sleep(Duration::from_millis(2));
    }
    daemon.shutdown();
}

/// A client that vanishes mid-session must not take the daemon down;
/// later clients get normal service.
#[test]
fn daemon_survives_client_disconnects() {
    let daemon = spawn_daemon(None);

    for _ in 0..5 {
        let mut c = VgpuClient::connect(daemon.addr).unwrap();
        let _ = c.node_info().unwrap();
        drop(c); // abrupt hang-up
    }

    let mut c = VgpuClient::connect(daemon.addr).unwrap();
    assert_eq!(c.node_info().unwrap().name, "sim-test");
    daemon.shutdown();
}
