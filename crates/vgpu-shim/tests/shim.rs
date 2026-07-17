//! The guest's-eye view, end to end: a daemon on real TCP, and a guest
//! that only ever touches the CUDA-shaped shim API. If milestone 2 works,
//! this file reads like a CUDA tutorial — that is the acceptance test.

use std::net::SocketAddr;

use vgpu_core::isa::{self, Instr};
use vgpu_core::node::PhysGpuConfig;
use vgpu_core::types::FRAME_SIZE;
use vgpu_core::vgpu::VgpuProfile;
use vgpu_shim::{Device, ShimError};
use vgpud::{serve, DaemonConfig, ServerHandle};

fn spawn_daemon() -> ServerHandle {
    let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
    serve(
        addr,
        DaemonConfig {
            gpu: PhysGpuConfig {
                name: "sim-shim".into(),
                vram_bytes: 256 * FRAME_SIZE,
                slice_cycles: 1000,
            },
            auto_tick: None, // the shim drives progress; tests stay deterministic
        },
    )
    .expect("daemon binds")
}

fn profile() -> VgpuProfile {
    VgpuProfile {
        name: "shim-2m".to_string(),
        vram_bytes: 32 * FRAME_SIZE,
        compute_weight: 1,
        max_channels: 4,
        ring_slots: 256,
    }
}

/// The canonical CUDA first program, verbatim in structure:
/// malloc ×3 → memcpy H2D ×2 → launch vector_add → sync → memcpy D2H.
#[test]
fn vector_add_like_a_cuda_tutorial() {
    let daemon = spawn_daemon();
    let mut dev = Device::connect(daemon.addr, profile()).unwrap();

    const N: u64 = 1000;
    let a = dev.malloc(N * 8).unwrap();
    let b = dev.malloc(N * 8).unwrap();
    let c = dev.malloc(N * 8).unwrap();

    let host_a: Vec<u8> = (0..N).flat_map(|i| i.to_le_bytes()).collect();
    let host_b: Vec<u8> = (0..N).flat_map(|i| (7 * i).to_le_bytes()).collect();
    dev.memcpy_htod(a, &host_a).unwrap();
    dev.memcpy_htod(b, &host_b).unwrap();

    let mut stream = dev.stream_create().unwrap();
    dev.launch(
        &stream,
        "vector_add",
        N as u32,
        &[a.0 .0, b.0 .0, c.0 .0],
        isa::vector_add(),
    )
    .unwrap();
    dev.synchronize(&mut stream).unwrap();

    let out = dev.memcpy_dtoh(c, N * 8).unwrap();
    for i in 0..N as usize {
        let got = u64::from_le_bytes(out[i * 8..i * 8 + 8].try_into().unwrap());
        assert_eq!(got, 8 * i as u64, "c[{i}]");
    }

    dev.destroy().unwrap();
    daemon.shutdown();
}

/// Streams are independent lanes: work queued on stream 1 does not gate
/// a sync on stream 2, and each stream's results land correctly.
#[test]
fn two_streams_sync_independently() {
    let daemon = spawn_daemon();
    let mut dev = Device::connect(daemon.addr, profile()).unwrap();

    let buf1 = dev.malloc(4096).unwrap();
    let buf2 = dev.malloc(4096).unwrap();
    let mut s1 = dev.stream_create().unwrap();
    let mut s2 = dev.stream_create().unwrap();

    dev.memset_async(&s1, buf1, 4096, 0x11).unwrap();
    dev.memset_async(&s2, buf2, 4096, 0x22).unwrap();

    // Sync only stream 2, then stream 1 — order must not matter.
    dev.synchronize(&mut s2).unwrap();
    assert_eq!(dev.memcpy_dtoh(buf2, 8).unwrap(), vec![0x22; 8]);
    dev.synchronize(&mut s1).unwrap();
    assert_eq!(dev.memcpy_dtoh(buf1, 8).unwrap(), vec![0x11; 8]);
    daemon.shutdown();
}

/// A wild device pointer kills the stream, not the device: synchronize
/// reports `StreamFaulted`, and a fresh stream on the same device works.
#[test]
fn faulted_stream_is_detected_and_contained() {
    let daemon = spawn_daemon();
    let mut dev = Device::connect(daemon.addr, profile()).unwrap();
    let good = dev.malloc(4096).unwrap();

    let mut bad_stream = dev.stream_create().unwrap();
    // Dereference far outside any mapping — the classic wild pointer.
    let wild = vec![
        Instr::Imm {
            dst: 1,
            value: 0x7_0000_0000,
        },
        Instr::Ld {
            dst: 2,
            addr: 1,
            offset: 0,
        },
        Instr::Halt,
    ];
    dev.launch(&bad_stream, "wild", 1, &[], wild).unwrap();
    match dev.synchronize(&mut bad_stream) {
        Err(ShimError::StreamFaulted) => {}
        other => panic!("expected StreamFaulted, got {other:?}"),
    }

    // The device survives: a new stream does real work.
    let mut ok_stream = dev.stream_create().unwrap();
    dev.memset_async(&ok_stream, good, 64, 0xAB).unwrap();
    dev.synchronize(&mut ok_stream).unwrap();
    assert_eq!(dev.memcpy_dtoh(good, 4).unwrap(), vec![0xAB; 4]);
    daemon.shutdown();
}

/// An infinite-loop kernel is killed by the device watchdog (the TDR
/// story) and surfaces exactly like any other stream fault.
#[test]
fn runaway_kernel_hits_the_watchdog() {
    let daemon = spawn_daemon();
    let mut dev = Device::connect(daemon.addr, profile()).unwrap();

    let mut stream = dev.stream_create().unwrap();
    let forever = vec![
        Instr::Imm { dst: 1, value: 1 },
        Instr::Bnz { cond: 1, target: 1 },
    ];
    dev.launch(&stream, "forever", 1, &[], forever).unwrap();
    match dev.synchronize(&mut stream) {
        Err(ShimError::StreamFaulted) => {}
        other => panic!("expected StreamFaulted from the watchdog, got {other:?}"),
    }
    daemon.shutdown();
}

/// Malformed programs are rejected at the doorbell with a typed error —
/// before they ever occupy the engine.
#[test]
fn bad_program_is_rejected_at_submit() {
    let daemon = spawn_daemon();
    let mut dev = Device::connect(daemon.addr, profile()).unwrap();
    let stream = dev.stream_create().unwrap();

    let bad = vec![Instr::Imm { dst: 99, value: 0 }]; // register out of range
    match dev.launch(&stream, "bad", 1, &[], bad) {
        Err(ShimError::Client(vgpu_proto::ClientError::Device(
            vgpu_core::types::VgpuError::BadProgram { .. },
        ))) => {}
        other => panic!("expected typed BadProgram, got {other:?}"),
    }
    daemon.shutdown();
}

/// Budget errors surface through the shim as the same typed error a
/// local caller sees — `cudaMalloc` returning `cudaErrorMemoryAllocation`,
/// with actual numbers attached.
#[test]
fn malloc_over_budget_is_a_typed_error() {
    let daemon = spawn_daemon();
    let mut dev = Device::connect(daemon.addr, profile()).unwrap();
    match dev.malloc(33 * FRAME_SIZE) {
        Err(ShimError::Client(vgpu_proto::ClientError::Device(
            vgpu_core::types::VgpuError::VramBudgetExceeded { budget_left, .. },
        ))) => assert_eq!(budget_left, 32 * FRAME_SIZE),
        other => panic!("expected VramBudgetExceeded, got {other:?}"),
    }
    daemon.shutdown();
}
