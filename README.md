# VGPU-X — Software-Defined GPU Virtualization Fabric

**One physical GPU. Many mutually untrusting tenants. No tenant can read
another's memory or starve another's work — and you can read every line of
code that makes that true.**

VGPU-X is a from-first-principles implementation of GPU virtualization,
built as both a working system and a systems-programming text. The core is
a deterministic userspace *device model* of a virtualized GPU: VRAM
management, GPU page tables, command rings, doorbells, fences, and a
weighted fair scheduler — with zero dependencies, so every mechanism in the
repository is implemented here, explained here, and tested here.

```rust
use vgpu_core::prelude::*;

let mut node = GpuNode::new(PhysGpuConfig {
    name: "sim-a".into(), vram_bytes: 256 * FRAME_SIZE, slice_cycles: 1000,
});

// Admit a tenant under a fixed resource contract (a "profile").
let tenant = node.create_vgpu(VgpuProfile {
    name: "sim-2m.1x".into(), vram_bytes: 32 * FRAME_SIZE,
    compute_weight: 1, max_channels: 2, ring_slots: 64,
    ..Default::default()          // no QoS caps: pure proportional share
})?;
node.start_vgpu(tenant)?;

// The classic guest workflow: allocate, upload, compute, fence, download.
let buf = node.alloc_memory(tenant, 4096)?;
node.dma_write(tenant, buf, b"hello, device")?;
let ch = node.create_channel(tenant)?;
node.submit(tenant, ch, Command::KernelLaunch {
    name: "noop".into(), threads: 1, args: vec![],
    program: vgpu_core::isa::busy(500),   // kernels are real programs (M2)
})?;
node.submit(tenant, ch, Command::FenceSignal { value: 1 })?;
node.tick(10_000);
assert_eq!(node.fence_value(tenant, ch)?, 1);
```

The same session, as a *service* — `vgpud` serves the device model over
TCP and `VgpuClient` is the typed client (milestone 1):

```
$ vgpud --listen 127.0.0.1:7677 --vram-mib 1024
vgpud listening on 127.0.0.1:7677
```
```rust
let mut c = VgpuClient::connect("127.0.0.1:7677")?;
let tenant = c.create_vgpu(profile)?;          // same operations,
c.start_vgpu(tenant)?;                          // same errors, same
let buf = c.alloc_memory(tenant, 4096)?;        // isolation — through
c.dma_write(tenant, buf, b"hello, device")?;    // a socket
```

Profiles can also carry a hard ceiling and a guaranteed floor — the two
things a scheduler weight can never express, since weights only bind under
contention:

```rust
VgpuProfile {
    name: "sim-quarter-card".into(), vram_bytes: 32 * FRAME_SIZE,
    compute_weight: 1, max_channels: 2, ring_slots: 64,
    qos: QosLimits {
        max_share_pct: Some(25),  // never more, even on an idle GPU
        min_share_pct: Some(10),  // never less, however crowded it gets
    },
}
```

## Why this project exists

GPU virtualization is one of the most technically dense areas in systems
programming, and one of the worst-documented. The real implementations are
either proprietary firmware (NVIDIA vGPU), retired kernel modules (Intel
GVT-g), or hardware you cannot see into (SR-IOV, MIG). Most engineers who
"know GPU virtualization" are actually pattern-matching on one of five very
different techniques without a clear model of what each layer does.

VGPU-X fixes that by building the whole stack where you can watch it run:

1. **Every mechanism is real.** The buddy allocator coalesces, the page
   walk faults, the ring wraps, the scheduler equalizes virtual runtimes.
   These are the same algorithms production drivers use (`drm_buddy`,
   GMMU radix tables, CFS-style vruntime), not cartoons of them.
2. **Every claim is a test.** "Tenants are isolated" is not a slogan; it is
   [`tenants_are_isolated_by_construction`](crates/vgpu-core/tests/fabric.rs)
   plus the type-system argument in the docs. "Weights are honored" is an
   exact `assert_eq!`, possible because the simulation is deterministic.
3. **Every line is explained.** The `docs/` directory builds the conceptual
   foundation first, then walks the code algorithm by algorithm.

## Reading order

| # | Document | What it teaches |
|---|----------|-----------------|
| 1 | [docs/01-the-landscape.md](docs/01-the-landscape.md) | The five ways to virtualize a GPU, and what each actually virtualizes |
| 2 | [docs/02-how-a-gpu-executes.md](docs/02-how-a-gpu-executes.md) | Rings, doorbells, channels, fences, the GMMU — the machine model everything else assumes |
| 3 | [docs/03-architecture.md](docs/03-architecture.md) | VGPU-X's layers, the two isolation invariants, and the milestone roadmap |
| 4 | [docs/04-walkthrough-memory.md](docs/04-walkthrough-memory.md) | Line-by-line: buddy allocator, page tables, translation, scrubbing |
| 5 | [docs/05-walkthrough-execution.md](docs/05-walkthrough-execution.md) | Line-by-line: rings, the engine, virtual-runtime scheduling, the tick loop |
| 6 | [docs/06-walkthrough-daemon.md](docs/06-walkthrough-daemon.md) | Line-by-line: the wire format, the single-owner device thread, where the wall clock lives |
| 7 | [docs/07-walkthrough-shim.md](docs/07-walkthrough-shim.md) | Line-by-line: the kernel ISA and interpreter, the watchdog (TDR), and the CUDA-shaped guest shim |
| 8 | [docs/08-walkthrough-migration.md](docs/08-walkthrough-migration.md) | Line-by-line: dirty bits, heap-shape replay, and the pre-copy live migration driver |
| 9 | [docs/09-walkthrough-fabric.md](docs/09-walkthrough-fabric.md) | Line-by-line: control vs data plane, best-fit placement, evacuation, and the closing argument |
| 10 | [docs/10-industrial-edge-cases.md](docs/10-industrial-edge-cases.md) | Six defects found by *probing* a system with 85 passing tests, and what generalizes from each |
| 11 | [docs/11-operating-the-fabric.md](docs/11-operating-the-fabric.md) | Telemetry, QoS caps and reservations, and checkpoint/restore/clone — making it runnable |

## Repository map

```
crates/vgpu-core/          the device model (M0)
  src/types.rs             newtype address/ID discipline + the error taxonomy
  src/vram.rs              buddy allocator + sparse, scrubbed backing store
  src/gmmu.rs              per-vGPU two-level page tables (the isolation boundary)
  src/cmd.rs               commands, rings, doorbell semantics, fences, channels
  src/isa.rs               the kernel ISA: 9 instructions, static validation (M2)
  src/metrics.rs           per-tenant and per-node telemetry counters
  src/vgpu.rs              profiles, budgets, lifecycle state machine
  src/sched.rs             vruntime fair scheduler + QoS caps and reservations
  src/engine.rs            translate-then-touch command execution
  src/node.rs              the mediator: admission, DMA, the tick loop
  tests/fabric.rs          multi-tenant isolation & fairness scenarios
crates/vgpu-proto/         the wire protocol (M1)
  src/wire.rs              total codecs + length-prefixed framing (hostile-input safe)
  src/msg.rs               Request/Response vocabulary; lossless VgpuError codec
  src/client.rs            VgpuClient, the blocking typed client
  src/migrate.rs           the pre-copy live migration driver (M3)
  src/checkpoint.rs        checkpoint / restore / clone, composed from those verbs
crates/vgpu-fabric/        the control plane (M4)
  src/lib.rs               best-fit placement, inventory, migration, evacuation
  tests/fabric.rs          a three-node fleet: exact placement, drains, registry
crates/vgpu-shim/          the guest runtime (M2)
  src/lib.rs               CUDA-shaped API: malloc/memcpy/streams/launch/sync
  tests/shim.rs            the CUDA-tutorial flow, faults, watchdog — over TCP
crates/vgpud/              the node daemon (M1)
  src/lib.rs               single-owner device thread + thread-per-connection server
  src/main.rs              the vgpud binary
  tests/daemon.rs          end-to-end over real TCP: isolation, errors, concurrency
  tests/migration.rs       live migration between two daemons (M3)
  tests/hardening.rs       hostile clients, live-guest migration, framing limits
  tests/checkpoint.rs      freeze to bytes, restore, clone across nodes
docs/                      the book
```

## Running it

```
cargo test                    # 112 tests: unit, integration, doctest
cargo run -p vgpud -- --help  # run a node daemon
cargo doc --open              # the API reference is written as part of the text
```

No GPU required — that is the point of a device model. The simulation is
deterministic (no wall clock, no randomness), so every test asserts exact
values, including exact fair-share cycle counts.

## Roadmap

| Milestone | Contents | Status |
|-----------|----------|--------|
| **M0 — Device model** | Memory virtualization, command execution, fair scheduling, isolation | ✅ |
| **M1 — Node daemon** | `vgpud`: the device model behind a wire protocol; concurrency story | ✅ |
| **M2 — Guest shim** | `vgpu-shim`: the CUDA-shaped remoting API + a real kernel ISA, interpreter, and watchdog | ✅ |
| **M3 — Live migration** | Pre-copy migration between nodes: dirty-page tracking, heap-shape replay, channel export | ✅ |
| **M4 — The fabric** | `vgpu-fabric`: best-fit placement, live inventory, tenant migration, node drain | ✅ |

The roadmap is complete: guest app → shim → wire → daemon → device model,
coordinated by a control plane, with live migration between nodes.

Beyond the roadmap, two further passes:

| Pass | Contents | Status |
|------|----------|--------|
| **Hardening** | Six defects found by probing rather than reading — a cross-tenant DoS via fault accounting, an unbounded host allocation from a client-supplied length, live migration that could not survive a live guest, a fence clock that ran backwards, and a fabric that invented capacity ([docs/10](docs/10-industrial-edge-cases.md)) | ✅ |
| **Operability** | Telemetry (counters joined to stable tenant identity), QoS hard caps and reservations, and checkpoint/restore/clone ([docs/11](docs/11-operating-the-fabric.md)) | ✅ |

Where a reader could take it next: TLB shootdowns, copy/compute engine
parallelism, post-copy migration, a C-ABI shim for real `LD_PRELOAD`
interposition, or consensus-backed fabric HA.

## License

MIT — see [LICENSE](LICENSE).
