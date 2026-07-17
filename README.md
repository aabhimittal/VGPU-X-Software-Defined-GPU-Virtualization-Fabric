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
    name: "sim-a", vram_bytes: 256 * FRAME_SIZE, slice_cycles: 1000,
});

// Admit a tenant under a fixed resource contract (a "profile").
let tenant = node.create_vgpu(VgpuProfile {
    name: "sim-2m.1x", vram_bytes: 32 * FRAME_SIZE,
    compute_weight: 1, max_channels: 2, ring_slots: 64,
})?;
node.start_vgpu(tenant)?;

// The classic guest workflow: allocate, upload, compute, fence, download.
let buf = node.alloc_memory(tenant, 4096)?;
node.dma_write(tenant, buf, b"hello, device")?;
let ch = node.create_channel(tenant)?;
node.submit(tenant, ch, Command::KernelLaunch { name: "noop", cost: 500 })?;
node.submit(tenant, ch, Command::FenceSignal { value: 1 })?;
node.tick(10_000);
assert_eq!(node.fence_value(tenant, ch)?, 1);
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

## Repository map

```
crates/vgpu-core/          the device model (this milestone)
  src/types.rs             newtype address/ID discipline + the error taxonomy
  src/vram.rs              buddy allocator + sparse, scrubbed backing store
  src/gmmu.rs              per-vGPU two-level page tables (the isolation boundary)
  src/cmd.rs               commands, rings, doorbell semantics, fences, channels
  src/vgpu.rs              profiles, budgets, lifecycle state machine
  src/sched.rs             weighted virtual-runtime fair scheduler
  src/engine.rs            translate-then-touch command execution
  src/node.rs              the mediator: admission, DMA, the tick loop
  tests/fabric.rs          multi-tenant isolation & fairness scenarios
docs/                      the book
```

## Running it

```
cargo test          # 43 tests: unit, integration, doctest
cargo doc --open    # the API reference is written as part of the text
```

No GPU required — that is the point of a device model. The simulation is
deterministic (no wall clock, no randomness), so every test asserts exact
values, including exact fair-share cycle counts.

## Roadmap

| Milestone | Contents | Status |
|-----------|----------|--------|
| **M0 — Device model** (this) | Memory virtualization, command execution, fair scheduling, isolation | ✅ |
| M1 — Node daemon | `vgpud`: the device model behind a wire protocol; concurrency story | ⏳ |
| M2 — Guest shim | `libvgpu`: API interception (the rCUDA/API-remoting layer) + a kernel interpreter | ⏳ |
| M3 — Live migration | Suspend/copy/resume of a vGPU between nodes; dirty-page tracking | ⏳ |
| M4 — The fabric | Multi-node control plane: placement, profiles-as-Tetris, telemetry | ⏳ |

## License

MIT — see [LICENSE](LICENSE).
