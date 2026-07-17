# 3. VGPU-X Architecture

## 3.1 The end state

VGPU-X grows into a fabric: a control plane placing tenant workloads onto
a fleet of GPU nodes, each node mediating one or more physical GPUs among
vGPUs, each vGPU driven by an unmodified-looking guest API.

```
                    ┌──────────────────────────┐
                    │   fabric control plane   │   M4: placement, telemetry
                    └───────┬─────────┬────────┘
                            │         │
                  ┌─────────▼──┐   ┌──▼─────────┐
                  │   vgpud    │   │   vgpud    │   M1: node daemon (wire protocol)
                  │  node A    │   │  node B    │
                  ├────────────┤   ├────────────┤
                  │  GpuNode   │   │  GpuNode   │   M0: device model  ← THIS REPO, NOW
                  │ ┌────────┐ │   │            │
                  │ │vGPU 0..n│◀───┼── live migration (M3)
                  │ └────────┘ │   │            │
                  └─────▲──────┘   └────────────┘
                        │
                 ┌──────┴──────┐
                 │   libvgpu   │   M2: guest shim (API remoting into the fabric)
                 │ guest app   │
                 └─────────────┘
```

The milestones are ordered so each layer only ever talks to a layer that
already exists and is tested. The device model comes first because every
other component is either a client of it (shim, daemon) or a coordinator
of it (control plane, migration).

## 3.2 The device model's internal layering

Dependencies point strictly downward; no cycles, no layer reaches around
another:

```
   node.rs      the mediator: admission control, host DMA, the tick loop
      │ owns
      ├─ vgpu.rs     profiles, budgets, lifecycle, per-tenant receipts
      │     │ owns
      │     ├─ gmmu.rs    the per-tenant address space (isolation boundary)
      │     └─ cmd.rs     channels → rings → commands
      ├─ sched.rs    vruntime fair scheduler (knows only IDs and weights)
      ├─ engine.rs   translate-then-touch execution (pure function)
      └─ vram.rs     buddy allocator + scrubbed backing store (physical truth)
                              │
   types.rs     newtypes + errors, used by everyone
```

Three structural decisions deserve explanation:

**The node owns everything physical; vGPUs hold receipts.** `Vgpu`
methods that need physical resources take `&mut VramAllocator` /
`&mut FrameStore` as *parameters* — a vGPU cannot reach VRAM behind the
node's back because it does not contain a reference to it. In a
hypervisor this boundary is privilege levels; here the borrow checker
plays that role, and the API shape documents the mediation.

**The scheduler is deliberately ignorant.** `sched.rs` knows identities,
weights, runnability, and charges — not commands, not memory, not
channels. Fairness logic that is entangled with execution logic can't be
tested exactly, reasoned about separately, or (in M4) reused for
placement decisions across nodes. The narrow interface is the point.

**The engine is a pure-ish function.** `execute(cmd, aspace, store)` has
no access to the node, the scheduler, or other vGPUs. The complete list
of code that can touch physical memory on behalf of a guest is: this
function, and the two DMA helpers in `node.rs`. All three translate
first. That greppable smallness is the security review story.

## 3.3 The two invariants

Everything the README claims about isolation reduces to two invariants,
both enforced structurally rather than by runtime checks:

**I1 — Guests speak only guest virtual addresses.**
Look at the guest-reachable API surface (`GpuNode`'s public methods and
the `Command` variants): every address is a `GpuVirtAddr`. `VramAddr` and
`FrameNum` never cross that surface in either direction. Since the type
system prevents fabricating a translation result, a tenant cannot even
*state* a request about physical memory.

**I2 — Every byte touched is translated through the submitting vGPU's
page tables, and translation failure kills only the faulting channel.**
The engine and DMA helpers call `translate_range` before any
`FrameStore` access; a `PageFault` propagates to `tick()`, which kills
the channel and records the fault. Executable evidence:

| Claim | Test |
|---|---|
| Same numeric VA, different tenants, different data; no cross-tenant path exists | `tenants_are_isolated_by_construction` |
| A wild pointer kills one channel; siblings and other tenants unaffected | `fault_blast_radius_is_one_channel` |
| Freed VRAM is scrubbed before the next tenant can read it | `freed_vram_is_scrubbed_before_reuse` |
| Compute shares follow profile weights exactly under saturation | `compute_shares_follow_profile_weights` |
| Suspend freezes execution and submission; resume completes queued work | `suspend_freezes_execution_and_resume_continues` |

(All in `crates/vgpu-core/tests/fabric.rs`.)

## 3.4 Determinism as a design requirement

The core contains no wall clock and no randomness. Time is a logical
cycle counter advanced by command costs; the buddy allocator always
splits the lowest-addressed block; the scheduler breaks ties by ID.
Consequences, in increasing order of importance:

1. Tests assert *exact* values — fairness is `assert_eq!(h, 12_000)`,
   not "roughly 3:1 with tolerance".
2. Any bug report is a replayable trace.
3. Live migration (M3) fundamentally requires serializing a vGPU's state
   and resuming it elsewhere with identical semantics — a property you
   get for free in a deterministic model and retrofit painfully anywhere
   else.

This mirrors how serious infrastructure is now built (deterministic
simulation testing à la FoundationDB/TigerBeetle), and it is the reason
`Date::now()`-style APIs are absent from the codebase.

## 3.5 What the profiles model

`VgpuProfile` copies the *shape* of NVIDIA's vGPU types (fixed named
bundles like `A100-2-10C`): a hard VRAM budget, a compute weight, channel
limits. Fixed bundles are a fabric-level decision, not a node-level one —
placement across a fleet becomes bin-packing of known shapes instead of a
per-request knapsack problem. The node enforces the *guarantee* side:
admission control refuses profiles whose VRAM budgets would oversubscribe
the card (`create_vgpu`), so an admitted tenant can always allocate to its
budget.

Compute, by contrast, is deliberately work-conserving rather than
reserved: an idle GPU runs whoever has work, and weights only bind under
contention. VRAM = guaranteed partition, compute = proportional share.
That asymmetry matches both what tenants want (memory determinism,
compute throughput) and what NVIDIA's own "best effort" scheduler ships.

## 3.6 Simplifications, and where their fixes attach

Honesty section. Each simplification is a milestone hook, not a hidden
assumption:

| Simplification | Where the fix lands |
|---|---|
| ~~Single-threaded~~ | Landed in M1: `vgpud`'s single-owner device thread serializes all access (the ring stays lock-free by design — see docs/06) |
| ~~Kernels are opaque cycle costs~~ | Landed in M2: `isa.rs` + the interpreter in `engine.rs` route every kernel load/store through the GMMU, with a watchdog for runaways |
| No TLB model → no shootdown protocol | M3 needs unmap-during-migration; the TLB model arrives with the thing that makes it observable |
| Fences poll; no interrupts | M1's wire protocol adds completion notification |
| One execution front-end (no engine parallelism) | Post-M2, copy/compute engine overlap becomes a scheduler dimension |

Next: the code itself, algorithm by algorithm —
[04-walkthrough-memory.md](04-walkthrough-memory.md).
