# 7. Walkthrough: Kernels and the Guest Shim (Milestone 2)

Milestone 2 closes the two gaps milestone 1 left open. Kernels were
opaque cycle costs — the claim "everything a kernel touches goes through
the GMMU" was true only vacuously, because kernels touched nothing. And
there was no *guest-facing* layer: applications would have had to speak
raw `VgpuClient` calls. This chapter reads the new pieces: the ISA
(`isa.rs`), the interpreter (`engine.rs`), the protocol's first version
bump, and `vgpu-shim`.

## 7.1 The ISA: nine instructions, three constraints

`isa.rs` defines a teaching ISA in the spirit of PTX/SASS reduced to
essentials: `Imm`, `Mov`, `Add`, `Sub`, `Mul`, `Ld`, `St`, `Bnz`, `Halt`.
The instruction *set* matters less than the three constraints it was
chosen under:

1. **Memory access is only expressible as a guest VA.** Registers hold
   plain u64s; only `Ld`/`St` interpret one as an address, and the
   interpreter immediately translates it. The ISA cannot spell a physical
   address — isolation invariant I1 extends into kernel code by
   construction, not by checking.
2. **Deterministic cost.** Every instruction has a fixed price (ALU = 1,
   memory = 4 — memory is slower than arithmetic everywhere, and the
   visible ratio makes kernel costs teach what real profilers teach).
   `busy(n)` builds a program costing *exactly* n cycles, which is how
   the scheduler fairness tests stayed `assert_eq!`-exact after kernels
   became real programs.
3. **Verifiable by eye.** Registers fit in a `u8` index, branches are
   absolute `u16` targets, and there is exactly one control-flow
   instruction. Warps, shared memory, predication: absent until a
   milestone needs them.

The execution model collapses CUDA's grid to one dimension: `threads`
copies of the program, `r0 = thread index`, `r1.. = kernel args`.
Threads run *sequentially* — for data-race-free kernels (the only ones
with defined results on real hardware either) this is observably
equivalent to the parallel schedule, and it keeps results and cycle
accounting deterministic.

## 7.2 Static vs dynamic checking: where each rule lives

The split is a small case study in systems design:

* **Statically checkable → validated at the doorbell.** Register bounds,
  branch targets, argument count (`isa::validate`, called from
  `Vgpu::submit`). A malformed program is a typed `BadProgram` error to
  the *submitter*, before the program ever occupies the engine — and the
  interpreter's hot loop indexes registers unchecked, safely.
* **Dynamically checkable only → handled per event.** Memory addresses
  are data-dependent, so every `Ld`/`St` takes the page walk (and 8-byte
  misalignment faults, as on real hardware — an aligned u64 can never
  straddle a 64 KiB frame, which is what lets a kernel access be one
  translate + one store access).
* **Not checkable at all → bounded by force.** Termination is
  undecidable, so the watchdog (`WATCHDOG_INSTRUCTIONS`) kills any
  launch that exceeds its instruction budget: `KernelTimeout`, channel
  dead, node fine. This is the device model's TDR — Windows' Timeout
  Detection & Recovery, the driver watchdogs on Linux — and the reason
  is the same everywhere: *a guest cannot be trusted to terminate its
  own infinite loop; the device must.*

## 7.3 Fault atomicity: kernels are not transactions

Fills and copies validate their entire range before touching a byte — a
fault writes nothing (`fill_faults_atomically_on_unmapped_tail`).
Kernels are deliberately the opposite: a program that faults at
instruction 40 keeps the stores its first 39 instructions made
(`kernel_wild_pointer_faults_but_keeps_prior_stores`). That asymmetry is
faithful — hardware copy engines can validate up front because the
access pattern is known; kernels' access patterns are data-dependent, so
real GPUs fault them mid-flight too. The channel dies either way; what
differs is what the guest's memory looks like afterwards.

This forced one honest refactor: `execute` now returns
`ExecOutcome { cycles, result }` instead of `Result<Cycles>`, because a
faulting kernel has a *partial* cost that must still be charged. The
scheduler charges reality — a tenant cannot get cheap scheduling passes
by faulting — and `watchdog_kills_infinite_loops` asserts the runaway
kernel was billed for every cycle it burned.

## 7.4 The protocol's first real version bump

`KernelLaunch` changed shape on the wire (name+cost → name, threads,
args, program), so `VERSION` went 1 → 2 and v1 peers are now refused
with `VersionMismatch`. This is the versioning policy doing its job:
the alternative — decoding v1 bytes with v2 field order — would misparse
*silently*, and silent misparse on a device-control channel is how
another tenant's day gets ruined. Two error variants (`BadProgram`,
`KernelTimeout`) also joined the closed set; match exhaustiveness forced
codec arms for both at compile time, and `error_roundtrips_losslessly`
pins them — the flat-enum discipline from milestone 0 paying rent again.

## 7.5 `vgpu-shim`: API remoting lands

The shim is technique 1 from the landscape (API remoting, the rCUDA
idea) composed on technique 3 (the mediated device model). A guest
program links against a CUDA-*shaped* API and never sees the wire:

| CUDA | shim | becomes |
|---|---|---|
| context | `Device::connect` | create + start a vGPU |
| `cudaMalloc` | `Device::malloc` | budgeted alloc + GMMU map |
| stream | `Stream` | a channel |
| launch | `Device::launch` | `KernelLaunch` on the ring |
| `cudaStreamSynchronize` | `Device::synchronize` | the fence protocol |

The one algorithm worth reading closely is `synchronize`:

```rust
let target = stream.next_fence;      // 1. claim a fence number
stream.next_fence += 1;
self.submit(stream, FenceSignal { value: target })?;   // 2. fence the ring
loop {
    if fence_value(..)? >= target { return Ok(()); }   // 3. landed?
    let report = self.client.tick(SYNC_TICK_BUDGET)?;  // 4. drive the GPU
    if report.cycles == 0 { /* dead stream — see below */ }
}
```

Three properties, each doing real work:

* **Correctness comes from ring FIFO order.** The fence completes only
  after everything submitted before it — so "fence ≥ target" *is*
  "everything before this sync is done". No per-command tracking.
* **Progress is client-driven.** The waiting guest asks the node to run
  (`tick`) rather than depending on the daemon's optional wall-clock
  auto-tick. Tests stay deterministic; a node with no auto-tick still
  serves guests perfectly.
* **Dead streams are detected, not spun on.** If a tick runs zero cycles
  while our fence hasn't landed, the queue drained without reaching the
  fence — on an in-order ring only a killed channel does that. The shim
  returns `StreamFaulted`; the device and its other streams keep
  working (`faulted_stream_is_detected_and_contained`).

What the shim deliberately is *not*, yet: a binary-compatible
`libcudart.so`. Real interposition is this same API surface exported
with C ABI symbols and loaded via `LD_PRELOAD` — mechanical work on top
of what exists, and it belongs to the milestone where a real workload
demands it. The interesting decisions (streams→channels, sync→fences,
errors as typed values end to end) are all here already.

## 7.6 What the tests prove

`crates/vgpu-shim/tests/shim.rs` — each a claim, over real TCP:

| Claim | Test |
|---|---|
| The canonical CUDA tutorial flow works verbatim in structure | `vector_add_like_a_cuda_tutorial` |
| Streams sync independently, in any order | `two_streams_sync_independently` |
| A wild pointer kills one stream; the device survives | `faulted_stream_is_detected_and_contained` |
| Infinite loops die by watchdog and surface as stream faults | `runaway_kernel_hits_the_watchdog` |
| Malformed programs are rejected at submit, typed | `bad_program_is_rejected_at_submit` |
| Budget violations arrive with their numbers intact | `malloc_over_budget_is_a_typed_error` |

And in the core, the interpreter's own suite: `vector_add` computing
through scattered frames, exact loop costs, watchdog billing, misaligned
access faults.

Next (M3): live migration — suspend, serialize, resume on another node.
The state machine has been waiting for it since milestone 0.
