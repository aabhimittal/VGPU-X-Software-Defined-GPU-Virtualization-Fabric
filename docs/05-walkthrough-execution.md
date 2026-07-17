# 5. Walkthrough: The Execution Path, Line by Line

This chapter reads `cmd.rs`, `engine.rs`, `sched.rs`, and `node.rs` —
the path a command travels from a guest's `submit` to bytes moving in
VRAM, and the scheduler that decides *when* it travels.

## 5.1 `cmd.rs`: the ring

```rust
pub struct Ring {
    slots: Vec<Option<Command>>,
    head: usize,   // next slot the device will consume
    tail: usize,   // next slot the guest will fill
}
```

### The one-slot-empty convention

```rust
pub fn push(&mut self, cmd: Command) -> Result<()> {
    let next = (self.tail + 1) % self.slots.len();
    if next == self.head {
        return Err(VgpuError::RingFull);
    }
    ...
}
```

A circular buffer where both indices chase each other has an ambiguity:
`head == tail` could mean empty *or* full. The classic resolution, used
here and in countless hardware rings: declare `head == tail` to mean
empty, and refuse the push that would make `tail` catch `head`. Cost: one
permanently unusable slot. Benefit: no extra state, and every arithmetic
line is verifiable by eye. The `len()` formula
`(tail + N - head) % N` is the standard wrap-safe distance in ring space.

Two deliberate choices worth pausing on:

* **`push` fails rather than blocks.** Whether to spin, sleep, or drop on
  a full ring is a *client* policy — a latency-sensitive shim and a batch
  submitter want different answers. The mechanism layer returns
  `RingFull` and stays policy-free.
* **Plain integers, not atomics.** The foundation is single-threaded by
  design; dressing the ring in `AtomicUsize` now would *suggest*
  cross-thread guarantees the type does not actually keep (there is no
  memory-ordering story until M1's daemon defines one). Honest types over
  impressive types.

### Channels and `kill`

```rust
pub(crate) fn kill(&mut self, fault: VgpuError) {
    self.state = ChannelState::Faulted;
    self.fault = Some(fault);
    while self.ring.pop().is_some() {}
}
```

A faulted channel drains its queue. Reasoning: commands after the fault
were ordered *behind* a command that never completed; executing them
would present the guest with an execution order it cannot reason about.
Real drivers do the same on robust-channel recovery — the channel dies,
queued work dies with it, the guest recreates from a known state. `kill`
is `pub(crate)`: only the mediator may declare a fault, never the guest.

## 5.2 `engine.rs`: translate, then touch

The whole module is one function with one shape per command:

```rust
Command::MemFill { dst, len, value } => {
    let segs = aspace.translate_range(*dst, *len, AccessKind::Write)?;   // ALL translation first
    for (pa, seg_len) in segs {
        store.fill(pa, seg_len as usize, *value);                        // then ALL touching
    }
}
```

The `?` sits *between* translation and touching. If any byte of the range
is unmapped, the fault propagates before byte one is written —
`fill_faults_atomically_on_unmapped_tail` asserts a fill that runs one
byte off the end of a mapping writes *nothing*. Partial writes on fault
would leave guest memory in a state the guest cannot reason about
(compare: hardware copy engines validate the page range up front too).

`MemCopy` stages through a host buffer rather than walking source and
destination segment lists in lockstep:

```rust
let src_segs = aspace.translate_range(*src, *len, AccessKind::Read)?;
let dst_segs = aspace.translate_range(*dst, *len, AccessKind::Write)?;
// read all src segs into `data`, then write `data` across dst segs
```

Lockstep streaming is what real engines do, but it must handle the src
and dst page phases being different (`src` at offset 500 in its page,
`dst` at offset 0) — a two-pointer merge with fiddly boundary cases.
Staging is trivially correct for that *and* for overlapping ranges, and
correctness is the foundation's optimization target. The comment in the
code says exactly this, so the future optimizer knows the contract.

Note what `execute` *cannot* do: it receives one `&AddressSpace` — the
submitting vGPU's — and has no access to the node, other tenants, or the
allocator. Isolation invariant I2 is enforced by a function signature.

## 5.3 `sched.rs`: virtual runtime

### The idea, derived rather than asserted

Goal: tenant i with weight `w_i` gets `w_i / Σw` of the cycles, over any
saturated window. Trick: give each tenant a private clock that runs at
speed `1/w_i`:

```rust
acc.vruntime += cycles as u128 * SCALE / acc.weight as u128;
```

and always run the tenant whose clock is furthest behind:

```rust
.filter(|(_, a)| a.runnable)
.min_by_key(|(id, a)| (a.vruntime, **id))
```

If the scheduler succeeds at equalizing vruntimes, then for any two
tenants `c_a / w_a ≈ c_b / w_b` — which *is* the proportional-share
property, rearranged. Fairness isn't computed anywhere; it is the fixed
point the argmin rule converges to. This is Linux CFS's core idea
(`SCALE` here plays `NICE_0_LOAD = 1024`, keeping integer division
precise), and NVIDIA's vGPU best-effort scheduler approximates the same
contract in firmware.

`vruntime` is `u128` because the worst case (`u64::MAX` cycles × 1024)
overflows u64 — the kind of bound worth writing down once in a comment
and never thinking about again.

### The sleeper clamp, or: fairness is not a savings account

```rust
if runnable && !acc.runnable {
    acc.vruntime = acc.vruntime.max(self.min_vruntime);
}
```

Without this, a tenant that sleeps for a million cycles keeps its ancient
(tiny) vruntime and, on waking, monopolizes the GPU until it "catches
up" — historical credit converted into a future monopoly, which is
starvation for everyone else. The fix (CFS's fix): fairness applies only
to the *runnable*. On the false→true edge, clamp the waker's clock up to
the high-water mark (`min_vruntime`, advanced monotonically in `pick`).
It competes fairly from *now*; it does not collect back-pay.
`sleeper_does_not_monopolize_on_wake` pins the behavior, and
`register()` applies the same clamp to late joiners for the same reason.

### Determinism, again

Ties break by `(vruntime, VgpuId)` — arbitrary but *stable*. A replayed
trace schedules identically, so a fairness bug is a reproducible test
case rather than a heisenbug.

## 5.4 `node.rs`: the tick loop

`tick(budget)` is the mediator's whole job in twenty lines. Annotated:

```rust
while report.cycles < budget {
    // (1) recompute runnability — rings may have drained last slice
    for (id, vgpu) in &self.vgpus {
        self.sched.set_runnable(*id, vgpu.has_pending_work());
    }
    // (2) whom does fairness owe the next slice?
    let Some(id) = self.sched.pick() else { break };

    // (3) drain up to one slice from this vGPU
    let slice_target = self.config.slice_cycles.min(budget - report.cycles);
    let mut slice_used = 0;
    while slice_used < slice_target {
        let Some((ch, cmd)) = self.pop_round_robin(id) else { break };
        match execute(&cmd, &vgpu.aspace, &mut self.store) {
            Ok(cost) => { /* fence bookkeeping */ slice_used += cost; }
            Err(error) => { vgpu.channels[ch].kill(error); slice_used += cmd.cost(); }
        }
    }
    // (4) charge reality, not the target
    self.sched.charge(id, slice_used);
    report.cycles += slice_used;
}
```

Line-level reasoning:

* **(1) Runnability is recomputed every iteration**, not cached. A vGPU
  whose last command just drained must stop being picked *now*; the cost
  is a linear scan, and correctness-first is the right default until a
  profiler says otherwise.
* **(2) `pick` can return `None`** — all rings empty — and the loop exits
  early. `tick_with_no_work_is_a_clean_noop` asserts an idle GPU consumes
  zero cycles: work-conserving, never busy-waiting.
* **(3) Commands are not preempted mid-execution.** `slice_used` can
  overshoot `slice_target` by up to one command — modeling
  command-boundary preemption granularity (§2.6). The slice is a target,
  not a guillotine.
* **(4) The overrun is charged.** Because vruntime accounts *actual*
  cycles, a tenant that overran its slice is picked correspondingly later
  next time; the fairness error from any single overrun decays
  automatically instead of accumulating. This one line is why
  `compute_shares_follow_profile_weights` can assert an *exact* 3:1 split
  even with command-granular preemption.
* **The defensive `slice_used == 0` branch** marks a picked-but-workless
  vGPU unrunnable and continues — unreachable given (1), but a scheduler
  loop's failure mode is spinning forever, and a belt-and-suspenders exit
  costs two lines.
* **Faults consume the faulting command's cost.** The engine was occupied
  until the fault was recognized; charging it keeps even *misbehavior*
  inside the fairness accounting (a tenant cannot get free scheduling
  passes by faulting).

`pop_round_robin` rotates a cursor across the vGPU's channels so one
busy channel cannot starve its siblings *within* the tenant's own slice —
the same fairness idea, one level down, solved with the simpler tool
(round-robin) because channels within a tenant have no weights.

Admission control (`create_vgpu`) closes the resource story: profiles'
VRAM budgets are checked against *uncommitted* capacity, so the sum of
promises never exceeds the card. The budget guarantees frames *exist*;
buddy fragmentation can still fail a specific large contiguous request —
a distinction the docs of `create_vgpu` spell out, because conflating
"capacity" with "contiguity" is a classic operator confusion.

## 5.5 The complete journey of one command

Tie it all together — `MemCopy` from submission to completion:

1. Guest calls `node.submit(id, ch, MemCopy{..})` — the doorbell. The
   command (guest VAs only, invariant I1) lands in the channel's ring.
2. Some later `tick`: runnability refresh marks the vGPU runnable;
   `pick()` selects it when its vruntime is the minimum.
3. `pop_round_robin` pulls the command; `execute` translates src and dst
   ranges through *this vGPU's* page tables into scatter-gather segments
   (invariant I2), stages the bytes, writes them out.
4. Cost `1 + len/8` cycles is added to `slice_used`; eventually the slice
   ends and `charge` advances the tenant's vruntime.
5. The guest's following `FenceSignal` executes in order; the channel's
   `completed_fence` rises; the guest's poll (`fence_value`) observes
   completion.
6. If step 3 had faulted: channel killed, queue drained, fault recorded
   in the `TickReport`, every other channel and tenant untouched.

That is the device model. Milestone 1 puts a wire protocol in front of
`GpuNode`; nothing in this chapter changes.
