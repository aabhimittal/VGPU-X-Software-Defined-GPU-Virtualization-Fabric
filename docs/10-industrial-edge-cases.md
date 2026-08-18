# 10. Industrial Edge Cases: What Probing Found

The first five milestones were built test-first, and every test passed.
Then the system was *probed* — driven the way a hostile client, a
distracted operator, and an ordinary busy guest would drive it — and six
real defects fell out in an afternoon. Two of them broke claims this
repository makes in its README.

That gap is the lesson, so this chapter is organized around it rather
than around the fixes. **Tests written alongside a feature encode what
the author expected to happen. They cannot encode what the author did
not think of.** Every bug below was invisible to a suite that already
covered isolation, fairness, faults, migration, and placement — because
each one lives in a case the author never imagined, and the author wrote
the tests.

## 10.1 The one that mattered: a cheap fault stole the whole GPU

**The probe.** One tenant submits a single command:

```rust
Command::MemFill { dst: GpuVirtAddr(0), len: u64::MAX / 2, value: 1 }
```

Nothing is mapped at VA 0, so this faults on translation and does *zero
work*. A neighbouring tenant has ten ordinary fences queued. Then
`tick(10_000)`:

```
tick(10_000): cycles=576460752303423488 commands=0 faults=1
charged to faulting tenant : 576460752303423488
delivered to good neighbour: 0
```

The neighbour got **nothing**. One malformed command, instantly refused,
consumed the entire scheduling window.

**Why.** The tick loop charged a faulting command its *nominal* cost —
what the command would have cost had it run. That number is derived
directly from a length the guest chose, and `u64::MAX / 2` bytes prices
out at 5.7×10¹⁷ cycles. The budget blew instantly and the loop exited
before anyone else was scheduled.

**Why the existing tests missed it.** `fault_blast_radius_is_one_channel`
already proved a fault damages only the faulting channel — and it does.
The blast radius was never the problem. The *bill* was. The suite
verified the correctness of the fault path and never asked what the fault
path costs, because a fault costing anything at all is not an obvious
thought.

**The fix.** A command that dies in validation is charged `FAULT_COST` —
a page walk — because that is what it did. The general rule, which is
worth more than the fix:

> In a multi-tenant system, any quantity derived from an untrusted
> request must be bounded by work actually performed, or it becomes a
> weapon.

Note what shape the vulnerability had: no memory was disclosed, no
isolation boundary was crossed, nothing crashed. The attacker simply got
the *accounting* to lie on their behalf. Resource-accounting bugs are
security bugs, and they are easy to miss precisely because nothing
observable goes wrong for the attacker.

Pinned by `a_cheap_fault_cannot_starve_the_node`.

## 10.2 The daemon sized host memory from a number a client chose

**The probe.** `dma_read` with `len = 8 GiB` against a 64 KiB mapping.

The daemon's handler read:

```rust
Request::DmaRead { vgpu, src, len } => {
    let mut buf = vec![0u8; len as usize];   // len came off a socket
    map(node.dma_read(vgpu, src, &mut buf), ...)
}
```

The validation that would have rejected this address happens *inside*
`node.dma_read` — after the allocation. With a large enough `len`, the
allocation fails, Rust aborts, and the process hosting **every tenant on
the card** dies. No exploit needed: one integer.

**Why the existing tests missed it.** Every DMA test asked for a sensible
number of bytes, because every DMA test was written by someone trying to
make DMA work. Hostile input is a different activity from correct input,
and a suite of the second kind never accidentally becomes a suite of the
first.

**The fix.** `MAX_DMA_BYTES` in the core, checked by `check_transfer_len`
*before* anything is sized — the ordering is the entire fix. The daemon
calls it first and only then allocates. The shim chunks large guest
transfers transparently, so the limit costs guests nothing (real drivers
segment DMA rings behind a single `cudaMemcpy` for the same reason).

While fixing this, the framing layer turned out to have the mirror
problem: `write_frame` enforced no limit at all, while `read_frame`
rejected anything over 16 MiB. A peer could therefore emit a frame its
own reader would refuse, and a receiver cannot skip a body it declined to
size — the connection is unrecoverable. Worse, `body.len() as u32`
truncates silently past 4 GiB, framing the wrong byte count. **Asymmetric
limits between a writer and its reader are always a bug**; the writer now
refuses what the reader would.

Pinned by `a_hostile_transfer_length_cannot_kill_the_daemon`,
`transfers_at_the_limit_are_still_served`,
`the_framer_refuses_to_emit_what_it_could_not_read`.

## 10.3 Live migration could not survive a live guest

**The probe.** Migrate a tenant while it calls `malloc` — the single most
ordinary thing a running guest does.

```
migrate result = Err(Client(Device(PageFault { addr: .., access: Write })))
```

**Why.** Pre-copy replays the source's allocations onto the twin, then
copies dirty pages while the guest keeps running. New pages are *born
dirty* (correctly — they have never been copied anywhere), so
`take_dirty` returned pages belonging to an allocation the twin did not
have, and writing them faulted the destination.

The irony is exact: the milestone-3 docs already discussed structure
drift and handled it — but only *after* the suspend, on the assumption
that drift was an unlucky corner. It is not a corner. "Live" means the
guest is running, and a running guest allocates. **The condition the
feature exists to handle was treated as the exception.**

**The fix.** Pre-copy rounds copy only pages inside the replayed
structure and defer the rest to the post-suspend pass, which rebuilds
against frozen truth. Correct always, fast in the common case.

The same probe surfaced a smaller sibling: migration demanded a
`Running` source, so a tenant an operator had just suspended — or one
merely *placed* and not yet started — could not be moved. Those are
precisely the tenants an operator most wants to relocate. The
precondition now names the property migration actually needs (*the rings
cannot change*) instead of one state that happens to imply it, and
`export_channels` accepts `Created` as well as `Suspended`.

Pinned by `migration_survives_a_guest_that_allocates_mid_flight`,
`suspended_and_unstarted_tenants_can_be_migrated`.

## 10.4 The completion clock could run backwards

**The probe.** Signal fence 5, then fence 1.

```
after signal 5: fence = 5
submit fence 1 (regression) accepted = true
after signal 1: fence = 1
```

Every waiter in this codebase — `vgpu_shim::synchronize`, any guest
polling a fence — reasons "fence ≥ N implies everything submitted before
N has completed". A fence that moves backwards makes that reasoning
false: a wait that was satisfied becomes unsatisfied, and a later wait
for a low value returns instantly against a stale high one. The guest
observes a `synchronize` that returns before its work is done, which
surfaces as data corruption arbitrarily far away.

The docs had said "values must be monotonically increasing per channel"
since milestone 0. Nothing enforced it. **A documented invariant with no
enforcement is a comment**, and the one place able to enforce it is the
device.

**The fix.** `Channel::submit` rejects any non-increasing fence with a
typed `FenceRegression`. The check is against the highest value
*submitted*, not completed — two queued fences must still order relative
to each other. The high-water mark travels in `ChannelExport`, so a guest
cannot rewind its own clock by being migrated.

Pinned by `fence_values_cannot_move_backwards`,
`fence_monotonicity_survives_migration`,
`fence_regression_is_refused_over_the_wire`.

## 10.5 The fabric could invent capacity that did not exist

**The probe.** Register the same node address twice.

```
registered same address as node0 and node1
fabric believes total VRAM = 128 frames (actual: 64)
```

Registration is exactly the operation an operator retries and a config
reload repeats. Two ids for one card meant every downstream placement
decision was computed against a fiction, surfacing much later as an
inexplicable admission failure at a node the fabric was certain had room.

**The fix.** `add_node` is idempotent by address. And, relatedly:
placement now walks down the ranking when a node refuses, because the
fabric reads capacity and *then* admits — a node's refusal is not
evidence the fleet is full.

Pinned by `registering_a_node_twice_does_not_invent_capacity`,
`placement_falls_through_when_the_best_candidate_cannot_host`.

## 10.6 What this chapter is really about

Six defects, found by an afternoon of probing a system that had 85
passing tests and five chapters of design documentation. They cluster
into three kinds, and the kinds generalize past this codebase:

1. **Untrusted numbers used unbounded** (§10.1, §10.2). A length, a cost,
   a count — anything a client names must be bounded before it sizes an
   allocation or a bill. Both bugs here were one ordering away from
   correct: validate, *then* use.
2. **The exceptional case was the normal case** (§10.3). Pre-copy assumed
   a quiet guest; "live" means the opposite. When a feature's whole
   purpose is to run concurrently with something, the concurrent case is
   the design center, not the edge.
3. **Invariants asserted in prose, not in code** (§10.4, §10.5). Fence
   monotonicity was documented for four milestones and enforced in none.
   Idempotent registration was assumed by everyone and implemented by
   no one.

The methodological point: these were found by *asking the system hostile
questions*, not by re-reading it. A test suite grown alongside features
inherits the author's blind spots wholesale. Deliberately switching roles
— from "make this work" to "make this fail" — is a different activity and
finds a different class of bug, and it is the cheapest quality
intervention available on a codebase that already works.

Every probe in this chapter is now a permanent test, which is the other
half of the discipline: a bug found by probing that does not become a
regression test has been fixed once, rather than fixed.

Next: [11 — telemetry, QoS, and checkpointing](11-operating-the-fabric.md),
the features an operator needs once the thing is trustworthy enough to run.
