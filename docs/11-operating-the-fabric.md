# 11. Operating the Fabric: Telemetry, QoS, and Checkpoints

The first five milestones made the fabric *correct*. Chapter 10 made it
*hard to break*. This chapter adds the three things that decide whether a
correct, robust system is actually runnable — and each one exists because
of a question the design could not previously answer.

| Question | Feature | Where |
|---|---|---|
| "What is happening on my fleet?" | telemetry | `vgpu_core::metrics`, `Fabric::telemetry` |
| "What did this tenant actually buy?" | QoS caps and reservations | `sched::QosLimits` |
| "How do I free a card with nowhere to move to?" | checkpoint / restore / clone | `vgpu_proto::checkpoint` |

## 11.1 QoS: what weights cannot say

The milestone-0 scheduler answers one question well: *who gets the GPU
now?* Proportional share is the right answer to it. But operators sell
things that question cannot express.

**"You may never exceed 25%, even on an idle card."** A weight cannot say
this, because weights only bind under contention: a weight-1 tenant alone
on the card gets 100% of it. That sounds generous until you follow it
through. A customer benchmarks at 3am, sees full-card throughput, and
builds a capacity plan on it. At 9am the neighbours arrive and their
throughput halves. Nothing is broken, the vendor cannot reproduce it, and
the customer is right to be angry — they were shown a number that was
never theirs. A hard cap trades throughput for the thing a tier actually
sells: **predictability**. NVIDIA ships this distinction as "fixed share"
versus "best effort" scheduling.

**"You are guaranteed 30% under any contention."** Weights give a
*ratio*, and a ratio's floor collapses as the room fills: a weight-3
tenant holds 75% against one weight-1 neighbour and 23% against nine. An
SLA is an absolute floor, so it needs to be expressed as one.

Both are enforced over a window (`QOS_WINDOW_CYCLES`), because an
instantaneous "share" is not a measurable quantity — you can only observe
a share by watching an interval. Three design points are worth pulling
out:

* **Idle counts against the window.** If the window only advanced when
  work ran, a capped-out tenant alone on the card would stall the clock
  and then be handed a fresh budget with no time having passed — a
  limiter that limits nothing. So `advance_idle` accounts for the GPU
  sitting out, and `TickReport::idle_cycles` reports it. That idle is the
  visible price of predictability, and it belongs in telemetry rather
  than hidden: it is capacity an operator could sell by raising a cap.
* **Reservations are admission-controlled, exactly like VRAM.** Floors
  that sum past 100% cannot all be honoured, and once a tenant is
  admitted the only ways out are breaking an SLA or evicting someone.
  The honest moment to refuse is before that, so `create_vgpu` sums the
  live reservations and says no.
* **Two tiers is enough.** `pick()` filters out capped tenants entirely,
  then prefers any tenant still below its floor, then falls back to
  proportional share. A floor is a promise that outranks fairness;
  everything above its floor is back to competing normally.

Both limits are opt-in per profile, which is why every fairness test
written in milestone 0 still passes unchanged — the default is exactly
the old behaviour.

## 11.2 Telemetry: counters, and the identity that makes them useful

The metrics themselves are unremarkable and that is the point; what makes
them worth reading is two decisions.

**Counters, not gauges.** Everything except genuine levels (`vram_used`,
`queued_commands`, `state`) is monotonic. Counters are *idempotent under
scraping*: a collector that misses a sample, double-samples, or restarts
loses nothing, because rates are derived by differencing at query time.
This is why Prometheus counters won and why every hand-rolled "current
ops/sec" gauge eventually lies. The one measurement that cannot be a
counter — `window_consumed`, which resets each QoS window — is documented
as a level so nobody differences it.

**The fabric joins metrics to stable identity.** A node knows it is
running `vgpu3`. Only the fabric knows `vgpu3` is `tenant7`, who has
migrated twice this week and whose id has changed each time. An alert
fires against a customer, not against a slot, so `Fabric::telemetry`
pairs every `TenantMetrics` with its `TenantId` — and surfaces `None` for
vGPUs the fabric did *not* place, because unmanaged tenants consume
capacity the fabric is busy promising to someone else.

The single most valuable field is `capped_out`. "My job is slow" and "my
job is slow *because I bought a quarter card*" are indistinguishable from
inside the guest and have opposite remedies. Only the device can tell
them apart, so it says so — turning a support investigation into a
support answer.

`hottest_node` is deliberately advisory rather than automatic. A fabric
that rebalances on its own reading of "hot" will chase transients and
migrate tenants during their busiest minute — the classic autoscaler
failure mode. Surfacing the candidate and letting a human (or a policy
with hysteresis) decide is the conservative default; `migrate_tenant` is
right there when the decision is made.

## 11.3 Checkpoints: moving a tenant out of time

Migration's destination is a live peer, so the source can always be asked
another question. A checkpoint has no peer: the bytes must be
self-describing and complete, because whatever reads them may run next
month on a node sharing nothing with this one. That single difference
buys three capabilities migration cannot provide:

* **Suspend to disk.** Free a card when there is nowhere to evacuate
  *to*. Migration cannot help here — capacity is the very problem it
  needs solved first.
* **Forensics.** Freeze a tenant exactly as it misbehaved and restore it
  repeatedly on a debugging node. Because the device model is
  deterministic, a restored checkpoint replays identically: a bug report
  that is a *file* rather than a story.
* **Clone.** Restore one checkpoint N times for N identical warm
  tenants. This is the one with direct production value: an inference
  worker spends its first minutes loading weights into VRAM, and every
  replica after the first can skip that entirely. Fork, for GPUs.

### Composed, not built

There are **no new wire messages**. A checkpoint is `list_allocations` +
`dma_read` + `export_channels`; a restore is `create_vgpu` +
`alloc_memory_at` + `dma_write` + `import_channels`. Milestone 3 argued
that small verbs beat one snapshot blob because a caller can recombine
them into things the protocol never anticipated. This chapter is that
claim being cashed: an entire feature at the client layer, with the
device untouched.

Two details that are easy to get wrong:

* **`checkpoint` leaves the tenant suspended.** Resuming automatically
  would make the checkpoint a lie the instant it was taken — the guest
  would start writing to memory the bytes claim to describe. For
  `clone_tenant` there is a second reason: the fork instant is the only
  moment at which parent and child are known identical, and resuming the
  parent first would silently make the clone a copy of a *past* state.
* **A checkpoint carries its profile, and restore re-admits under it.**
  A checkpoint is therefore not a way to smuggle a tenant onto a node at
  a QoS tier it did not buy, and a full card refuses a restore exactly as
  it refuses any other admission.

The version byte matters more in a file than on a wire. A peer with the
wrong version is an error you see immediately; a file with the wrong
version is a corruption you see in six months — so `from_bytes` checks it
first, and is total: arbitrary bytes yield a `WireError`, never a panic.
A checkpoint is exactly as trustworthy as a socket.

## 11.4 Probing the new features, immediately

Chapter 10's lesson is that a suite grown alongside a feature inherits
its author's blind spots, so these three features were probed the same
day they were written rather than shipped on the strength of their own
tests. Three probes confirmed correct behaviour: a faulted channel
survives a checkpoint still faulted, an empty tenant round-trips (a
50-byte file), and a clone of a 60%-reserved tenant is refused at 120%
by admission control.

The fourth found a real gap. **A tenant capped at 25% that had spent its
window arrived on a migration destination with a fresh budget** — so a
tenant migrated once per window collected its ceiling twice.

The interesting part is the shape of the mistake. Milestone 3 had
already decided, correctly, that vruntime does *not* migrate: fairness is
relative to a node's other tenants, so importing a vruntime would be
meaningless. When QoS arrived, its window spend inherited that decision
by default — and it is the opposite kind of state. A weight is relative;
a cap is absolute, and "never more than 25%" means the same thing on
every card in the fleet. **A migration has to decide, for each piece of
state it carries, whether that state is relative or absolute**, and a new
feature does not get to inherit the answer from an older one.

The fix carries the window spend across the move, with one deliberate
property: `adopt_qos_window` only ever *raises* the figure. A caller can
therefore throttle itself and nothing else, which is what makes the
operation safe to expose on an unauthenticated device API — the safe
direction is the only direction available.

Pinned by `a_qos_cap_is_not_refreshed_by_migrating` and
`adopting_a_qos_window_can_only_raise_it`.

## 11.5 What the tests prove

| Claim | Test |
|---|---|
| A hard cap binds on an idle GPU; the idle is reported, and cap state is visible mid-window | `a_hard_cap_binds_even_with_the_gpu_to_itself` |
| A cap limits a rate, not a lifetime total | `a_capped_tenant_stays_capped_across_windows` |
| A reservation beats raw weights against heavier neighbours | `a_reservation_holds_against_heavier_neighbours` |
| Unkeepable promises are refused at admission | `reservations_are_admission_controlled` |
| A cap never idles the GPU while anyone is eligible | `a_capped_tenant_does_not_block_its_neighbours` |
| Node counters join to fabric-stable tenant identity | `telemetry_joins_node_counters_to_stable_tenant_identity` |
| Telemetry separates "slow" from "throttled by policy" | `telemetry_distinguishes_slow_from_capped` |
| A tenant round-trips through bytes with heap holes, fences, and pending work | `a_tenant_survives_a_round_trip_through_bytes` |
| Clones are warm, independent, and don't share memory or identity | `cloning_a_warm_tenant_yields_independent_copies` |
| A clone can land on a different node | `a_clone_can_be_restored_onto_another_node` |
| Corrupt, truncated, and stale checkpoints are refused, never trusted | `corrupt_and_stale_checkpoints_are_refused_not_trusted` |
| Restore obeys admission control and leaves nothing behind on failure | `restoring_still_obeys_admission_control` |
| A cap is not refreshed by migrating | `a_qos_cap_is_not_refreshed_by_migrating` |
| Carrying a window spend can only throttle, never exempt | `adopting_a_qos_window_can_only_raise_it` |

## 11.6 The through-line

Each feature here is a case of the same pattern: an operational question
the existing abstractions could not express, answered by extending the
*vocabulary* rather than the mechanism. Caps and reservations reuse the
vruntime scheduler. Telemetry reads state the node already maintained.
Checkpointing adds no wire messages at all. That is what a design pays
back when its early constraints were chosen well — the fifth feature
costs less than the first, not more.
