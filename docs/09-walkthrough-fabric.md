# 9. Walkthrough: The Fabric (Milestone 4)

The last layer. A fleet of `vgpud` nodes becomes one pool of GPU
capacity, managed by a control plane that answers three questions:
*where should this tenant run*, *what is running where*, and *how do I
move things*. This chapter reads `vgpu-fabric` — deliberately the
smallest crate in the workspace, and the reasons it gets to be small are
the chapter's real content.

## 9.1 Control plane vs data plane

`Fabric::place` returns a `TenantHandle` — node address plus vGPU id —
and from then on the guest talks to that node *directly* (via
`VgpuClient` or the milestone-2 shim). The fabric never proxies a
doorbell write or a DMA.

This split is the defining shape of fleet systems (Kubernetes does not
forward your pod's packets; a storage control plane does not sit in the
read path), and the reasons transfer exactly:

* **Load**: placement happens once per tenant; submissions happen
  millions of times. A proxying fabric would scale with tenant
  *traffic*; this one scales with tenant *churn*.
* **Blast radius**: a crashed fabric strands no data-plane traffic —
  every guest keeps computing against its node. Only *placement* pauses.
* **Clarity**: policy (where things go) and mechanism (how things run)
  get separate codebases, separate failure modes, separate upgrade
  cadences.

The same logic shows up small in a subtler choice: the fabric opens a
**short-lived connection per operation** instead of holding one per
node. Control-plane operations are rare enough that connection cost is
noise, and the fabric never pins a node connection slot or holds a
stream across a minutes-long migration.

## 9.2 Placement: the milestone-0 promise, kept

Milestone 0's profile design (`docs/03`, §3.5) made a forward-looking
claim: fixed bundles "make placement decidable for a fabric scheduler:
it can pack profiles onto cards like Tetris pieces instead of solving a
knapsack problem per request." This crate is where that promise is
cashed. Because a tenant's demand is one known number — the profile's
VRAM budget, *guaranteed* by node admission control — placement is
textbook bin-packing, and the entire policy fits in one function:

```rust
// best_fit: among nodes with uncommitted >= requested,
// choose the one with minimal slack; ties break by NodeId.
```

**Why best-fit?** The three classic greedy rules pull differently:

* *First-fit*: fastest, packs indifferently.
* *Worst-fit* (most free space): spreads load — and is exactly how a
  fleet ends up 40% free with nowhere to put one large tenant, because
  every big contiguous capacity got nibbled by small placements.
* **Best-fit**: packs tightly, *preserving the large capacities* for the
  large profiles only a few nodes can host.

Bin-packing is NP-hard; best-fit is not optimal, just good and
explainable — and with profile-shaped demands the greedy rule performs
well in practice, which is the actual reason clouds sell instance
*types* rather than arbitrary-sized VMs. The placement test pins the
policy exactly (`placement_is_best_fit_and_deterministic` computes every
slack by hand in comments), which only works because placement reads
*live* capacity from `node_info` and ties break deterministically.

Capacity failures are typed and carry the numbers a caller acts on
(`NoCapacity { requested, best_available }`): queue the tenant, shed
load, or buy hardware — the caller decides, the error informs.

## 9.3 Identity: what stays stable when everything moves

`(NodeId, VgpuId)` changes when a tenant migrates. `TenantId` does not —
it is the fabric-stable name a client holds, resolved to current
coordinates via `Fabric::handle`. This is the standard two-level naming
move (stable logical name → volatile physical location) that makes
migration *usable*, not just possible: milestone 3 built the mechanism,
but only a name that survives the move lets anyone build on top of it.

## 9.4 Evacuation: migration becomes an operation

```rust
pub fn evacuate(&mut self, node: NodeId) -> FabricResult<u32>
```

Drain-for-maintenance: every resident tenant live-migrates to the
best-fit *other* node. Two design points:

* **One at a time, registry updated after each.** A mid-drain failure
  leaves a consistent, partially-drained fabric — never a lost tenant,
  never a registry pointing at a vGPU that isn't there. (Milestone 3's
  contract does the heavy lifting: a failed migration leaves its source
  intact.)
* **The destination choice reuses `best_fit` with an exclusion**, not a
  second policy. Evacuation is just placement under a constraint.

The test gives a tenant data *and a pending fence*, drains the node, and
asserts the fence completes on the destination — the milestone-3
machinery exercised through the operational front door.

## 9.5 State ownership, honestly

The registry lives in the fabric; the truth lives on the nodes. They
stay consistent because the fabric is the **single writer** — the same
discipline the daemon used for its device thread, one level up. The
module docs state plainly what is *not* implemented and why it is a
boundary, not an oversight: fabric HA (a second instance breaks
single-writer; the fix is consensus — Raft — a different book), node
failure detection, and reconciliation of out-of-band mutations. Knowing
where your system's honesty ends is part of the design.

## 9.6 What the tests prove

| Claim | Test |
|---|---|
| Best-fit placement is exact and deterministic, verified against hand-computed slack | `placement_is_best_fit_and_deterministic` |
| Capacity exhaustion is a typed error with actionable numbers | `placement_overflow_is_typed` |
| Draining a node live-migrates every tenant with data and pending work intact; the node ends empty | `evacuation_drains_a_node_with_tenants_intact` |
| Explicit migration updates the registry; destroy releases capacity; bogus targets refused up front | `migrate_tenant_and_destroy_keep_the_registry_true` |

## 9.7 The whole stack, closed

With M4 the original roadmap completes. Reading bottom-up:

```
vgpu-fabric   places tenants and moves them        (bin-packing, drain)
vgpud         serves one GPU to many guests        (single-owner thread)
vgpu-proto    makes every operation a wire message (total codecs, v3)
vgpu-shim     gives guests a familiar API          (streams, fences)
vgpu-core     makes sharing safe at all            (GMMU, budgets, vruntime)
```

Every layer leans on a decision made below it and earlier than it:
placement leans on profiles (M0), evacuation leans on migration (M3),
migration leans on `Suspended` and determinism (M0), the shim leans on
fences (M0) and the wire (M1), and everything leans on the two isolation
invariants that have not changed since the first commit. That is the
book's closing argument: **in systems, the leverage is all in the early
constraints.**

Where a reader could take it next, each a real project: a TLB model with
shootdowns, copy/compute engine parallelism (a second scheduler
dimension), post-copy migration, a C-ABI `libvcuda.so` for real
`LD_PRELOAD` interposition, or consensus-backed fabric HA.
