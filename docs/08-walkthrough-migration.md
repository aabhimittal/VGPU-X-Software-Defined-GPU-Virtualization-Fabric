# 8. Walkthrough: Live Migration (Milestone 3)

Move a running tenant's vGPU from node A to node B such that the tenant
cannot tell it happened — same pointers, same data, same fence values,
its queued work completing on the new card. This chapter reads the new
pieces: dirty bits in the GMMU, the migration primitives, and the
pre-copy driver in `vgpu_proto::migrate`.

## 8.1 Why this is possible at all: the indirection pays out

Everything in this milestone rests on one fact established in milestone
0: **guests hold virtual addresses, and only page tables know about
frames.** So migration never moves physical state — it recreates it:

* The destination's buddy allocator hands the twin *whatever frames it
  has* (the tests deliberately pre-occupy the destination's low frames
  to prove the physical layouts differ).
* The GMMU maps those different frames at the *same guest VAs*.
* The tenant's pointers — in registers, in its own data structures, in
  queued commands — remain valid without rewriting a single one.

The two other milestone-0 decisions that were quietly waiting for this
chapter also come due. `Suspended` in the lifecycle: a frozen vGPU's
rings are a closed set, which is what makes channel export a snapshot
rather than a chase. And determinism: the migration tests assert exact
dirty sets and exact fence values because nothing in the core is timing-
dependent.

## 8.2 Dirty bits: hardware A/D bits, repurposed as they really are

The PTE grows one bit:

```rust
struct Pte { frame: FrameNum, writable: bool, dirty: bool }
```

Semantics, each earning its keep:

* **Born dirty.** A freshly mapped page has never been copied anywhere,
  so from a migrator's perspective it *is* dirty. Consequence: the first
  `take_dirty` returns every mapped page — the pre-copy loop's "round 1
  = bulk copy" needs no special case.
* **Set on write, by every write path.** The engine's fill, copy, and
  kernel store, and the node's host DMA, all call `mark_dirty_range`
  after a successful write translation. This invariant is the scary one:
  a missed call site is not a crash but *silent post-migration
  corruption*. It is therefore pinned by a test that exercises all four
  write paths and asserts each page surfaces dirty
  (`every_write_path_marks_dirty`) — and reads, pointedly, do not.
* **Cleared on harvest.** `take_dirty` returns the set and resets it —
  the read-and-reset cycle real migration code performs against MMU A/D
  bits. Clear-on-read is what makes round N+1 mean "what changed while
  round N was copying".

## 8.3 The primitives: small verbs, not a snapshot blob

Migration could have been one `Snapshot`/`Restore` message pair carrying
a giant serialized vGPU. It is instead five small verbs (`GetProfile`,
`ListAllocations`, `TakeDirty`, `ExportChannels`/`ImportChannels`, plus
`AllocMemoryAt`), and page contents move over the *existing* DMA
messages. The reasons are the usual ones for preferring verbs to blobs:

* Each verb reuses a tested mechanism (replay uses the allocator, page
  transfer uses DMA-through-translation — so even the migrator cannot
  touch memory except through the twin's page tables).
* The driver's control flow — the actual pre-copy algorithm — lives in
  *client* code where it can be read, tested, and replaced, instead of
  being frozen into the protocol.
* A blob is all-or-nothing; verbs stream, interleave with guest work,
  and fail at a known step.

### The bug the design almost shipped: heap holes

The first replay design was "re-run `alloc_memory(bytes)` in creation
order — the bump heap is deterministic, so the twin gets the same VAs."
True only for a heap that has never seen `free`. A guest that allocated
A, B and freed A leaves a *hole*; naive replay would compact it, shifting
B to A's address and silently invalidating every pointer the guest holds
into B. The fix is `alloc_memory_at(base, bytes)`: replay reproduces the
heap's exact *shape*, holes included, and the bump pointer just moves
past the highest placed allocation. The over-the-wire test gives the
tenant a freed hole and asserts the hole is still unmapped on the twin —
pointer-compaction bugs fail loudly here.

## 8.4 The pre-copy driver, annotated

`vgpu_proto::migrate` is the classic algorithm (QEMU's, vMotion's), and
its shape is worth internalizing because it recurs across all of
infrastructure:

```text
1. twin = dst.create_vgpu(profile)        admission control still applies
2. replay allocations (placed)            structure, not contents
3. while rounds remain, source RUNNING:
     dirty = src.take_dirty()             round 1 = everything
     copy dirty pages                     guest keeps mutating underneath
4. src.suspend()                          brownout begins
5. copy the final dirty set               small, if step 3 converged
6. export/import channels                 pending work + fence values
7. dst.start(twin); src.destroy()         brownout ends
```

**Why it converges:** each round copies what the guest dirtied during
the previous round's copy. Guest dirtying slower than the wire moves
pages ⇒ dirty sets shrink geometrically ⇒ the suspended-time copy in
step 5 is tiny. **Why it might not:** a guest that redirties memory
faster than the wire forever. Hence the round budget and the fallback —
guaranteed termination at the price of a longer brownout. The module
docs spell out this trade because it is the entire engineering content
of "live" migration.

**Structure drift** (step 4b in the code): the guest may malloc/free
*during* step 3, staling the replayed structure. After suspending, the
driver re-lists allocations against the now-frozen truth; on mismatch it
rebuilds the twin and full-copies. Correct always, fast usually — the
standard shape for optimistic protocols.

**What deliberately does not migrate:** scheduler vruntime. Fairness is
relative to a node's *other* tenants; an imported vruntime would be
meaningless on the destination (and could grant a huge historical credit
there). The twin joins at the destination's high-water mark like any
newcomer — the milestone-1 `register` clamp handles it with zero new
code.

**Failure discipline:** any error unwinds the twin (never leak a
half-built vGPU on the destination) and leaves the source intact — the
`migration_to_a_full_node_fails_cleanly` test drives a migration into a
full node and asserts the tenant stays home, running, data intact, with
a *typed* admission error.

## 8.5 What the tests prove

| Claim | Test |
|---|---|
| Every write path (DMA, fill, copy, kernel store) marks dirty; reads do not | `every_write_path_marks_dirty` (fabric) |
| A vGPU moves across nodes onto different frames with all guest-visible state intact, pending work completing on the destination | `stop_and_copy_migration_between_nodes` (fabric, in-core) |
| The same holds over real TCP between two daemons, including a heap hole and post-migration liveness | `live_migration_between_two_daemons` |
| Writes landing between pre-copy rounds are caught by the next round's exact dirty delta | `writes_between_precopy_rounds_are_not_lost` |
| Migration to a full node fails typed, cleanly, source untouched | `migration_to_a_full_node_fails_cleanly` |

Protocol note: the five new messages (plus `AllocMemoryAt`) took the
wire to v3 — appended tags, one version bump, exhaustive roundtrips, per
the policy set in milestone 1.

Next (M4): the fabric — a control plane that *decides* these migrations:
placement as profile bin-packing, node inventory, and rebalancing.
