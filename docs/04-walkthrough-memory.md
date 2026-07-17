# 4. Walkthrough: The Memory Subsystem, Line by Line

This chapter reads `types.rs`, `vram.rs`, and `gmmu.rs` the way a reviewer
should: every non-obvious line gets its *why*. Have the files open
alongside.

## 4.1 `types.rs`: making address confusion a compile error

```rust
pub struct GpuVirtAddr(pub u64);
pub struct VramAddr(pub u64);
pub struct FrameNum(pub u64);
```

Three wrappers around `u64` that cost nothing at runtime (a newtype
compiles to the bare integer). What they buy: a `GpuVirtAddr` cannot be
passed where a `VramAddr` is expected. In device-model code the classic
catastrophic bug is exactly that mix-up — guest-controlled integer used
as a physical offset — and it is the bug class an IOMMU exists to stop in
hardware. Here the type checker stops it before the code runs.

```rust
pub const FRAME_SIZE: u64 = 64 * 1024;
```

64 KiB, not the CPU-reflex 4 KiB, for two hardware-faithful reasons:
GPUs natively support and prefer "big pages" for VRAM (NVIDIA's GMMU has
a 64 KiB page mode), and a GPU TLB miss stalls thousands of threads, so
fewer/bigger pages per buffer is a first-order performance decision, not
a tuning detail.

The error enum is flat and closed (`VgpuError`) rather than per-module:
milestone 1 must serialize errors over a wire protocol, and one closed
set keeps that mapping total — every error the model can produce has a
name today, before the wire exists.

## 4.2 `vram.rs`: the buddy allocator

### Why buddy, restated in one sentence

VRAM outlives every tenant, so the allocator must *reassemble* large
contiguous blocks after arbitrary alloc/free churn — the buddy system
guarantees coalescing at O(log n) cost, which is why Linux's page
allocator and DRM's `drm_buddy` (used by amdgpu/i915 for VRAM) chose it.

### The data structure

```rust
free: Vec<BTreeSet<u64>>,
```

`free[k]` = start frames of every free block of exactly `2^k` frames.
The inner collection is a `BTreeSet`, not a `Vec` or `HashSet`, for one
reason: `iter().next()` yields the *lowest-addressed* free block, making
allocation deterministic. Determinism is a load-bearing property (see
§3.4) — the scrubbing test literally relies on a new tenant landing on
the same frames a destroyed tenant vacated.

### Seeding: non-power-of-two cards

```rust
let mut cursor = 0u64;
for order in (0..=max_order).rev() {
    let block = 1u64 << order;
    if total_frames & block != 0 {
        free[order as usize].insert(cursor);
        cursor += block;
    }
}
```

A 24 GiB card is 393,216 frames — not a power of two. Instead of
rounding the card down (wasting VRAM) or up (inventing VRAM), seed the
free lists with the *binary decomposition* of the frame count: one
maximal block per set bit (`2^18 + 2^17` for 24 GiB), packed from
address 0 upward.

The subtle line is the loop *order*: descending. Placing bigger blocks
first means `cursor` is always a multiple of the next (smaller) block
size, so every seeded block is **naturally aligned** — a block of size
`2^k` starts at an address divisible by `2^k`. Natural alignment is not
cosmetic; the XOR trick below is *only correct* for naturally aligned
blocks.

### Allocation: search up, split down

```rust
let found = (order..=self.max_order)
    .find(|&k| !self.free[k as usize].is_empty())
    .ok_or(VgpuError::OutOfVram { ... })?;
```

Search *upward* from the exact-fit order. Never grab a big block when an
exact-fit block exists — that discipline is what bounds fragmentation.

```rust
let mut k = found;
while k > order {
    k -= 1;
    let buddy = start + (1u64 << k);
    self.free[k as usize].insert(buddy);
}
```

Each iteration halves the block: keep the lower half (continue splitting
it), push the upper half — the **buddy** — onto the free list one order
down. Splitting an order-4 block to satisfy an order-0 request leaves
buddies at orders 3, 2, 1, 0: exactly the breadcrumbs `free` needs to
reassemble the order-4 block later.

### Free: the XOR trick

```rust
while order < self.max_order {
    let buddy = start ^ (1u64 << order);
    if !self.free[order as usize].remove(&buddy) {
        break;
    }
    start = start.min(buddy);
    order += 1;
}
```

Where is my buddy? Flip bit `k` of my start address. Why: two buddies at
order `k` are the halves of one naturally-aligned block at order `k+1`,
so their addresses are identical except for bit `k` (the lower half has
it clear, the upper half has it set). `start ^ (1 << k)` is therefore a
constant-time buddy lookup, no metadata needed.

The loop then climbs: if the buddy is free, remove it, merge (the merged
block starts at `min` of the two — the one with bit `k` clear), and try
to merge again one order up. Free space "bubbles" back into the largest
possible blocks automatically. If the buddy is busy, stop — nothing more
can coalesce until it frees.

One honest failure mode is pinned by a test
(`fragmentation_can_fail_with_free_space`): a buddy allocator can hold
free frames yet fail a larger request, because free frames whose buddies
are busy cannot merge. Real allocators live with this; hiding it would
misteach.

### `FrameStore`: sparse truth + scrubbing

The backing store maps `frame → Box<[u8; 64 KiB]>` lazily: simulating an
8 GiB card costs host memory only for frames actually written. Two
deliberate semantics:

* Reads of never-written frames return zeros — matching "scrubbed" VRAM.
* `scrub(frame)` drops the box on free, so recycled frames read as zeros
  for the next tenant. `freed_vram_is_scrubbed_before_reuse` proves the
  end-to-end property: write a secret, free, destroy the tenant, admit a
  new tenant, allocate (deterministically the same frames), read zeros.

`split()` at the bottom `debug_assert!`s that no physical access crosses
a frame boundary. That assert encodes a *global* invariant: nothing in
the crate ever assumes two frames are physically adjacent — scatter-gather
segmentation is always the caller's job (see `translate_range` below).
It is a `debug_assert` and not an `Err` because violating it is a bug in
the engine, not bad guest input; guests cannot reach this code with
unvalidated values.

## 4.3 `gmmu.rs`: the page tables

### The address split

```text
  bit  35                26 25              16 15               0
       +------------------+------------------+------------------+
       |  PDE index (10b) |  PTE index (10b) |   offset (16b)   |
       +------------------+------------------+------------------+
```

```rust
let offset  =  addr.0                            & ((1 << OFFSET_BITS) - 1);
let pte_idx = ((addr.0 >> OFFSET_BITS)           & ((1 << PTE_BITS) - 1)) as usize;
let pde_idx = ((addr.0 >> (OFFSET_BITS+PTE_BITS)) & ((1 << PDE_BITS) - 1)) as usize;
```

Pure bit surgery: shift the field down, mask its width. 16 offset bits
because frames are 2^16 bytes; 10+10 index bits give a 2^20-page = 64 GiB
per-tenant space. Real GMMUs use 4–5 levels to reach 49 bits; two levels
preserves every structural property (radix tree, lazy interior nodes,
valid bits) at teachable size, and the constants are named so the level
count is a parameter of the design, not a magic number.

Addresses ≥ 2^36 are rejected before decoding — hardware has no wires
for those bits; software should have no path for them.

### The radix tree, and why it's lazy

```rust
directory: Vec<Option<PageTable>>,          // 1024 slots
entries:   Box<[Option<Pte>; 1024]>,        // per table
```

A flat table for a 2^20-page space would burn megabytes per tenant even
for a guest that mapped one page. So the directory starts as 1024
`None`s, and a second-level table materializes on first map into its
64 MiB region (`get_or_insert_with(PageTable::new)` — that one call *is*
the lazy allocation). `Option<Pte>` plays the hardware "valid bit";
`Option<PageTable>` plays the hardware "PDE valid bit".

Each table also keeps `live: u32`, a count of valid entries, so `unmap`
can free an emptied table in O(1) instead of scanning 1024 slots. Sparse
guests stay sparse in both directions.

### Two-pass mutation: the all-or-nothing rule

Both `map` and `unmap` run **validate-everything, then mutate-everything**:

```rust
// Pass 1: pure validation, no mutation.
for ... { check AlreadyMapped / NotMapped / range ... }
// Pass 2: guaranteed to succeed — mutate.
for ... { install / clear PTEs }
```

The alternative — mutate as you validate, undo on failure — leaves a
window where the structure is half-changed, and undo code is the least
tested code in any system. A failed `map` here provably changes nothing
(`overlapping_map_is_rejected_atomically` asserts it), which is what
makes the operation safely *retryable* by a control plane. This
validate-then-commit shape recurs in `vgpu.rs` (allocation rollback) and
`engine.rs` (translate before touching); it is the crate's house style
for fallible mutation.

### Translation

```rust
let table = self.directory[pde].as_ref().ok_or(PageFault)?;
let entry = table.entries[pte].ok_or(PageFault)?;
if matches!(access, Write) && !entry.writable { return Err(PageFault); }
Ok(VramAddr(entry.frame.base_addr().0 + offset))
```

The walk reads exactly like the hardware's microcode: directory lookup,
table lookup, permission check, frame base + offset. One deliberate
choice: a write to a read-only page and an access to an unmapped page
return the *same* fault shape to the guest. Distinguishing them would
leak mapping-state information across the trust boundary; the host-side
fault record keeps the distinction for debugging.

### `translate_range`: software scatter-gather

```rust
while cur < end {
    let pa = self.translate(GpuVirtAddr(cur), access)?;
    let in_page = FRAME_SIZE - va.frame_offset();
    let take = in_page.min(end - cur);
    segs.push((pa, take));
    cur += take;
}
```

A byte range in virtual space is a list of segments in physical space,
broken at every page boundary, because adjacent virtual pages map to
arbitrary frames. `in_page` is "bytes until the next boundary"; `take`
caps it by "bytes still needed". Every memory-touching operation in the
crate — fills, copies, host DMA — consumes these segment lists. This is
precisely what a DMA engine's scatter-gather unit does with a mapped
buffer, and it is the mechanism that lets `vgpu.rs` back one contiguous
guest buffer with non-contiguous physical blocks.

## 4.4 `vgpu.rs`: composing allocator + GMMU under a budget

`alloc_memory` is the best single function to study because it stacks
four decisions in a deliberate order:

```rust
// 1. budget first
if self.vram_used + charged > self.profile.vram_bytes { return Err(...); }
// 2. binary decomposition
while remaining > 0 {
    let chunk = 1u64 << (63 - remaining.leading_zeros());   // highest set bit
    match vram.alloc(chunk) { Ok(r) => ranges.push(r),
        Err(e) => { for r in ranges { vram.free(r); } return Err(e); } }  // 3. rollback
    remaining -= chunk;
}
// 4. map last
self.aspace.map(base, frames, true)?;
```

1. **Budget before physical.** The tenant's contract is checked first —
   a busy card must not turn a budget violation into a confusing
   `OutOfVram`.
2. **Binary decomposition defeats round-up waste.** Asking the buddy
   allocator for 5 frames directly would burn 8. Asking for 4 + 1 burns
   exactly 5, and the GMMU maps both blocks as one contiguous VA range —
   the allocator provides *frames*, paging provides *contiguity*. This
   composition is the reason real drivers can run a buddy allocator
   without an internal-fragmentation crisis.
3. **Rollback leaves no orphans.** If the 1-frame chunk fails after the
   4-frame chunk succeeded, the 4 frames go back before the error
   propagates (`failed_allocation_rolls_back_cleanly` pins it).
4. **Map last.** Page tables only ever point at frames the tenant owns.
   The reverse order would create a window where a mapped VA points at
   frames that might be rolled back.

`free_memory` runs the mirror image — unmap, **scrub every frame**,
return blocks — and `destroy` loops it over everything the tenant holds,
idempotently, so control-plane retries are safe.

Next: [05-walkthrough-execution.md](05-walkthrough-execution.md) — rings,
the engine, vruntime scheduling, and the tick loop.
