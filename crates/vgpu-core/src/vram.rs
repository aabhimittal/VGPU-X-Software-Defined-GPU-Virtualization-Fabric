//! Physical VRAM management: a buddy allocator for frames, and a sparse
//! byte-addressable backing store so the simulation moves *real bytes*.
//!
//! # Why a buddy allocator?
//!
//! VRAM is a single flat range of physical memory shared by every vGPU on
//! the card, and it lives for the lifetime of the host — so fragmentation
//! is not a theoretical concern, it is *the* concern. The buddy system
//! (used by the Linux page allocator and by DRM's `drm_buddy`, which
//! amdgpu/i915 use for exactly this job) gives:
//!
//! * O(log n) alloc/free,
//! * guaranteed coalescing of freed neighbors back into large blocks,
//! * power-of-two block sizes, which match how GPU page tables want
//!   physically-contiguous spans for big-page mappings.
//!
//! The trade is internal fragmentation (a 3-frame request burns 4 frames).
//! Real drivers accept the same trade for the same reason we do: predictable
//! reassembly of large contiguous blocks matters more than a few percent of
//! waste, because a GPU that cannot find a contiguous block for a new
//! vGPU's page directory is a GPU that cannot admit tenants.

use std::collections::{BTreeSet, HashMap};

use crate::types::{FrameNum, Result, VgpuError, VramAddr, FRAME_SIZE};

/// A contiguous run of physical frames handed out by the allocator.
///
/// `order` is remembered so `free` does not need a lookup table to know how
/// big the block was — the receipt carries its own size. Dropping a
/// `FrameRange` without freeing it leaks VRAM, exactly like real hardware;
/// the owning vGPU is responsible for returning it (and does, in
/// `Vgpu::destroy`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameRange {
    /// First frame in the run.
    pub start: FrameNum,
    /// Number of frames: always `1 << order`.
    pub count: u64,
    /// Buddy order this block was carved at (needed to free it).
    pub order: u8,
}

impl FrameRange {
    /// Iterate the frame numbers in this range.
    pub fn frames(&self) -> impl Iterator<Item = FrameNum> + '_ {
        (self.start.0..self.start.0 + self.count).map(FrameNum)
    }
}

/// Buddy allocator over the card's frame space.
pub struct VramAllocator {
    /// `free[k]` holds the start frame of every free block of size `2^k`
    /// frames. A `BTreeSet` (not `Vec`) so allocation is deterministic:
    /// we always split the *lowest-addressed* block, which keeps the layout
    /// reproducible across runs — a property the whole test suite leans on.
    free: Vec<BTreeSet<u64>>,
    /// Total frames on the card (may include a rounded-up region marked
    /// permanently allocated if the capacity is not a power of two).
    total_frames: u64,
    /// Frames currently handed out.
    allocated_frames: u64,
    /// Largest supported order (a single block spanning all of VRAM).
    max_order: u8,
}

impl VramAllocator {
    /// Build an allocator for `vram_bytes` of VRAM.
    ///
    /// Capacity must be a frame multiple. Non-power-of-two capacities are
    /// handled by seeding the free lists with the *binary decomposition* of
    /// the frame count — e.g. 24 GiB = 393216 frames = 2^18 + 2^17 blocks —
    /// instead of forcing a power-of-two card size.
    pub fn new(vram_bytes: u64) -> Self {
        assert!(
            vram_bytes.is_multiple_of(FRAME_SIZE) && vram_bytes > 0,
            "VRAM capacity must be a positive multiple of FRAME_SIZE"
        );
        let total_frames = vram_bytes / FRAME_SIZE;
        // Highest bit set in total_frames bounds the largest block we can
        // ever need a free list for.
        let max_order = (63 - total_frames.leading_zeros()) as u8;
        let mut free: Vec<BTreeSet<u64>> = vec![BTreeSet::new(); max_order as usize + 1];

        // Seed: walk the bits of total_frames from high to low, placing one
        // maximal block per set bit, packed from address 0 upward. Every
        // block is naturally aligned to its own size because higher-order
        // blocks are placed first, so `cursor` is always aligned to the
        // next (smaller) block size. Natural alignment is what makes the
        // buddy XOR trick in `free_block` valid.
        let mut cursor = 0u64;
        for order in (0..=max_order).rev() {
            let block = 1u64 << order;
            if total_frames & block != 0 {
                free[order as usize].insert(cursor);
                cursor += block;
            }
        }

        Self {
            free,
            total_frames,
            allocated_frames: 0,
            max_order,
        }
    }

    /// Frames not currently allocated.
    pub fn free_frames(&self) -> u64 {
        self.total_frames - self.allocated_frames
    }

    /// Total frames on the card.
    pub fn total_frames(&self) -> u64 {
        self.total_frames
    }

    /// Allocate a naturally-aligned block of at least `frames` frames.
    ///
    /// The classic buddy loop:
    /// 1. round the request up to a power of two (`order`),
    /// 2. find the smallest free block of order ≥ that,
    /// 3. split it in half repeatedly, returning the halves to the free
    ///    lists, until the block is exactly the requested order.
    pub fn alloc(&mut self, frames: u64) -> Result<FrameRange> {
        if frames == 0 {
            return Err(VgpuError::BadAddress {
                addr: crate::types::GpuVirtAddr(0),
                why: "zero-length allocation",
            });
        }
        let order = order_for(frames);
        if order > self.max_order {
            return Err(VgpuError::OutOfVram {
                requested_frames: frames,
                free_frames: self.free_frames(),
            });
        }

        // Find the smallest order that actually has a free block. Searching
        // upward from `order` (never downward) is what bounds fragmentation:
        // we never split a big block when an exact-fit block exists.
        let found = (order..=self.max_order)
            .find(|&k| !self.free[k as usize].is_empty())
            .ok_or(VgpuError::OutOfVram {
                requested_frames: frames,
                free_frames: self.free_frames(),
            })?;

        // Take the lowest-addressed block at that order (determinism).
        let start = *self.free[found as usize].iter().next().expect("non-empty");
        self.free[found as usize].remove(&start);

        // Split down: each split cuts the block in half; we keep the lower
        // half and push the upper half (the "buddy") onto the free list one
        // order below. Keeping the lower half preserves the low-address
        // packing discipline.
        let mut k = found;
        while k > order {
            k -= 1;
            let buddy = start + (1u64 << k);
            self.free[k as usize].insert(buddy);
            // `start` stays — the lower half is what we keep splitting.
        }

        let count = 1u64 << order;
        self.allocated_frames += count;
        Ok(FrameRange {
            start: FrameNum(start),
            count,
            order,
        })
    }

    /// Return a block to the allocator, coalescing with its buddy at every
    /// order where the buddy is also free.
    ///
    /// The buddy of block `b` at order `k` is `b XOR 2^k`: flipping the
    /// bit that distinguishes the two halves of the order-(k+1) block they
    /// were split from. This only works because every block is naturally
    /// aligned — which `new` and `alloc` guarantee.
    pub fn free(&mut self, range: FrameRange) {
        let mut start = range.start.0;
        let mut order = range.order;
        self.allocated_frames -= range.count;

        while order < self.max_order {
            let buddy = start ^ (1u64 << order);
            if !self.free[order as usize].remove(&buddy) {
                break; // buddy busy (or out of range): cannot merge further
            }
            // Merged block starts at the lower of the two halves.
            start = start.min(buddy);
            order += 1;
        }
        self.free[order as usize].insert(start);
    }
}

/// Smallest order `k` with `2^k >= frames`.
fn order_for(frames: u64) -> u8 {
    // next_power_of_two(1) == 1 -> order 0, etc.
    frames.next_power_of_two().trailing_zeros() as u8
}

// ---------------------------------------------------------------------------
// Backing store
// ---------------------------------------------------------------------------

/// Sparse byte store standing in for the physical VRAM chips.
///
/// We cannot (and must not) reserve gigabytes of host RAM to simulate an
/// 8 GiB card, so frames materialize lazily on first *write*. Reads of
/// never-written frames return zeros — which is also a deliberate modeling
/// choice: real vGPU managers scrub VRAM before handing frames to a new
/// tenant, otherwise tenant A's weights leak to tenant B. Our store gives
/// "scrubbed" semantics for free, and `scrub()` makes it explicit on free.
pub struct FrameStore {
    frames: HashMap<u64, Box<[u8]>>,
}

impl FrameStore {
    /// Empty store: all of VRAM reads as zeros.
    pub fn new() -> Self {
        Self {
            frames: HashMap::new(),
        }
    }

    /// Read `buf.len()` bytes starting at physical address `addr`.
    /// The caller (the GMMU-aware execution engine) guarantees the span
    /// does not cross a frame boundary — physical contiguity of *frames*
    /// is never assumed anywhere in the crate.
    pub fn read(&self, addr: VramAddr, buf: &mut [u8]) {
        let (frame, off) = split(addr, buf.len());
        match self.frames.get(&frame) {
            Some(data) => buf.copy_from_slice(&data[off..off + buf.len()]),
            None => buf.fill(0), // untouched VRAM reads as scrubbed zeros
        }
    }

    /// Write `buf` starting at physical address `addr` (same single-frame
    /// contract as `read`). Materializes the frame on first touch.
    pub fn write(&mut self, addr: VramAddr, buf: &[u8]) {
        let (frame, off) = split(addr, buf.len());
        let data = self
            .frames
            .entry(frame)
            .or_insert_with(|| vec![0u8; FRAME_SIZE as usize].into_boxed_slice());
        data[off..off + buf.len()].copy_from_slice(buf);
    }

    /// Fill `len` bytes at `addr` with `value` (single-frame contract).
    pub fn fill(&mut self, addr: VramAddr, len: usize, value: u8) {
        let (frame, off) = split(addr, len);
        if value == 0 && !self.frames.contains_key(&frame) {
            return; // zero-filling untouched VRAM is a no-op: stay sparse
        }
        let data = self
            .frames
            .entry(frame)
            .or_insert_with(|| vec![0u8; FRAME_SIZE as usize].into_boxed_slice());
        data[off..off + len].fill(value);
    }

    /// Scrub a frame when it is returned to the allocator, so the next
    /// tenant that receives it cannot read the previous tenant's data.
    /// Dropping the backing memory both zeroes it (semantically) and
    /// returns host RAM.
    pub fn scrub(&mut self, frame: FrameNum) {
        self.frames.remove(&frame.0);
    }

    /// Number of frames that have actually been materialized (test/metric
    /// hook — this is the simulation's real host-RAM footprint).
    pub fn resident_frames(&self) -> usize {
        self.frames.len()
    }
}

impl Default for FrameStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Decompose a physical address into (frame number, in-frame offset) and
/// assert the access does not cross the frame boundary — crossing would
/// mean some caller assumed physical contiguity between frames, which is
/// a logic bug in the *engine*, not bad guest input, hence `debug_assert`.
fn split(addr: VramAddr, len: usize) -> (u64, usize) {
    let frame = addr.0 / FRAME_SIZE;
    let off = (addr.0 % FRAME_SIZE) as usize;
    debug_assert!(
        off + len <= FRAME_SIZE as usize,
        "physical access crosses a frame boundary"
    );
    (frame, off)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_decomposes_non_power_of_two_capacity() {
        // 3 frames = 2 + 1: one order-1 block at 0, one order-0 block at 2.
        let a = VramAllocator::new(3 * FRAME_SIZE);
        assert_eq!(a.free_frames(), 3);
        assert!(a.free[1].contains(&0));
        assert!(a.free[0].contains(&2));
    }

    #[test]
    fn alloc_splits_and_free_coalesces() {
        let mut a = VramAllocator::new(16 * FRAME_SIZE);
        let r1 = a.alloc(1).unwrap(); // splits 16 -> 8,4,2,1 : gets frame 0
        assert_eq!(r1.start, FrameNum(0));
        let r2 = a.alloc(1).unwrap(); // exact-fit from the split leftovers
        assert_eq!(r2.start, FrameNum(1));
        assert_eq!(a.free_frames(), 14);
        a.free(r1);
        a.free(r2);
        // Everything must coalesce back into one 16-frame block.
        assert_eq!(a.free_frames(), 16);
        assert!(a.free[4].contains(&0));
    }

    #[test]
    fn rounds_up_to_power_of_two() {
        let mut a = VramAllocator::new(16 * FRAME_SIZE);
        let r = a.alloc(3).unwrap();
        assert_eq!(r.count, 4); // internal fragmentation, by design
        assert_eq!(a.free_frames(), 12);
    }

    #[test]
    fn exhaustion_reports_free_frames() {
        let mut a = VramAllocator::new(4 * FRAME_SIZE);
        let _r = a.alloc(4).unwrap();
        match a.alloc(1) {
            Err(VgpuError::OutOfVram {
                requested_frames: 1,
                free_frames: 0,
            }) => {}
            other => panic!("expected OutOfVram, got {other:?}"),
        }
    }

    #[test]
    fn fragmentation_can_fail_with_free_space() {
        // Buddy allocators can hold free frames yet fail a large request —
        // this test pins that honest failure mode instead of hiding it.
        let mut a = VramAllocator::new(4 * FRAME_SIZE);
        let r0 = a.alloc(1).unwrap(); // frame 0
        let _r1 = a.alloc(1).unwrap(); // frame 1 (stays busy)
        let _r2 = a.alloc(2).unwrap(); // frames 2-3
        a.free(r0); // frame 0 free, but its buddy (1) is busy -> no merge
        assert_eq!(a.free_frames(), 1);
        assert!(a.alloc(2).is_err()); // 1 frame free, but no order-1 block
    }

    #[test]
    fn store_reads_zero_until_written_and_scrubs() {
        let mut s = FrameStore::new();
        let addr = VramAddr(5 * FRAME_SIZE + 100);
        let mut buf = [0xAAu8; 4];
        s.read(addr, &mut buf);
        assert_eq!(buf, [0, 0, 0, 0]);
        s.write(addr, &[1, 2, 3, 4]);
        s.read(addr, &mut buf);
        assert_eq!(buf, [1, 2, 3, 4]);
        assert_eq!(s.resident_frames(), 1);
        s.scrub(FrameNum(5));
        s.read(addr, &mut buf);
        assert_eq!(buf, [0, 0, 0, 0]);
        assert_eq!(s.resident_frames(), 0);
    }
}
