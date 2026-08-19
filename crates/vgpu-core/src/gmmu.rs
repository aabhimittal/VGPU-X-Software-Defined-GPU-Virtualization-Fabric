//! The GMMU: per-vGPU guest-virtual → physical-VRAM translation.
//!
//! This module is the isolation boundary of the whole system. A vGPU never
//! names physical memory; every address in every command it submits is a
//! guest virtual address, and the only way it becomes a physical address is
//! through *this* page walk, over page tables that only the device model
//! (never the guest) can write. That is precisely the contract real GPU
//! hardware enforces: on NVIDIA parts the GMMU walks per-channel page
//! tables and a VA with no valid PTE raises an engine fault — the guest
//! physically cannot express "read another tenant's VRAM".
//!
//! # Layout of a guest virtual address
//!
//! We use 64 KiB pages and a two-level table (real GMMUs use 4-5 levels to
//! cover 49-bit spaces; two levels keeps the walk teachable while keeping
//! every structural property that matters):
//!
//! ```text
//!   bit  35                26 25              16 15               0
//!        +------------------+------------------+------------------+
//!        |  PDE index (10b) |  PTE index (10b) |   offset (16b)   |
//!        +------------------+------------------+------------------+
//! ```
//!
//! * offset: 16 bits because `FRAME_SIZE` = 64 KiB = 2^16.
//! * 10 + 10 bits of table index → 1024-entry directory of 1024-entry
//!   tables → 2^20 pages → a 64 GiB virtual space per vGPU.
//!
//! # Why lazy second-level tables?
//!
//! A fully materialized table would be 1M entries per vGPU even for a
//! guest that mapped one page. Directories start empty and second-level
//! tables are allocated on first map into their 64 MiB region — the same
//! reason CPU and GPU page tables are radix trees and not flat arrays:
//! the occupied portion of a huge sparse space is what should cost memory.

use crate::types::{AccessKind, FrameNum, GpuVirtAddr, Result, VgpuError, VramAddr, FRAME_SIZE};

/// log2(FRAME_SIZE): number of offset bits.
const OFFSET_BITS: u32 = 16;
/// Bits of index into a second-level page table.
const PTE_BITS: u32 = 10;
/// Bits of index into the page directory.
const PDE_BITS: u32 = 10;
/// Entries per table / per directory.
const TABLE_ENTRIES: usize = 1 << PTE_BITS;
/// Total virtual-address bits this GMMU decodes.
pub const VA_BITS: u32 = OFFSET_BITS + PTE_BITS + PDE_BITS; // 36 -> 64 GiB
/// One past the highest legal guest virtual address.
pub const VA_LIMIT: u64 = 1 << VA_BITS;

/// One page table entry.
///
/// Real PTEs pack frame number + permission bits + caching attributes into
/// a single u64 for the hardware walker; we keep named fields because no
/// hardware reads this struct and the compiler packs it well enough. The
/// *semantics* — valid bit, writable bit, frame number — are the real ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Pte {
    frame: FrameNum,
    writable: bool,
    /// The hardware "dirty" (accessed/dirty) bit, repurposed exactly the
    /// way live migration repurposes it on real MMUs: set when the page
    /// is written, harvested and cleared by `take_dirty`. Starts `true`
    /// on map — a page that has never been copied anywhere is, from a
    /// migrator's point of view, dirty by definition.
    dirty: bool,
}

/// A second-level table: 1024 slots, each possibly holding a valid PTE.
/// `Option<Pte>` plays the role of the hardware "valid" bit.
struct PageTable {
    entries: Box<[Option<Pte>; TABLE_ENTRIES]>,
    /// Count of valid entries, so unmap can free empty tables in O(1)
    /// instead of scanning 1024 slots.
    live: u32,
}

impl PageTable {
    fn new() -> Self {
        Self {
            entries: Box::new([None; TABLE_ENTRIES]),
            live: 0,
        }
    }
}

/// A vGPU's whole address space: the page directory plus bookkeeping.
pub struct AddressSpace {
    /// First level. `None` = no table for that 64 MiB region yet.
    directory: Vec<Option<PageTable>>,
    /// Pages currently mapped (metric + budget hooks).
    mapped_pages: u64,
}

/// Split a guest VA into its three hardware fields. Returns an error for
/// addresses beyond the decodable range — hardware would fault the same
/// way, since it has no wires for those bits.
fn split_va(addr: GpuVirtAddr) -> Result<(usize, usize, u64)> {
    if addr.0 >= VA_LIMIT {
        return Err(VgpuError::BadAddress {
            addr,
            why: "beyond 36-bit VA space".to_string(),
        });
    }
    let offset = addr.0 & ((1 << OFFSET_BITS) - 1);
    let pte_idx = ((addr.0 >> OFFSET_BITS) & ((1 << PTE_BITS) - 1)) as usize;
    let pde_idx = ((addr.0 >> (OFFSET_BITS + PTE_BITS)) & ((1 << PDE_BITS) - 1)) as usize;
    Ok((pde_idx, pte_idx, offset))
}

impl AddressSpace {
    /// Fresh, empty address space: every access faults until mapped.
    pub fn new() -> Self {
        let mut directory = Vec::with_capacity(1 << PDE_BITS);
        directory.resize_with(1 << PDE_BITS, || None);
        Self {
            directory,
            mapped_pages: 0,
        }
    }

    /// Number of currently mapped pages.
    pub fn mapped_pages(&self) -> u64 {
        self.mapped_pages
    }

    /// Map `frames` starting at page-aligned `base`, one PTE per frame.
    ///
    /// The frames need not be physically contiguous — that is the entire
    /// point of paging: the guest sees one contiguous VA range backed by
    /// whatever scattered physical frames the buddy allocator produced.
    ///
    /// Mapping is all-or-nothing: we validate the whole range for overlap
    /// *before* touching any PTE, so a failed map never leaves a
    /// half-mapped range behind (a half-applied mapping is the kind of
    /// state a control plane can never safely retry against).
    pub fn map(
        &mut self,
        base: GpuVirtAddr,
        frames: impl Iterator<Item = FrameNum> + Clone,
        writable: bool,
    ) -> Result<()> {
        if !base.0.is_multiple_of(FRAME_SIZE) {
            return Err(VgpuError::BadAddress {
                addr: base,
                why: "map base not page-aligned".to_string(),
            });
        }

        // Pass 1: pure validation, no mutation.
        for (i, _f) in frames.clone().enumerate() {
            let va = GpuVirtAddr(base.0.checked_add(i as u64 * FRAME_SIZE).ok_or(
                VgpuError::BadAddress {
                    addr: base,
                    why: "VA overflow".to_string(),
                },
            )?);
            let (pde, pte, _) = split_va(va)?;
            if let Some(table) = &self.directory[pde] {
                if table.entries[pte].is_some() {
                    return Err(VgpuError::AlreadyMapped { addr: va });
                }
            }
        }

        // Pass 2: guaranteed to succeed — mutate.
        for (i, frame) in frames.enumerate() {
            let va = GpuVirtAddr(base.0 + i as u64 * FRAME_SIZE);
            let (pde, pte, _) = split_va(va).expect("validated in pass 1");
            let table = self.directory[pde].get_or_insert_with(PageTable::new);
            table.entries[pte] = Some(Pte {
                frame,
                writable,
                dirty: true, // never-copied = dirty (see Pte docs)
            });
            table.live += 1;
            self.mapped_pages += 1;
        }
        Ok(())
    }

    /// Unmap `page_count` pages starting at `base`, returning the frames
    /// that backed them (the caller returns those to the allocator and
    /// scrubs them — the GMMU does not own physical memory, it only
    /// references it; ownership stays with the vGPU's allocation records).
    pub fn unmap(&mut self, base: GpuVirtAddr, page_count: u64) -> Result<Vec<FrameNum>> {
        if !base.0.is_multiple_of(FRAME_SIZE) {
            return Err(VgpuError::BadAddress {
                addr: base,
                why: "unmap base not page-aligned".to_string(),
            });
        }
        // Pass 1: every page must currently be mapped (all-or-nothing).
        for i in 0..page_count {
            let va = GpuVirtAddr(base.0 + i * FRAME_SIZE);
            let (pde, pte, _) = split_va(va)?;
            let mapped = self.directory[pde]
                .as_ref()
                .is_some_and(|t| t.entries[pte].is_some());
            if !mapped {
                return Err(VgpuError::NotMapped { addr: va });
            }
        }
        // Pass 2: clear PTEs, free empty tables, collect frames.
        let mut freed = Vec::with_capacity(page_count as usize);
        for i in 0..page_count {
            let va = GpuVirtAddr(base.0 + i * FRAME_SIZE);
            let (pde, pte, _) = split_va(va).expect("validated in pass 1");
            let table = self.directory[pde].as_mut().expect("validated");
            let entry = table.entries[pte].take().expect("validated");
            table.live -= 1;
            self.mapped_pages -= 1;
            freed.push(entry.frame);
            if table.live == 0 {
                // Drop the empty table so a sparse guest stays sparse.
                self.directory[pde] = None;
            }
        }
        Ok(freed)
    }

    /// The page walk. Byte-granular: returns the physical address for one
    /// guest byte, or a page fault.
    ///
    /// Write access to a read-only page faults just like an unmapped page —
    /// from the guest's perspective both are "the GMMU said no", and
    /// reporting which is which to the *guest* would leak mapping state.
    /// The distinction lives in the host-side fault record instead.
    pub fn translate(&self, addr: GpuVirtAddr, access: AccessKind) -> Result<VramAddr> {
        let (pde, pte, offset) =
            split_va(addr).map_err(|_| VgpuError::PageFault { addr, access })?;
        let table = self.directory[pde]
            .as_ref()
            .ok_or(VgpuError::PageFault { addr, access })?;
        let entry = table.entries[pte].ok_or(VgpuError::PageFault { addr, access })?;
        if matches!(access, AccessKind::Write) && !entry.writable {
            return Err(VgpuError::PageFault { addr, access });
        }
        Ok(VramAddr(entry.frame.base_addr().0 + offset))
    }

    /// Translate a byte span into per-frame physical segments.
    ///
    /// This is the software analogue of scatter-gather DMA: a copy engine
    /// given a VA range must break it at page boundaries because adjacent
    /// virtual pages map to arbitrary physical frames. Every memory-touching
    /// command in `engine.rs` goes through this — nothing in the crate ever
    /// assumes two consecutive virtual pages are physically adjacent.
    pub fn translate_range(
        &self,
        addr: GpuVirtAddr,
        len: u64,
        access: AccessKind,
    ) -> Result<Vec<(VramAddr, u64)>> {
        let mut segs = Vec::new();
        let mut cur = addr.0;
        let end = cur.checked_add(len).ok_or(VgpuError::BadAddress {
            addr,
            why: "range overflows u64".to_string(),
        })?;
        while cur < end {
            let va = GpuVirtAddr(cur);
            let pa = self.translate(va, access)?;
            // Bytes until the next page boundary, capped by bytes remaining.
            let in_page = FRAME_SIZE - va.frame_offset();
            let take = in_page.min(end - cur);
            segs.push((pa, take));
            cur += take;
        }
        Ok(segs)
    }

    // -- dirty tracking (the migration substrate) ---------------------------

    /// Set the dirty bit on every page overlapping `[addr, addr+len)`.
    ///
    /// Callers are the write paths — the engine's fill/copy/kernel-store
    /// and the node's host DMA — invoked *after* a successful write
    /// translation. The invariant "every write marks" is load-bearing for
    /// migration correctness (a missed mark is silent post-migration
    /// corruption), so it is pinned by tests that exercise every write
    /// path and assert the page shows up dirty
    /// (`every_write_path_marks_dirty` in `fabric.rs`).
    ///
    /// Pages in the range are expected to be mapped (the write was just
    /// translated); an unmapped page here is an engine bug, hence
    /// `debug_assert`, not `Err`.
    pub fn mark_dirty_range(&mut self, addr: GpuVirtAddr, len: u64) {
        if len == 0 {
            return; // a zero-byte write dirties nothing
        }
        let mut cur = addr.0 - addr.frame_offset(); // round down to page
        let end = addr.0.saturating_add(len);
        while cur < end {
            let Ok((pde, pte, _)) = split_va(GpuVirtAddr(cur)) else {
                debug_assert!(false, "mark_dirty_range beyond VA space");
                return;
            };
            let entry = self.directory[pde]
                .as_mut()
                .and_then(|t| t.entries[pte].as_mut());
            match entry {
                Some(e) => e.dirty = true,
                None => debug_assert!(false, "mark_dirty_range on unmapped page"),
            }
            cur += FRAME_SIZE;
        }
    }

    /// Harvest and clear the dirty set: the page-aligned guest VA of every
    /// page written (or newly mapped) since the previous call. This is the
    /// read-and-reset cycle every pre-copy migration loop performs against
    /// MMU dirty bits; clearing on read is what makes successive rounds
    /// converge to "what changed while I was copying".
    pub fn take_dirty(&mut self) -> Vec<GpuVirtAddr> {
        let mut dirty = Vec::new();
        for (pde_idx, slot) in self.directory.iter_mut().enumerate() {
            let Some(table) = slot else { continue };
            for (pte_idx, entry) in table.entries.iter_mut().enumerate() {
                if let Some(pte) = entry {
                    if pte.dirty {
                        pte.dirty = false;
                        let va = ((pde_idx as u64) << (OFFSET_BITS + PTE_BITS))
                            | ((pte_idx as u64) << OFFSET_BITS);
                        dirty.push(GpuVirtAddr(va));
                    }
                }
            }
        }
        dirty
    }
}

impl Default for AddressSpace {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frames(list: &[u64]) -> impl Iterator<Item = FrameNum> + Clone + '_ {
        list.iter().copied().map(FrameNum)
    }

    #[test]
    fn unmapped_access_faults() {
        let a = AddressSpace::new();
        let err = a
            .translate(GpuVirtAddr(0x1_0000), AccessKind::Read)
            .unwrap_err();
        assert!(matches!(err, VgpuError::PageFault { .. }));
    }

    #[test]
    fn map_translate_roundtrip_with_scattered_frames() {
        let mut a = AddressSpace::new();
        // Two virtually-contiguous pages backed by wildly separated frames.
        a.map(GpuVirtAddr(0x2_0000), frames(&[7, 4242]), true)
            .unwrap();
        let pa0 = a
            .translate(GpuVirtAddr(0x2_0010), AccessKind::Read)
            .unwrap();
        assert_eq!(pa0, VramAddr(7 * FRAME_SIZE + 0x10));
        let pa1 = a
            .translate(GpuVirtAddr(0x3_0000), AccessKind::Write)
            .unwrap();
        assert_eq!(pa1, VramAddr(4242 * FRAME_SIZE));
    }

    #[test]
    fn write_to_readonly_page_faults() {
        let mut a = AddressSpace::new();
        a.map(GpuVirtAddr(0), frames(&[1]), false).unwrap();
        assert!(a.translate(GpuVirtAddr(0), AccessKind::Read).is_ok());
        let err = a.translate(GpuVirtAddr(0), AccessKind::Write).unwrap_err();
        assert!(matches!(
            err,
            VgpuError::PageFault {
                access: AccessKind::Write,
                ..
            }
        ));
    }

    #[test]
    fn overlapping_map_is_rejected_atomically() {
        let mut a = AddressSpace::new();
        a.map(GpuVirtAddr(0x1_0000), frames(&[10]), true).unwrap();
        // Second map covers pages 0 and 1; page 1 collides. Nothing may
        // change — page 0 must still be unmapped afterwards.
        let err = a.map(GpuVirtAddr(0), frames(&[20, 21]), true).unwrap_err();
        assert!(matches!(err, VgpuError::AlreadyMapped { .. }));
        assert!(a.translate(GpuVirtAddr(0), AccessKind::Read).is_err());
        assert_eq!(a.mapped_pages(), 1);
    }

    #[test]
    fn unmap_returns_frames_and_frees_empty_tables() {
        let mut a = AddressSpace::new();
        a.map(GpuVirtAddr(0), frames(&[3, 9]), true).unwrap();
        let freed = a.unmap(GpuVirtAddr(0), 2).unwrap();
        assert_eq!(freed, vec![FrameNum(3), FrameNum(9)]);
        assert_eq!(a.mapped_pages(), 0);
        assert!(a.directory.iter().all(|t| t.is_none()));
    }

    #[test]
    fn translate_range_splits_at_page_boundaries() {
        let mut a = AddressSpace::new();
        a.map(GpuVirtAddr(0), frames(&[100, 50]), true).unwrap();
        // 12 bytes spanning the boundary between pages 0 and 1.
        let segs = a
            .translate_range(GpuVirtAddr(FRAME_SIZE - 4), 12, AccessKind::Read)
            .unwrap();
        assert_eq!(
            segs,
            vec![
                (VramAddr(100 * FRAME_SIZE + FRAME_SIZE - 4), 4),
                (VramAddr(50 * FRAME_SIZE), 8),
            ]
        );
    }

    #[test]
    fn va_beyond_limit_faults() {
        let a = AddressSpace::new();
        assert!(a
            .translate(GpuVirtAddr(VA_LIMIT), AccessKind::Read)
            .is_err());
    }
}
