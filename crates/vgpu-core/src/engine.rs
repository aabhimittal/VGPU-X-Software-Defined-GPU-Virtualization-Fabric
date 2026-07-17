//! The execution engine: what "the GPU runs a command" means here.
//!
//! One function, one contract: take a command whose addresses are guest
//! virtual, translate every byte it touches through the *submitting vGPU's*
//! address space, and only then touch physical VRAM. There is no code path
//! from a command to the `FrameStore` that bypasses `translate_range` —
//! grep for `store.` and check for yourself; that greppability is the
//! security argument, kept small on purpose.
//!
//! Faults are returned, not panicked: a guest that submits a bad pointer
//! has its *channel* killed by the caller (see `node.rs`), never the node.

use crate::cmd::Command;
use crate::gmmu::AddressSpace;
use crate::types::{AccessKind, Cycles, Result};
use crate::vram::FrameStore;

/// Execute one command against a vGPU's address space, returning the
/// cycles it consumed. `FenceSignal` is a no-op here — fences are channel
/// state, and the channel is the caller's to update; the engine only
/// prices it.
pub fn execute(cmd: &Command, aspace: &AddressSpace, store: &mut FrameStore) -> Result<Cycles> {
    match cmd {
        Command::MemFill { dst, len, value } => {
            // Translate FIRST, for the whole range, before writing byte
            // one. A fill that faults halfway through would leave guest
            // memory in a state the guest cannot reason about; hardware
            // copy engines validate the page range up front for the same
            // reason (fault-and-replay is a Volta+ luxury we don't model).
            let segs = aspace.translate_range(*dst, *len, AccessKind::Write)?;
            for (pa, seg_len) in segs {
                store.fill(pa, seg_len as usize, *value);
            }
        }
        Command::MemCopy { src, dst, len } => {
            let src_segs = aspace.translate_range(*src, *len, AccessKind::Read)?;
            let dst_segs = aspace.translate_range(*dst, *len, AccessKind::Write)?;
            // Stage through a host-side buffer rather than walking both
            // segment lists in lockstep. A real copy engine streams, but
            // the staging version is trivially correct for overlapping
            // ranges *and* for the case where src and dst segment
            // boundaries don't line up — which is the norm, since the two
            // VAs sit at different page offsets.
            let mut data = Vec::with_capacity(*len as usize);
            for (pa, seg_len) in src_segs {
                let start = data.len();
                data.resize(start + seg_len as usize, 0);
                store.read(pa, &mut data[start..]);
            }
            let mut cursor = 0usize;
            for (pa, seg_len) in dst_segs {
                store.write(pa, &data[cursor..cursor + seg_len as usize]);
                cursor += seg_len as usize;
            }
        }
        Command::KernelLaunch { .. } => {
            // Foundation milestone: a kernel is an opaque cost. Executing
            // actual programs against guest memory is the milestone-2
            // interpreter's job; the scheduler and isolation stories are
            // complete without it because neither ever looks inside one.
        }
        Command::FenceSignal { .. } => {}
    }
    Ok(cmd.cost())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{FrameNum, GpuVirtAddr, VgpuError, FRAME_SIZE};

    fn space_with(frames: &[u64]) -> AddressSpace {
        let mut a = AddressSpace::new();
        a.map(GpuVirtAddr(0), frames.iter().copied().map(FrameNum), true)
            .unwrap();
        a
    }

    #[test]
    fn copy_across_scattered_frames_moves_the_right_bytes() {
        // Pages 0,1 -> frames 500,2 (deliberately out of order physically).
        let aspace = space_with(&[500, 2]);
        let mut store = FrameStore::new();

        // Write a pattern straddling the page boundary via a fill+copy.
        let src = GpuVirtAddr(FRAME_SIZE - 2); // last 2 bytes of page 0 ...
        execute(
            &Command::MemFill {
                dst: src,
                len: 4,
                value: 0xAB,
            },
            &aspace,
            &mut store,
        )
        .unwrap(); // ... and first 2 of page 1
        let dst = GpuVirtAddr(16);
        execute(&Command::MemCopy { src, dst, len: 4 }, &aspace, &mut store).unwrap();

        // Verify through translation, byte by byte.
        for i in 0..4 {
            let pa = aspace
                .translate(GpuVirtAddr(16 + i), AccessKind::Read)
                .unwrap();
            let mut b = [0u8; 1];
            store.read(pa, &mut b);
            assert_eq!(b[0], 0xAB, "byte {i}");
        }
    }

    #[test]
    fn fill_faults_atomically_on_unmapped_tail() {
        // One mapped page; a fill that runs off its end must fault without
        // writing anything.
        let aspace = space_with(&[7]);
        let mut store = FrameStore::new();
        let err = execute(
            &Command::MemFill {
                dst: GpuVirtAddr(0),
                len: FRAME_SIZE + 1,
                value: 0xFF,
            },
            &aspace,
            &mut store,
        )
        .unwrap_err();
        assert!(matches!(err, VgpuError::PageFault { .. }));
        assert_eq!(
            store.resident_frames(),
            0,
            "no bytes may be written before a fault"
        );
    }

    #[test]
    fn kernel_launch_costs_its_declared_cycles() {
        let aspace = AddressSpace::new();
        let mut store = FrameStore::new();
        let c = execute(
            &Command::KernelLaunch {
                name: "gemm",
                cost: 1234,
            },
            &aspace,
            &mut store,
        )
        .unwrap();
        assert_eq!(c, 1234);
    }
}
