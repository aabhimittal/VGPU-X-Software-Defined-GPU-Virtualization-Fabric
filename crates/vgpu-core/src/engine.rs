//! The execution engine: what "the GPU runs a command" means here.
//!
//! One contract: take a command whose addresses are guest virtual,
//! translate every byte it touches through the *submitting vGPU's*
//! address space, and only then touch physical VRAM. There is no code
//! path from a command to the `FrameStore` that bypasses translation —
//! grep for `store.` and check for yourself; that greppability is the
//! security argument, kept small on purpose. As of milestone 2 this
//! includes kernels: the interpreter routes every `Ld`/`St` through the
//! same page walk as the copy engine.
//!
//! # Fault atomicity differs by engine class, faithfully
//!
//! Fills and copies validate their whole range *before* touching a byte —
//! a fault writes nothing. Kernels are the opposite and that is honest:
//! a program that faults at instruction 40 has already performed its
//! first 39 instructions' stores, exactly as on real hardware, where
//! kernels are not transactions. The channel dies either way; the
//! difference is what the guest's memory looks like afterwards.

use crate::cmd::Command;
use crate::gmmu::AddressSpace;
use crate::isa::{Instr, REG_COUNT, WATCHDOG_INSTRUCTIONS};
use crate::types::{AccessKind, Cycles, GpuVirtAddr, Result, VgpuError};
use crate::vram::FrameStore;

/// What executing one command did: how long the engine was occupied, and
/// whether the command completed. The two are separate because a faulting
/// command still consumed engine time — reporting `(cycles, Err)` lets
/// the scheduler charge *reality*, so a tenant cannot get cheap
/// scheduling passes by faulting.
pub struct ExecOutcome {
    /// Engine cycles consumed (on fault: time until the fault was
    /// recognized — full command cost for validate-first commands,
    /// executed-instruction cost for kernels).
    pub cycles: Cycles,
    /// Completion or the fault that killed the command.
    pub result: Result<()>,
}

fn ok(cycles: Cycles) -> ExecOutcome {
    ExecOutcome {
        cycles,
        result: Ok(()),
    }
}

fn fault(cycles: Cycles, error: VgpuError) -> ExecOutcome {
    ExecOutcome {
        cycles,
        result: Err(error),
    }
}

/// Execute one command against a vGPU's address space. `FenceSignal` is a
/// no-op here — fences are channel state, and the channel is the caller's
/// to update; the engine only prices it.
pub fn execute(cmd: &Command, aspace: &mut AddressSpace, store: &mut FrameStore) -> ExecOutcome {
    match cmd {
        Command::MemFill { dst, len, value } => {
            // Translate FIRST, for the whole range, before writing byte
            // one (see module docs on fault atomicity).
            match aspace.translate_range(*dst, *len, AccessKind::Write) {
                Ok(segs) => {
                    for (pa, seg_len) in segs {
                        store.fill(pa, seg_len as usize, *value);
                    }
                    aspace.mark_dirty_range(*dst, *len); // migration substrate
                    ok(cmd.cost())
                }
                Err(e) => fault(cmd.cost(), e),
            }
        }
        Command::MemCopy { src, dst, len } => {
            let src_segs = match aspace.translate_range(*src, *len, AccessKind::Read) {
                Ok(s) => s,
                Err(e) => return fault(cmd.cost(), e),
            };
            let dst_segs = match aspace.translate_range(*dst, *len, AccessKind::Write) {
                Ok(s) => s,
                Err(e) => return fault(cmd.cost(), e),
            };
            // Stage through a host-side buffer rather than walking both
            // segment lists in lockstep: trivially correct for
            // overlapping ranges and for src/dst at different page
            // phases, which is the norm.
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
            aspace.mark_dirty_range(*dst, *len); // migration substrate
            ok(cmd.cost())
        }
        Command::KernelLaunch {
            threads,
            args,
            program,
            ..
        } => run_kernel(*threads, args, program, aspace, store),
        Command::FenceSignal { .. } => ok(cmd.cost()),
    }
}

/// The interpreter: `threads` sequential executions of `program`.
///
/// Sequential thread order (0, 1, 2, …) is observably equivalent to the
/// hardware's parallel schedule for data-race-free kernels — the only
/// kernels with defined results on real GPUs either — and it keeps both
/// the results and the cycle accounting deterministic.
///
/// The program was validated at submit (`isa::validate`), so register
/// indices and branch targets are in range by construction; the hot loop
/// indexes without checking. What *cannot* be validated statically —
/// memory addresses (data-dependent) and termination (undecidable) — is
/// handled dynamically: the page walk on every access, the watchdog on
/// every instruction.
fn run_kernel(
    threads: u32,
    args: &[u64],
    program: &[Instr],
    aspace: &mut AddressSpace,
    store: &mut FrameStore,
) -> ExecOutcome {
    let mut cycles: Cycles = 0;
    let mut executed: u64 = 0;

    for tid in 0..threads {
        let mut regs = [0u64; REG_COUNT];
        regs[0] = tid as u64;
        regs[1..=args.len()].copy_from_slice(args);

        let mut pc: usize = 0;
        // Falling off the end of the program halts the thread, like
        // returning from main; `Halt` is the explicit form.
        while pc < program.len() {
            if executed >= WATCHDOG_INSTRUCTIONS {
                return fault(cycles, VgpuError::KernelTimeout { executed });
            }
            executed += 1;
            let instr = &program[pc];
            cycles += instr.cost();
            match instr {
                Instr::Imm { dst, value } => {
                    regs[*dst as usize] = *value;
                    pc += 1;
                }
                Instr::Mov { dst, src } => {
                    regs[*dst as usize] = regs[*src as usize];
                    pc += 1;
                }
                Instr::Add { dst, a, b } => {
                    regs[*dst as usize] = regs[*a as usize].wrapping_add(regs[*b as usize]);
                    pc += 1;
                }
                Instr::Sub { dst, a, b } => {
                    regs[*dst as usize] = regs[*a as usize].wrapping_sub(regs[*b as usize]);
                    pc += 1;
                }
                Instr::Mul { dst, a, b } => {
                    regs[*dst as usize] = regs[*a as usize].wrapping_mul(regs[*b as usize]);
                    pc += 1;
                }
                Instr::Ld { dst, addr, offset } => {
                    let va = regs[*addr as usize].wrapping_add(*offset);
                    match load_u64(aspace, store, va) {
                        Ok(v) => {
                            regs[*dst as usize] = v;
                            pc += 1;
                        }
                        Err(e) => return fault(cycles, e),
                    }
                }
                Instr::St { src, addr, offset } => {
                    let va = regs[*addr as usize].wrapping_add(*offset);
                    match store_u64(aspace, store, va, regs[*src as usize]) {
                        Ok(()) => pc += 1,
                        Err(e) => return fault(cycles, e),
                    }
                }
                Instr::Bnz { cond, target } => {
                    pc = if regs[*cond as usize] != 0 {
                        *target as usize
                    } else {
                        pc + 1
                    };
                }
                Instr::Halt => break,
            }
        }
    }
    ok(cycles.max(1))
}

/// 8-byte guest load: alignment check, page walk, read. An aligned u64
/// can never straddle a 64 KiB frame, which is what lets this be a single
/// translate + single `FrameStore` access (and why misalignment must
/// fault rather than be emulated).
fn load_u64(aspace: &AddressSpace, store: &FrameStore, va: u64) -> Result<u64> {
    let addr = check_aligned(va)?;
    let pa = aspace.translate(addr, AccessKind::Read)?;
    let mut buf = [0u8; 8];
    store.read(pa, &mut buf);
    Ok(u64::from_le_bytes(buf))
}

/// 8-byte guest store: same contract as `load_u64`.
fn store_u64(aspace: &mut AddressSpace, store: &mut FrameStore, va: u64, value: u64) -> Result<()> {
    let addr = check_aligned(va)?;
    let pa = aspace.translate(addr, AccessKind::Write)?;
    store.write(pa, &value.to_le_bytes());
    aspace.mark_dirty_range(addr, 8); // migration substrate
    Ok(())
}

fn check_aligned(va: u64) -> Result<GpuVirtAddr> {
    if va.is_multiple_of(8) {
        Ok(GpuVirtAddr(va))
    } else {
        Err(VgpuError::BadAddress {
            addr: GpuVirtAddr(va),
            why: "misaligned 8-byte kernel access".to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::isa::{busy, vector_add};
    use crate::types::{FrameNum, VgpuError, FRAME_SIZE};

    fn space_with(frames: &[u64]) -> AddressSpace {
        let mut a = AddressSpace::new();
        a.map(GpuVirtAddr(0), frames.iter().copied().map(FrameNum), true)
            .unwrap();
        a
    }

    fn launch(threads: u32, args: Vec<u64>, program: Vec<Instr>) -> Command {
        Command::KernelLaunch {
            name: "test".to_string(),
            threads,
            args,
            program,
        }
    }

    #[test]
    fn copy_across_scattered_frames_moves_the_right_bytes() {
        // Pages 0,1 -> frames 500,2 (deliberately out of order physically).
        let mut aspace = space_with(&[500, 2]);
        let mut store = FrameStore::new();

        let src = GpuVirtAddr(FRAME_SIZE - 2); // straddles the page boundary
        let out = execute(
            &Command::MemFill {
                dst: src,
                len: 4,
                value: 0xAB,
            },
            &mut aspace,
            &mut store,
        );
        out.result.unwrap();
        let dst = GpuVirtAddr(16);
        execute(
            &Command::MemCopy { src, dst, len: 4 },
            &mut aspace,
            &mut store,
        )
        .result
        .unwrap();

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
        let mut aspace = space_with(&[7]);
        let mut store = FrameStore::new();
        let out = execute(
            &Command::MemFill {
                dst: GpuVirtAddr(0),
                len: FRAME_SIZE + 1,
                value: 0xFF,
            },
            &mut aspace,
            &mut store,
        );
        assert!(matches!(out.result, Err(VgpuError::PageFault { .. })));
        assert_eq!(
            store.resident_frames(),
            0,
            "no bytes may be written before a fault"
        );
    }

    #[test]
    fn busy_kernel_costs_exactly_its_cycles() {
        let mut aspace = AddressSpace::new();
        let mut store = FrameStore::new();
        let out = execute(&launch(1, vec![], busy(1234)), &mut aspace, &mut store);
        out.result.unwrap();
        assert_eq!(out.cycles, 1234);

        // Cost scales linearly with thread count.
        let out = execute(&launch(3, vec![], busy(100)), &mut aspace, &mut store);
        out.result.unwrap();
        assert_eq!(out.cycles, 300);
    }

    #[test]
    fn vector_add_computes_through_the_gmmu() {
        // Three arrays of 100 u64s in one mapped page.
        let mut aspace = space_with(&[42]);
        let mut store = FrameStore::new();
        let (a, b, c) = (0u64, 800u64, 1600u64);
        for i in 0..100u64 {
            store_u64(&mut aspace, &mut store, a + i * 8, i).unwrap();
            store_u64(&mut aspace, &mut store, b + i * 8, 1000 + i).unwrap();
        }

        let out = execute(
            &launch(100, vec![a, b, c], vector_add()),
            &mut aspace,
            &mut store,
        );
        out.result.unwrap();

        for i in 0..100u64 {
            let got = load_u64(&aspace, &store, c + i * 8).unwrap();
            assert_eq!(got, 1000 + 2 * i, "c[{i}]");
        }
    }

    #[test]
    fn kernel_wild_pointer_faults_but_keeps_prior_stores() {
        let mut aspace = space_with(&[3]);
        let mut store = FrameStore::new();
        // Thread stores to a valid address, then dereferences an unmapped
        // one: the fault must surface AND the first store must remain —
        // kernels are not transactions (module docs).
        let program = vec![
            Instr::Imm { dst: 1, value: 64 }, // valid VA
            Instr::Imm { dst: 2, value: 7 },  // value
            Instr::St {
                src: 2,
                addr: 1,
                offset: 0,
            },
            Instr::Imm {
                dst: 3,
                value: FRAME_SIZE * 8,
            }, // unmapped VA
            Instr::Ld {
                dst: 4,
                addr: 3,
                offset: 0,
            },
            Instr::Halt,
        ];
        let out = execute(&launch(1, vec![], program), &mut aspace, &mut store);
        assert!(matches!(out.result, Err(VgpuError::PageFault { .. })));
        assert_eq!(load_u64(&aspace, &store, 64).unwrap(), 7);
    }

    #[test]
    fn watchdog_kills_infinite_loops() {
        let mut aspace = AddressSpace::new();
        let mut store = FrameStore::new();
        // r1 = 1; loop: if r1 != 0 goto loop  — never halts.
        let program = vec![
            Instr::Imm { dst: 1, value: 1 },
            Instr::Bnz { cond: 1, target: 1 },
        ];
        let out = execute(&launch(1, vec![], program), &mut aspace, &mut store);
        match out.result {
            Err(VgpuError::KernelTimeout { executed }) => {
                assert_eq!(executed, WATCHDOG_INSTRUCTIONS);
            }
            other => panic!("expected KernelTimeout, got {other:?}"),
        }
        // The runaway kernel was still *charged* for the cycles it burned.
        assert!(out.cycles >= WATCHDOG_INSTRUCTIONS);
    }

    #[test]
    fn misaligned_kernel_access_faults() {
        let mut aspace = space_with(&[3]);
        let mut store = FrameStore::new();
        let program = vec![
            Instr::Imm { dst: 1, value: 13 }, // not 8-byte aligned
            Instr::Ld {
                dst: 2,
                addr: 1,
                offset: 0,
            },
            Instr::Halt,
        ];
        let out = execute(&launch(1, vec![], program), &mut aspace, &mut store);
        assert!(matches!(out.result, Err(VgpuError::BadAddress { .. })));
    }

    #[test]
    fn counted_loop_terminates_and_costs_deterministically() {
        let mut aspace = AddressSpace::new();
        let mut store = FrameStore::new();
        // r1 = 10; r2 = 1; loop { r1 -= r2 } while r1 != 0
        let program = vec![
            Instr::Imm { dst: 1, value: 10 },
            Instr::Imm { dst: 2, value: 1 },
            Instr::Sub { dst: 1, a: 1, b: 2 },
            Instr::Bnz { cond: 1, target: 2 },
            Instr::Halt,
        ];
        let out = execute(&launch(1, vec![], program.clone()), &mut aspace, &mut store);
        out.result.unwrap();
        // 2 setup + 10×(Sub+Bnz) + Halt = 23 cycles, every run.
        assert_eq!(out.cycles, 23);
        let again = execute(&launch(1, vec![], program), &mut aspace, &mut store);
        assert_eq!(again.cycles, 23);
    }
}
