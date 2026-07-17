//! The kernel ISA: the smallest instruction set that makes `KernelLaunch`
//! honest.
//!
//! Until now a kernel was an opaque cycle cost — enough for the scheduler,
//! which never looks inside one, but a hole in the isolation story: the
//! claim "every byte a kernel touches goes through the GMMU" was true
//! vacuously. This module closes it. A kernel is now a real program over
//! registers and memory, and the interpreter in `engine.rs` routes every
//! load and store through the same `translate` walk as every copy engine.
//!
//! # Design constraints, in order
//!
//! 1. **Every memory access is a guest VA** — the ISA cannot even spell a
//!    physical address (registers hold u64s; only Ld/St interpret one as
//!    an address, and the interpreter translates it).
//! 2. **Deterministic cost** — each instruction has a fixed cycle price,
//!    so scheduler tests stay exact with real programs.
//! 3. **Small enough to verify by eye** — 9 instructions. This is a
//!    teaching ISA in the spirit of SASS/PTX reduced to essentials:
//!    ALU, memory, one conditional branch, halt. Everything else
//!    (call stacks, predication, shared memory, warps) is deliberately
//!    absent until a milestone needs it.
//!
//! # Execution model
//!
//! A launch runs `threads` copies of the same program (CUDA's flat grid,
//! collapsed to one dimension). Each thread gets `REG_COUNT` zeroed
//! registers, then `r0 = thread index` and `r1..` = launch arguments —
//! the moral equivalent of `threadIdx` and kernel parameters. Threads
//! execute sequentially, thread 0 first: for data-race-free kernels
//! (the only ones with defined results on real hardware too) this is
//! observably equivalent to the massively parallel schedule, and it keeps
//! the interpreter — and the fairness accounting — deterministic.

use crate::types::{Result, VgpuError};

/// Registers per thread. 16 is enough for real small kernels and keeps
/// register indices trivially validatable in a `u8`.
pub const REG_COUNT: usize = 16;

/// Per-launch instruction budget — the watchdog. Real GPUs enforce
/// exactly this (Windows TDR, Linux's ~10s job timeouts): a kernel that
/// never halts must be killed by the *device*, because no guest can be
/// trusted to kill it. Exceeding the budget faults the channel with
/// `KernelTimeout`; the node and other tenants are untouched.
pub const WATCHDOG_INSTRUCTIONS: u64 = 1_000_000;

/// One instruction. `dst`/`src`/register operands are indices into the
/// thread's register file; `validate` bounds-checks them once at submit
/// so the interpreter's hot loop can index without checking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Instr {
    /// `r[dst] = value`
    Imm {
        /// Destination register.
        dst: u8,
        /// Immediate value.
        value: u64,
    },
    /// `r[dst] = r[src]`
    Mov {
        /// Destination register.
        dst: u8,
        /// Source register.
        src: u8,
    },
    /// `r[dst] = r[a] + r[b]` (wrapping, like hardware).
    Add {
        /// Destination register.
        dst: u8,
        /// Left operand register.
        a: u8,
        /// Right operand register.
        b: u8,
    },
    /// `r[dst] = r[a] - r[b]` (wrapping).
    Sub {
        /// Destination register.
        dst: u8,
        /// Left operand register.
        a: u8,
        /// Right operand register.
        b: u8,
    },
    /// `r[dst] = r[a] * r[b]` (wrapping).
    Mul {
        /// Destination register.
        dst: u8,
        /// Left operand register.
        a: u8,
        /// Right operand register.
        b: u8,
    },
    /// `r[dst] = load_u64(r[addr] + offset)` — a *guest VA*, translated
    /// through the submitting vGPU's page tables. Must be 8-byte aligned
    /// (misalignment faults, as on real hardware).
    Ld {
        /// Destination register.
        dst: u8,
        /// Register holding the base guest VA.
        addr: u8,
        /// Byte offset added to the base.
        offset: u64,
    },
    /// `store_u64(r[addr] + offset, r[src])` — same VA rules as `Ld`.
    St {
        /// Register holding the value to store.
        src: u8,
        /// Register holding the base guest VA.
        addr: u8,
        /// Byte offset added to the base.
        offset: u64,
    },
    /// `if r[cond] != 0 { pc = target }` — the one control-flow op.
    /// With subtraction it builds counted loops; that is all a teaching
    /// ISA needs to be Turing-adjacent within the watchdog.
    Bnz {
        /// Register tested against zero.
        cond: u8,
        /// Absolute instruction index to jump to.
        target: u16,
    },
    /// End this thread.
    Halt,
}

impl Instr {
    /// Cycle price. ALU/control = 1; memory = 4 (memory is slower than
    /// arithmetic everywhere, and a visible ratio makes kernel costs
    /// teach the same lesson real profilers do).
    pub fn cost(&self) -> u64 {
        match self {
            Instr::Ld { .. } | Instr::St { .. } => 4,
            _ => 1,
        }
    }
}

/// Validate a program once, at submit time: register indices in range,
/// branch targets within the program, program non-empty, and enough
/// registers to receive `arg_count` arguments after `r0` (the thread
/// index).
///
/// Submit-time validation means the interpreter never re-checks these in
/// its hot loop *and* a malformed program is rejected as a typed error
/// before it ever occupies the engine — the same reason command front
/// ends validate packet headers at fetch, not at execute.
pub fn validate(program: &[Instr], arg_count: usize) -> Result<()> {
    fn reg(r: u8, what: &str) -> Result<()> {
        if (r as usize) < REG_COUNT {
            Ok(())
        } else {
            Err(VgpuError::BadProgram {
                why: format!("register r{r} out of range in {what}"),
            })
        }
    }

    if program.is_empty() {
        return Err(VgpuError::BadProgram {
            why: "empty program".to_string(),
        });
    }
    if program.len() > u16::MAX as usize {
        return Err(VgpuError::BadProgram {
            why: "program exceeds u16 address space".to_string(),
        });
    }
    if 1 + arg_count > REG_COUNT {
        return Err(VgpuError::BadProgram {
            why: format!("{arg_count} args do not fit after r0 in {REG_COUNT} registers"),
        });
    }
    for (pc, instr) in program.iter().enumerate() {
        match instr {
            Instr::Imm { dst, .. } => reg(*dst, "Imm")?,
            Instr::Mov { dst, src } => {
                reg(*dst, "Mov")?;
                reg(*src, "Mov")?;
            }
            Instr::Add { dst, a, b } | Instr::Sub { dst, a, b } | Instr::Mul { dst, a, b } => {
                reg(*dst, "ALU")?;
                reg(*a, "ALU")?;
                reg(*b, "ALU")?;
            }
            Instr::Ld { dst, addr, .. } => {
                reg(*dst, "Ld")?;
                reg(*addr, "Ld")?;
            }
            Instr::St { src, addr, .. } => {
                reg(*src, "St")?;
                reg(*addr, "St")?;
            }
            Instr::Bnz { cond, target } => {
                reg(*cond, "Bnz")?;
                if *target as usize >= program.len() {
                    return Err(VgpuError::BadProgram {
                        why: format!("branch target {target} outside program at pc {pc}"),
                    });
                }
            }
            Instr::Halt => {}
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Stock kernels — used by tests, docs, and the shim examples.
// ---------------------------------------------------------------------------

/// A compute-only kernel costing exactly `cycles` cycles for one thread:
/// `cycles - 1` no-op ALU instructions plus `Halt`. The exact-cost
/// property is what keeps the scheduler fairness tests assertable to the
/// cycle with *real* programs instead of opaque cost stubs.
///
/// Panics if `cycles == 0` (a kernel cannot cost less than its `Halt`).
pub fn busy(cycles: u64) -> Vec<Instr> {
    assert!(cycles >= 1, "minimum kernel cost is the Halt instruction");
    let mut program = Vec::with_capacity(cycles as usize);
    for _ in 0..cycles - 1 {
        program.push(Instr::Imm { dst: 15, value: 0 });
    }
    program.push(Instr::Halt);
    program
}

/// The canonical first kernel: `c[i] = a[i] + b[i]` over u64 elements.
/// Launch with `threads = n` and args `[a, b, c]` (guest VAs of the three
/// arrays), which arrive in `r1`, `r2`, `r3`; `r0` is the element index.
pub fn vector_add() -> Vec<Instr> {
    vec![
        Instr::Imm { dst: 4, value: 8 },   // r4 = sizeof(u64)
        Instr::Mul { dst: 5, a: 0, b: 4 }, // r5 = i * 8
        Instr::Add { dst: 6, a: 1, b: 5 }, // r6 = &a[i]
        Instr::Ld {
            dst: 7,
            addr: 6,
            offset: 0,
        }, // r7 = a[i]
        Instr::Add { dst: 8, a: 2, b: 5 }, // r8 = &b[i]
        Instr::Ld {
            dst: 9,
            addr: 8,
            offset: 0,
        }, // r9 = b[i]
        Instr::Add {
            dst: 10,
            a: 7,
            b: 9,
        }, // r10 = a[i] + b[i]
        Instr::Add {
            dst: 11,
            a: 3,
            b: 5,
        }, // r11 = &c[i]
        Instr::St {
            src: 10,
            addr: 11,
            offset: 0,
        }, // c[i] = r10
        Instr::Halt,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stock_kernels_validate() {
        validate(&busy(1), 0).unwrap();
        validate(&busy(100), 0).unwrap();
        validate(&vector_add(), 3).unwrap();
    }

    #[test]
    fn busy_costs_exactly_what_it_says() {
        for n in [1u64, 2, 100] {
            let total: u64 = busy(n).iter().map(Instr::cost).sum();
            assert_eq!(total, n);
        }
    }

    #[test]
    fn bad_register_and_bad_branch_are_rejected() {
        let err = validate(&[Instr::Imm { dst: 16, value: 0 }], 0).unwrap_err();
        assert!(matches!(err, VgpuError::BadProgram { .. }));

        let err = validate(&[Instr::Bnz { cond: 0, target: 5 }, Instr::Halt], 0).unwrap_err();
        assert!(matches!(err, VgpuError::BadProgram { .. }));
    }

    #[test]
    fn too_many_args_are_rejected() {
        let err = validate(&[Instr::Halt], REG_COUNT).unwrap_err();
        assert!(matches!(err, VgpuError::BadProgram { .. }));
    }

    #[test]
    fn empty_program_is_rejected() {
        assert!(matches!(
            validate(&[], 0),
            Err(VgpuError::BadProgram { .. })
        ));
    }
}
