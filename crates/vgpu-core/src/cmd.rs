//! Command submission: rings, doorbells, fences, channels.
//!
//! # The real-world shape we are modeling
//!
//! A GPU is not called like a function; it is fed like a tape drive. The
//! driver writes command packets into a *ring buffer* in memory, then
//! writes the new tail pointer to a *doorbell* register. The GPU's
//! front-end DMAs packets from head to tail, executes them, and advances
//! head. Completion flows back the other way through *fences*: a command
//! that writes a monotonically increasing value to a known location, which
//! the driver polls or receives an interrupt for. Every GPU stack — CUDA
//! streams, Vulkan queues, Metal command buffers — bottoms out in exactly
//! this structure.
//!
//! Getting this shape right matters for virtualization specifically:
//! because submission is asynchronous and mediated by memory, a device
//! model can *trap the doorbell* and schedule between tenants at that
//! boundary. That is the entire trick behind mediated passthrough
//! (NVIDIA vGPU, Intel GVT-g): guests write rings directly, the mediator
//! owns doorbells and time-slices the hardware between them.

use crate::types::{ChannelId, Cycles, GpuVirtAddr, Result, VgpuError};

/// One command packet, the unit of work a guest submits.
///
/// All addresses are **guest virtual**. This is a load-bearing invariant:
/// nothing a guest can place in a ring names physical memory, so the worst
/// a malicious guest can do is fault its own channel. The variants mirror
/// the three engine classes on a real card: copy engines (`MemFill`,
/// `MemCopy`), compute (`KernelLaunch`), and the synchronization
/// micro-ops every engine supports (`FenceSignal`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    /// Set `len` bytes at `dst` to `value` (what cudaMemset lowers to).
    MemFill {
        /// Destination guest VA.
        dst: GpuVirtAddr,
        /// Byte count.
        len: u64,
        /// Fill byte.
        value: u8,
    },
    /// Copy `len` bytes from `src` to `dst` (cudaMemcpyDeviceToDevice).
    /// Overlapping ranges are the guest's own foot-gun, as on real HW.
    MemCopy {
        /// Source guest VA.
        src: GpuVirtAddr,
        /// Destination guest VA.
        dst: GpuVirtAddr,
        /// Byte count.
        len: u64,
    },
    /// Launch a compute kernel: `threads` copies of `program`, each with
    /// `r0 = thread index` and `args` preloaded into `r1..`. Programs are
    /// statically validated at submit (`isa::validate`) and interpreted by
    /// the engine, with every `Ld`/`St` translated through the submitting
    /// vGPU's page tables — kernels have no other path to memory.
    KernelLaunch {
        /// Debug name (what a profiler would show).
        name: String,
        /// Number of threads in the (flat) grid.
        threads: u32,
        /// Kernel arguments, loaded into `r1..` of every thread.
        args: Vec<u64>,
        /// The program, shared by all threads.
        program: Vec<crate::isa::Instr>,
    },
    /// Write `value` to the channel's fence slot when all prior commands
    /// in this ring have completed. Fences are how the guest learns that
    /// work finished; values must be monotonically increasing per channel.
    FenceSignal {
        /// Fence value to publish.
        value: u64,
    },
}

impl Command {
    /// Modeled cost of executing this command, in cycles.
    ///
    /// Costs are deliberately simple, linear models — the scheduler only
    /// needs *relative* magnitudes to demonstrate fair sharing, and a
    /// transparent cost model keeps every fairness test explainable by
    /// hand. (1 cycle per 16 bytes ≈ "copies are bandwidth-bound".)
    ///
    /// For `KernelLaunch` this is the *straight-line estimate* (per-thread
    /// instruction costs × threads): exact for branch-free programs, a
    /// lower bound for loops. The engine reports the real executed cost;
    /// this estimate exists only for pre-execution accounting (ring
    /// pricing, fault charging for commands that never ran).
    pub fn cost(&self) -> Cycles {
        match self {
            Command::MemFill { len, .. } => 1 + len / 16,
            Command::MemCopy { len, .. } => 1 + len / 8, // read + write traffic
            Command::KernelLaunch {
                threads, program, ..
            } => {
                let per_thread: u64 = program.iter().map(crate::isa::Instr::cost).sum();
                (per_thread.saturating_mul(*threads as u64)).max(1)
            }
            Command::FenceSignal { .. } => 1,
        }
    }
}

/// A fixed-capacity single-producer/single-consumer ring buffer.
///
/// # The one-slot-empty convention
///
/// `head == tail` must unambiguously mean "empty", so a ring of capacity N
/// holds at most N-1 commands: pushing the Nth would make `tail` wrap onto
/// `head` and become indistinguishable from empty. Real rings solve this
/// the same way (or with separate wrap counters); we use the classic
/// convention because it needs no extra state and its arithmetic is easy
/// to verify by eye.
///
/// `head`/`tail` are plain integers, not atomics: the foundation milestone
/// is single-threaded by design (see `docs/03-architecture.md` — the
/// concurrency story arrives with the daemon milestone, and bolting atomics
/// on now would suggest cross-thread guarantees the type does not yet keep).
#[derive(Debug)]
pub struct Ring {
    slots: Vec<Option<Command>>,
    /// Next slot the device will consume.
    head: usize,
    /// Next slot the guest will fill.
    tail: usize,
}

impl Ring {
    /// A ring with room for `capacity - 1` in-flight commands.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity >= 2, "ring needs at least one usable slot");
        Self {
            slots: (0..capacity).map(|_| None).collect(),
            head: 0,
            tail: 0,
        }
    }

    /// Guest side: append a command. Fails with `RingFull` instead of
    /// blocking — the *caller* decides whether to spin, sleep, or drop,
    /// because the right policy differs between a latency-sensitive shim
    /// and a batch submitter.
    pub fn push(&mut self, cmd: Command) -> Result<()> {
        let next = (self.tail + 1) % self.slots.len();
        if next == self.head {
            return Err(VgpuError::RingFull);
        }
        self.slots[self.tail] = Some(cmd);
        self.tail = next;
        Ok(())
    }

    /// Device side: consume the oldest command, if any.
    pub fn pop(&mut self) -> Option<Command> {
        if self.head == self.tail {
            return None;
        }
        let cmd = self.slots[self.head]
            .take()
            .expect("occupied slot between head and tail");
        self.head = (self.head + 1) % self.slots.len();
        Some(cmd)
    }

    /// Commands currently queued.
    pub fn len(&self) -> usize {
        // Wrapping subtraction in ring space.
        (self.tail + self.slots.len() - self.head) % self.slots.len()
    }

    /// True if no commands are queued.
    pub fn is_empty(&self) -> bool {
        self.head == self.tail
    }

    /// Iterate the queued commands, oldest first, without consuming them.
    /// Exists for migration: a suspended vGPU's rings are a closed set
    /// (nothing is producing or consuming), so a non-destructive walk is
    /// exactly a snapshot of in-flight work.
    pub fn iter_pending(&self) -> impl Iterator<Item = &Command> + '_ {
        (0..self.len()).map(move |i| {
            self.slots[(self.head + i) % self.slots.len()]
                .as_ref()
                .expect("occupied slot between head and tail")
        })
    }
}

/// Lifecycle of a channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelState {
    /// Accepting and executing work.
    Active,
    /// A command faulted; the channel is dead until torn down. Mirrors
    /// hardware "robust channel" recovery: the faulting context is killed,
    /// *other* channels and other vGPUs are untouched — faults must never
    /// escape their blast radius.
    Faulted,
}

/// A channel's migratable state: everything a destination node needs to
/// reconstruct the channel exactly — the still-queued commands, the fence
/// the guest has observed, and whether the channel was already dead.
///
/// Note what is *absent*: head/tail indices (positions in a ring are an
/// implementation detail; the pending commands in order are the truth),
/// and any physical resource. This is the general shape of migratable
/// state: guest-observable facts only.
#[derive(Debug, Clone, PartialEq)]
pub struct ChannelExport {
    /// Commands submitted but not yet executed, oldest first.
    pub pending: Vec<Command>,
    /// Last fence value signaled to the guest.
    pub completed_fence: u64,
    /// Highest fence value ever *submitted* on this channel (>= the
    /// completed one, since queued fences have not signaled yet).
    /// Carried across migration so the monotonicity rule survives the
    /// move: a channel that has promised fence 9 must keep refusing 9 on
    /// its new node, or a guest could rewind its own completion clock by
    /// migrating.
    pub submitted_fence: u64,
    /// True if the channel had been killed by a fault (a dead channel
    /// migrates as dead — migration must not resurrect it).
    pub faulted: bool,
}

/// A channel: one submission ring plus its fence state.
///
/// Channels belong to exactly one vGPU and inherit its address space —
/// there is deliberately no way to express "this channel, that other
/// vGPU's memory".
#[derive(Debug)]
pub struct Channel {
    /// This channel's ID within its vGPU.
    pub id: ChannelId,
    /// The submission ring.
    pub ring: Ring,
    /// Last fence value the device has signaled (guest-visible completion
    /// state; starts at 0, so guests use values >= 1).
    pub completed_fence: u64,
    /// Highest fence value accepted at submit. Tracked separately from
    /// `completed_fence` because monotonicity must be judged against
    /// what is *queued*, not what has run: two fences sitting in the
    /// ring have not signaled yet, and the second must still be greater
    /// than the first.
    pub submitted_fence: u64,
    /// Active or dead.
    pub state: ChannelState,
    /// The fault that killed the channel, kept for the host-side record.
    pub fault: Option<VgpuError>,
}

impl Channel {
    /// New active channel with a ring of `ring_capacity` slots.
    pub fn new(id: ChannelId, ring_capacity: usize) -> Self {
        Self {
            id,
            ring: Ring::new(ring_capacity),
            completed_fence: 0,
            submitted_fence: 0,
            state: ChannelState::Active,
            fault: None,
        }
    }

    /// Guest-facing submit: refuse work on a dead channel, enforce the
    /// fence clock's monotonicity, else ring push.
    ///
    /// Rejecting a non-increasing fence is not pedantry. Every waiter in
    /// the stack — `vgpu_shim`'s `synchronize`, any guest polling a fence
    /// — reasons "fence >= N implies everything submitted before N has
    /// completed". A guest that signals 5 and then 1 makes the channel's
    /// completion clock run *backwards*, and a waiter blocked on 3 that
    /// had already been satisfied would, after the rewind, see an
    /// unsatisfied fence — or worse, a *later* wait for 3 returns
    /// immediately against the stale 5. Hardware fence/timeline
    /// semaphores are monotonic for exactly this reason; the device is
    /// the only place that can enforce it, so it does.
    pub fn submit(&mut self, cmd: Command) -> Result<()> {
        if self.state == ChannelState::Faulted {
            return Err(VgpuError::ChannelFaulted(self.id));
        }
        let fence = match &cmd {
            Command::FenceSignal { value } if *value <= self.submitted_fence => {
                return Err(VgpuError::FenceRegression {
                    last: self.submitted_fence,
                    attempted: *value,
                })
            }
            Command::FenceSignal { value } => Some(*value),
            _ => None,
        };
        self.ring.push(cmd)?;
        // Advanced only after the push succeeded: a command rejected by a
        // full ring must not move a clock it never joined.
        if let Some(value) = fence {
            self.submitted_fence = value;
        }
        Ok(())
    }

    /// Device-facing: mark the channel dead after a fault.
    pub(crate) fn kill(&mut self, fault: VgpuError) {
        self.state = ChannelState::Faulted;
        self.fault = Some(fault);
        // Drain remaining commands: a faulted channel must not execute
        // work queued after the fault (ordering would be unobservable
        // to the guest and therefore meaningless).
        while self.ring.pop().is_some() {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_fifo_order_and_capacity() {
        let mut r = Ring::new(4); // 3 usable slots
        r.push(Command::FenceSignal { value: 1 }).unwrap();
        r.push(Command::FenceSignal { value: 2 }).unwrap();
        r.push(Command::FenceSignal { value: 3 }).unwrap();
        assert!(matches!(
            r.push(Command::FenceSignal { value: 4 }),
            Err(VgpuError::RingFull)
        ));
        assert_eq!(r.len(), 3);
        assert_eq!(r.pop(), Some(Command::FenceSignal { value: 1 }));
        // Space freed: push succeeds again, and wraps correctly.
        r.push(Command::FenceSignal { value: 4 }).unwrap();
        assert_eq!(r.pop(), Some(Command::FenceSignal { value: 2 }));
        assert_eq!(r.pop(), Some(Command::FenceSignal { value: 3 }));
        assert_eq!(r.pop(), Some(Command::FenceSignal { value: 4 }));
        assert!(r.pop().is_none());
        assert!(r.is_empty());
    }

    #[test]
    fn ring_survives_many_wraps() {
        let mut r = Ring::new(3);
        for i in 0..1000u64 {
            r.push(Command::FenceSignal { value: i }).unwrap();
            assert_eq!(r.pop(), Some(Command::FenceSignal { value: i }));
        }
    }

    #[test]
    fn faulted_channel_rejects_submits_and_drops_queue() {
        let mut ch = Channel::new(ChannelId(0), 8);
        ch.submit(Command::FenceSignal { value: 1 }).unwrap();
        ch.kill(VgpuError::RingFull); // any error value works for the test
        assert!(ch.ring.is_empty(), "queued work is discarded on fault");
        assert!(matches!(
            ch.submit(Command::FenceSignal { value: 2 }),
            Err(VgpuError::ChannelFaulted(_))
        ));
    }

    #[test]
    fn command_costs_scale_with_size() {
        let small = Command::MemCopy {
            src: GpuVirtAddr(0),
            dst: GpuVirtAddr(0),
            len: 8,
        };
        let big = Command::MemCopy {
            src: GpuVirtAddr(0),
            dst: GpuVirtAddr(0),
            len: 8000,
        };
        assert!(big.cost() > small.cost());
    }
}
