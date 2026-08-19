//! The GPU node: one physical GPU, many vGPUs, one mediation loop.
//!
//! `GpuNode` is the piece that corresponds to the *mediator* in mediated
//! passthrough (the `nvidia-vgpu-mgr` process / GVT-g's kernel module):
//! it owns every physical resource (the allocator, the backing store, the
//! scheduler) and exposes only vGPU-scoped operations. All guest-facing
//! entry points take a `VgpuId`, and everything they do is confined to
//! that vGPU's address space and budgets.
//!
//! `tick()` is the heart: the time-slicing loop that multiplexes one
//! physical command front-end across tenants. Everything else is plumbing
//! around admission control and the host-DMA path.

use std::collections::{BTreeMap, HashMap};

use crate::cmd::{ChannelState, Command};
use crate::engine::execute;
use crate::metrics::{Counters, NodeMetrics, TenantMetrics};
use crate::sched::Scheduler;
use crate::types::{
    AccessKind, ChannelId, Cycles, GpuVirtAddr, Result, VgpuError, VgpuId, MAX_DMA_BYTES,
};
use crate::vgpu::{Vgpu, VgpuProfile, VgpuState};
use crate::vram::{FrameStore, VramAllocator};

/// Reject transfers past the device's per-transfer ceiling.
///
/// Checked before any buffer is sized. On the network path the length
/// arrives from a client, and `vec![0; len]` with an unbounded `len` is
/// an abort — of the process serving *every* tenant on the node. The
/// device enforcing its own limit means the daemon can size a buffer
/// only after the core has agreed the number is sane.
pub fn check_transfer_len(len: u64) -> Result<()> {
    if len > MAX_DMA_BYTES {
        return Err(VgpuError::TransferTooLarge {
            requested: len,
            limit: MAX_DMA_BYTES,
        });
    }
    Ok(())
}

/// Static description of the physical GPU this node manages.
#[derive(Debug, Clone)]
pub struct PhysGpuConfig {
    /// Card name, for logs and the fabric inventory.
    pub name: String,
    /// Total VRAM in bytes (frame multiple).
    pub vram_bytes: u64,
    /// Preferred time-slice length in cycles. Shorter slices = lower
    /// latency between tenants, more scheduling overhead; NVIDIA vGPU
    /// exposes exactly this dial (0.5-30 ms). Slices are a *target*:
    /// commands are not preempted mid-execution, so a slice may overrun
    /// by up to one command's cost (the overrun is charged, so vruntime
    /// self-corrects — see `tick`).
    pub slice_cycles: Cycles,
}

/// What one `tick` did — returned to the caller (and, later, shipped to
/// the fabric control plane as telemetry).
#[derive(Debug, Default)]
pub struct TickReport {
    /// Real cycles consumed this tick.
    pub cycles: Cycles,
    /// Commands executed to completion.
    pub commands: u64,
    /// Channels killed by faults this tick.
    pub faults: Vec<FaultRecord>,
    /// Cycles the GPU sat idle *with work queued* because every runnable
    /// tenant had spent its QoS ceiling. This is the visible price of a
    /// hard cap, and it belongs in telemetry rather than hidden: an
    /// operator seeing idle cycles next to queued work should be able to
    /// tell "capped by policy" from "nothing to do".
    pub idle_cycles: Cycles,
}

/// One channel-killing fault.
#[derive(Debug)]
pub struct FaultRecord {
    /// Offending vGPU.
    pub vgpu: VgpuId,
    /// Channel that was killed.
    pub channel: ChannelId,
    /// The fault itself.
    pub error: VgpuError,
}

/// One physical GPU and its tenants.
pub struct GpuNode {
    config: PhysGpuConfig,
    vram: VramAllocator,
    store: FrameStore,
    vgpus: BTreeMap<VgpuId, Vgpu>,
    sched: Scheduler,
    /// Per-vGPU round-robin cursor over its channels, so one busy channel
    /// cannot starve its siblings within the vGPU's own time slice.
    rr_cursor: HashMap<VgpuId, usize>,
    next_id: u32,
    /// VRAM promised to live vGPUs via their profiles. Admission control
    /// compares against capacity so profiles can never oversubscribe —
    /// the foundation models *guaranteed* VRAM, not ballooning.
    committed_vram: u64,
    clock: Cycles,
    /// Per-tenant counters, keyed alongside `vgpus`.
    counters: BTreeMap<VgpuId, Counters>,
    /// Node-wide counters (see `metrics` for why these are counters).
    busy_cycles: Cycles,
    capped_idle_cycles: Cycles,
}

impl GpuNode {
    /// Bring up a node for the given card.
    pub fn new(config: PhysGpuConfig) -> Self {
        let vram = VramAllocator::new(config.vram_bytes);
        Self {
            config,
            vram,
            store: FrameStore::new(),
            vgpus: BTreeMap::new(),
            sched: Scheduler::new(),
            rr_cursor: HashMap::new(),
            next_id: 0,
            committed_vram: 0,
            clock: 0,
            counters: BTreeMap::new(),
            busy_cycles: 0,
            capped_idle_cycles: 0,
        }
    }

    /// Logical time elapsed on this node.
    pub fn clock(&self) -> Cycles {
        self.clock
    }

    /// The card description this node was brought up with.
    pub fn config(&self) -> &PhysGpuConfig {
        &self.config
    }

    /// Real cycles a vGPU has consumed (scheduler account).
    pub fn consumed(&self, id: VgpuId) -> Cycles {
        self.sched.consumed(id)
    }

    /// Is this tenant currently held back by its own QoS ceiling?
    /// Surfaced because "my job is slow" and "my job is slow *because I
    /// bought a quarter card*" are different support tickets, and only
    /// the device can tell them apart.
    pub fn is_capped_out(&self, id: VgpuId) -> bool {
        self.sched.is_capped_out(id)
    }

    /// Cycles this tenant has taken inside the current QoS window.
    pub fn window_consumed(&self, id: VgpuId) -> Cycles {
        self.sched.window_consumed(id)
    }

    /// Adopt a QoS window figure carried from another node (migration).
    ///
    /// A cap is an *absolute* promise to a customer — "never more than
    /// 25%" — so it has to survive a move. This is exactly where it
    /// differs from vruntime, which migration deliberately drops:
    /// vruntime is meaningful only relative to a node's *other* tenants,
    /// so importing it would be nonsense, while a cap means the same
    /// thing on every card in the fleet. Without this, a tenant migrated
    /// once per window would collect its ceiling twice over.
    ///
    /// Only ever raises the figure (see `Scheduler::adopt_window_consumed`).
    pub fn adopt_qos_window(&mut self, id: VgpuId, consumed: Cycles) -> Result<()> {
        self.vgpu(id)?; // exists?
        self.sched.adopt_window_consumed(id, consumed);
        Ok(())
    }

    /// VRAM bytes not yet promised to any profile.
    pub fn uncommitted_vram(&self) -> u64 {
        self.config.vram_bytes - self.committed_vram
    }

    // -- lifecycle ----------------------------------------------------------

    /// Admit a new vGPU under `profile`.
    ///
    /// Admission checks the *profile* budget against *uncommitted*
    /// capacity — not current free frames — because the contract is that
    /// an admitted vGPU can always allocate up to its budget. (Buddy
    /// fragmentation can still fail a specific large allocation; the
    /// budget guarantees frames exist, not that any given contiguity
    /// exists. See `docs/04-walkthrough-memory.md`.)
    pub fn create_vgpu(&mut self, profile: VgpuProfile) -> Result<VgpuId> {
        profile.validate()?;
        if self.committed_vram + profile.vram_bytes > self.config.vram_bytes {
            return Err(VgpuError::ProfileUnsatisfiable {
                why: "insufficient uncommitted VRAM on this node".to_string(),
            });
        }
        // Compute reservations are admission-controlled exactly like VRAM
        // budgets: a floor is a promise, and promises that sum past the
        // card cannot all be kept. Refusing here is the only honest
        // moment — after admission the only options are breaking an SLA
        // or evicting someone.
        let reserved: u32 = self
            .vgpus
            .values()
            .filter_map(|v| v.profile.qos.min_share_pct)
            .sum();
        if let Some(want) = profile.qos.min_share_pct {
            if reserved + want > 100 {
                return Err(VgpuError::ProfileUnsatisfiable {
                    why: format!(
                        "compute reservations would total {}%, over 100%",
                        reserved + want
                    ),
                });
            }
        }
        let id = VgpuId(self.next_id);
        self.next_id += 1; // never reused: stale IDs must not alias new tenants
        let weight = profile.compute_weight;
        let qos = profile.qos;
        self.committed_vram += profile.vram_bytes;
        self.vgpus.insert(id, Vgpu::new(id, profile)?);
        self.sched.register_with_qos(id, weight, qos);
        self.rr_cursor.insert(id, 0);
        self.counters.insert(id, Counters::default());
        Ok(id)
    }

    /// Tear down a vGPU: all VRAM scrubbed and returned, scheduler account
    /// closed, profile commitment released.
    pub fn destroy_vgpu(&mut self, id: VgpuId) -> Result<()> {
        let vgpu = self.vgpus.get_mut(&id).ok_or(VgpuError::NoSuchVgpu(id))?;
        vgpu.destroy(&mut self.vram, &mut self.store);
        self.committed_vram -= vgpu.profile.vram_bytes;
        self.sched.unregister(id);
        self.rr_cursor.remove(&id);
        self.counters.remove(&id);
        self.vgpus.remove(&id);
        Ok(())
    }

    fn vgpu_mut(&mut self, id: VgpuId) -> Result<&mut Vgpu> {
        self.vgpus.get_mut(&id).ok_or(VgpuError::NoSuchVgpu(id))
    }

    fn vgpu(&self, id: VgpuId) -> Result<&Vgpu> {
        self.vgpus.get(&id).ok_or(VgpuError::NoSuchVgpu(id))
    }

    /// Start a created vGPU.
    pub fn start_vgpu(&mut self, id: VgpuId) -> Result<()> {
        self.vgpu_mut(id)?.start()
    }

    /// Suspend a running vGPU (stops scheduling and submissions).
    pub fn suspend_vgpu(&mut self, id: VgpuId) -> Result<()> {
        self.vgpu_mut(id)?.suspend()
    }

    /// Resume a suspended vGPU.
    pub fn resume_vgpu(&mut self, id: VgpuId) -> Result<()> {
        self.vgpu_mut(id)?.resume()
    }

    /// Lifecycle state of a vGPU.
    pub fn vgpu_state(&self, id: VgpuId) -> Result<VgpuState> {
        Ok(self.vgpu(id)?.state())
    }

    // -- migration primitives ------------------------------------------------
    //
    // Deliberately small verbs rather than one "snapshot blob" operation:
    // a migrator composes them (see `vgpu_proto::migrate`), and each verb
    // reuses an existing, tested mechanism — allocation replay uses
    // `alloc_memory`, page transfer uses `dma_read`/`dma_write`, and only
    // channels need a dedicated export/import pair.

    /// The profile a vGPU was admitted under (a migrator re-admits the
    /// twin under the identical contract).
    pub fn vgpu_profile(&self, id: VgpuId) -> Result<VgpuProfile> {
        Ok(self.vgpu(id)?.profile.clone())
    }

    /// Live allocations as `(base VA, bytes)`, in creation order — replay
    /// them on a fresh vGPU to reproduce identical guest VAs.
    pub fn list_allocations(&self, id: VgpuId) -> Result<Vec<(GpuVirtAddr, u64)>> {
        Ok(self.vgpu(id)?.list_allocations())
    }

    /// Harvest and clear the dirty-page set for `id` (page-aligned guest
    /// VAs written since the previous call; a fresh vGPU reports every
    /// mapped page). The pre-copy loop's read-and-reset primitive.
    pub fn take_dirty(&mut self, id: VgpuId) -> Result<Vec<GpuVirtAddr>> {
        Ok(self.vgpu_mut(id)?.take_dirty())
    }

    /// Export channel state (requires Suspended).
    pub fn export_channels(&self, id: VgpuId) -> Result<Vec<crate::cmd::ChannelExport>> {
        self.vgpu(id)?.export_channels()
    }

    /// Import channel state into a fresh, not-yet-started vGPU.
    pub fn import_channels(
        &mut self,
        id: VgpuId,
        exports: Vec<crate::cmd::ChannelExport>,
    ) -> Result<()> {
        self.vgpu_mut(id)?.import_channels(exports)
    }

    // -- guest-facing operations (each confined to one vGPU) ----------------

    /// Allocate device memory for `id`; returns a guest VA.
    pub fn alloc_memory(&mut self, id: VgpuId, bytes: u64) -> Result<GpuVirtAddr> {
        let Self { vgpus, vram, .. } = self;
        vgpus
            .get_mut(&id)
            .ok_or(VgpuError::NoSuchVgpu(id))?
            .alloc_memory(vram, bytes)
    }

    /// Allocate device memory at a specific guest VA (migration replay).
    pub fn alloc_memory_at(
        &mut self,
        id: VgpuId,
        base: GpuVirtAddr,
        bytes: u64,
    ) -> Result<GpuVirtAddr> {
        let Self { vgpus, vram, .. } = self;
        vgpus
            .get_mut(&id)
            .ok_or(VgpuError::NoSuchVgpu(id))?
            .alloc_memory_at(vram, base, bytes)
    }

    /// Free a device allocation by its base VA.
    pub fn free_memory(&mut self, id: VgpuId, base: GpuVirtAddr) -> Result<()> {
        let Self {
            vgpus, vram, store, ..
        } = self;
        vgpus
            .get_mut(&id)
            .ok_or(VgpuError::NoSuchVgpu(id))?
            .free_memory(vram, store, base)
    }

    /// Create a command channel on `id`.
    pub fn create_channel(&mut self, id: VgpuId) -> Result<ChannelId> {
        self.vgpu_mut(id)?.create_channel()
    }

    /// Submit one command to `id`'s channel `ch` (the doorbell write).
    pub fn submit(&mut self, id: VgpuId, ch: ChannelId, cmd: Command) -> Result<()> {
        self.vgpu_mut(id)?.submit(ch, cmd)
    }

    /// Last signaled fence value on a channel.
    pub fn fence_value(&self, id: VgpuId, ch: ChannelId) -> Result<u64> {
        self.vgpu(id)?.fence_value(ch)
    }

    /// Host→device DMA: write host bytes into a vGPU's memory at `dst`.
    ///
    /// This is the model's `cudaMemcpyHostToDevice`. It goes through the
    /// vGPU's page tables like everything else — the host path gets no
    /// physical back door, which is exactly how an IOMMU-protected DMA
    /// engine behaves.
    pub fn dma_write(&mut self, id: VgpuId, dst: GpuVirtAddr, data: &[u8]) -> Result<()> {
        check_transfer_len(data.len() as u64)?;
        let Self { vgpus, store, .. } = self;
        let vgpu = vgpus.get_mut(&id).ok_or(VgpuError::NoSuchVgpu(id))?;
        let segs = vgpu
            .aspace
            .translate_range(dst, data.len() as u64, AccessKind::Write)?;
        let mut cursor = 0usize;
        for (pa, len) in segs {
            store.write(pa, &data[cursor..cursor + len as usize]);
            cursor += len as usize;
        }
        vgpu.aspace.mark_dirty_range(dst, data.len() as u64); // migration substrate
        if let Some(c) = self.counters.get_mut(&id) {
            c.bytes_dma_in += data.len() as u64;
        }
        Ok(())
    }

    /// Device→host DMA: read a vGPU's memory at `src` into a host buffer.
    pub fn dma_read(&mut self, id: VgpuId, src: GpuVirtAddr, buf: &mut [u8]) -> Result<()> {
        check_transfer_len(buf.len() as u64)?;
        let vgpu = self.vgpus.get(&id).ok_or(VgpuError::NoSuchVgpu(id))?;
        let segs = vgpu
            .aspace
            .translate_range(src, buf.len() as u64, AccessKind::Read)?;
        let mut cursor = 0usize;
        for (pa, len) in segs {
            self.store.read(pa, &mut buf[cursor..cursor + len as usize]);
            cursor += len as usize;
        }
        if let Some(c) = self.counters.get_mut(&id) {
            c.bytes_dma_out += buf.len() as u64;
        }
        Ok(())
    }

    // -- the mediation loop --------------------------------------------------

    /// Run the GPU for up to `budget` cycles, time-slicing across runnable
    /// vGPUs. Returns what happened.
    ///
    /// Loop shape, per iteration:
    /// 1. refresh runnability (rings may have drained last slice),
    /// 2. `pick()` the minimum-vruntime runnable vGPU,
    /// 3. run its commands — round-robin across its channels — until the
    ///    slice target is met or it runs dry,
    /// 4. charge the *actual* cycles used (including any overrun from a
    ///    non-preemptible final command) so vruntime self-corrects: a
    ///    tenant that overran gets picked correspondingly later next time.
    ///
    /// Faults kill the offending channel and are reported; they consume
    /// the faulting command's cost (the hardware analogue: a faulted
    /// context still occupied the engine until the fault was recognized).
    pub fn tick(&mut self, budget: Cycles) -> TickReport {
        let mut report = TickReport::default();

        while report.cycles < budget {
            // (1) Runnability can change every slice; recompute honestly.
            for (id, vgpu) in &self.vgpus {
                self.sched.set_runnable(*id, vgpu.has_pending_work());
            }
            // (2) Whom does fairness owe the next slice?
            let Some(id) = self.sched.pick() else {
                if self.sched.all_runnable_are_capped() {
                    // Work is queued, but every tenant holding it has
                    // spent its ceiling. The GPU idles — that is what a
                    // hard cap *is* — and the window must still advance,
                    // or the cap would stall time and then refill itself.
                    let idle = budget - report.cycles;
                    self.sched.advance_idle(idle);
                    self.clock += idle;
                    report.idle_cycles += idle;
                }
                break;
            };

            // (3) Drain up to one slice from this vGPU.
            let slice_target = self.config.slice_cycles.min(budget - report.cycles);
            let mut slice_used: Cycles = 0;
            while slice_used < slice_target {
                let Some((ch, cmd)) = self.pop_round_robin(id) else {
                    break;
                };
                let vgpu = self.vgpus.get_mut(&id).expect("picked ids exist");
                let outcome = execute(&cmd, &mut vgpu.aspace, &mut self.store);
                // Charge engine occupancy regardless of outcome — a
                // faulting command still held the engine until the fault
                // was recognized (kernels: the instructions actually
                // executed before faulting).
                slice_used += outcome.cycles.max(1);
                if let Command::KernelLaunch { .. } = cmd {
                    if let Some(c) = self.counters.get_mut(&id) {
                        c.kernel_launches += 1;
                    }
                }
                match outcome.result {
                    Ok(()) => {
                        if let Command::FenceSignal { value } = cmd {
                            // Fences complete in submission order because
                            // this loop is the only executor and it is
                            // strictly in-order per channel.
                            vgpu.channels[ch.0 as usize].completed_fence = value;
                        }
                        report.commands += 1;
                        if let Some(c) = self.counters.get_mut(&id) {
                            c.commands_completed += 1;
                        }
                    }
                    Err(error) => {
                        vgpu.channels[ch.0 as usize].kill(error.clone());
                        if let Some(c) = self.counters.get_mut(&id) {
                            c.faults += 1;
                        }
                        report.faults.push(FaultRecord {
                            vgpu: id,
                            channel: ch,
                            error,
                        });
                    }
                }
            }

            if slice_used == 0 {
                // Defensive: a vGPU picked with no poppable work (should
                // be unreachable given the runnability refresh) must not
                // spin the loop forever.
                self.sched.set_runnable(id, false);
                continue;
            }
            // (4) Charge reality, not the target.
            self.sched.charge(id, slice_used);
            report.cycles += slice_used;
        }

        self.clock += report.cycles;
        self.busy_cycles += report.cycles;
        self.capped_idle_cycles += report.idle_cycles;
        report
    }

    /// Collect a full telemetry snapshot: the card, plus every tenant.
    ///
    /// Reads state the node already maintains — nothing here is sampled
    /// on the hot path, which is deliberate. Metrics that cost something
    /// to collect get collected rarely and are stale exactly when they
    /// matter; metrics that cost nothing get scraped every few seconds
    /// and are there when someone is paging.
    pub fn metrics(&self) -> NodeMetrics {
        let tenants = self
            .vgpus
            .iter()
            .map(|(id, v)| {
                let c = self.counters.get(id).cloned().unwrap_or_default();
                TenantMetrics {
                    vgpu: *id,
                    profile_name: v.profile.name.clone(),
                    state: v.state(),
                    cycles_consumed: self.sched.consumed(*id),
                    commands_completed: c.commands_completed,
                    faults: c.faults,
                    bytes_dma_in: c.bytes_dma_in,
                    bytes_dma_out: c.bytes_dma_out,
                    kernel_launches: c.kernel_launches,
                    vram_used: v.vram_used(),
                    vram_budget: v.profile.vram_bytes,
                    queued_commands: v.queued_commands(),
                    channels: v.channel_count(),
                    window_consumed: self.sched.window_consumed(*id),
                    capped_out: self.sched.is_capped_out(*id),
                }
            })
            .collect::<Vec<_>>();
        NodeMetrics {
            name: self.config.name.clone(),
            clock: self.clock,
            busy_cycles: self.busy_cycles,
            capped_idle_cycles: self.capped_idle_cycles,
            vram_bytes: self.config.vram_bytes,
            uncommitted_vram: self.uncommitted_vram(),
            faults: tenants.iter().map(|t| t.faults).sum(),
            tenants,
        }
    }

    /// Pop the next command from `id`'s channels, rotating the cursor so
    /// channels within a vGPU share its slice round-robin. Skips faulted
    /// channels (they are dead, and `kill` already drained them).
    fn pop_round_robin(&mut self, id: VgpuId) -> Option<(ChannelId, Command)> {
        let vgpu = self.vgpus.get_mut(&id)?;
        let n = vgpu.channels.len();
        if n == 0 {
            return None;
        }
        let cursor = self.rr_cursor.entry(id).or_insert(0);
        for i in 0..n {
            let idx = (*cursor + i) % n;
            let channel = &mut vgpu.channels[idx];
            if channel.state != ChannelState::Active {
                continue;
            }
            if let Some(cmd) = channel.ring.pop() {
                // Advance past the channel we just served.
                *cursor = (idx + 1) % n;
                return Some((ChannelId(idx as u32), cmd));
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sched::QosLimits;
    use crate::types::FRAME_SIZE;

    fn small_node() -> GpuNode {
        GpuNode::new(PhysGpuConfig {
            name: "sim-64f".to_string(),
            vram_bytes: 64 * FRAME_SIZE,
            slice_cycles: 100,
        })
    }

    fn profile(weight: u32, frames: u64) -> VgpuProfile {
        VgpuProfile {
            name: "t".to_string(),
            vram_bytes: frames * FRAME_SIZE,
            compute_weight: weight,
            max_channels: 4,
            ring_slots: 64,
            qos: QosLimits::default(),
        }
    }

    #[test]
    fn admission_control_refuses_oversubscription() {
        let mut node = small_node();
        node.create_vgpu(profile(1, 40)).unwrap();
        let err = node.create_vgpu(profile(1, 40)).unwrap_err();
        assert!(matches!(err, VgpuError::ProfileUnsatisfiable { .. }));
        // 24 frames remain uncommitted; a 24-frame profile fits.
        node.create_vgpu(profile(1, 24)).unwrap();
    }

    #[test]
    fn destroy_releases_commitment() {
        let mut node = small_node();
        let a = node.create_vgpu(profile(1, 64)).unwrap();
        assert_eq!(node.uncommitted_vram(), 0);
        node.destroy_vgpu(a).unwrap();
        assert_eq!(node.uncommitted_vram(), 64 * FRAME_SIZE);
    }

    #[test]
    fn tick_with_no_work_is_a_clean_noop() {
        let mut node = small_node();
        let a = node.create_vgpu(profile(1, 8)).unwrap();
        node.start_vgpu(a).unwrap();
        let r = node.tick(10_000);
        assert_eq!(r.cycles, 0);
        assert_eq!(r.commands, 0);
        assert_eq!(node.clock(), 0);
    }

    #[test]
    fn fences_signal_in_order() {
        let mut node = small_node();
        let a = node.create_vgpu(profile(1, 8)).unwrap();
        node.start_vgpu(a).unwrap();
        let ch = node.create_channel(a).unwrap();
        let buf = node.alloc_memory(a, FRAME_SIZE).unwrap();
        node.submit(
            a,
            ch,
            Command::MemFill {
                dst: buf,
                len: 256,
                value: 1,
            },
        )
        .unwrap();
        node.submit(a, ch, Command::FenceSignal { value: 1 })
            .unwrap();
        node.submit(
            a,
            ch,
            Command::KernelLaunch {
                name: "k".to_string(),
                threads: 1,
                args: vec![],
                program: crate::isa::busy(50),
            },
        )
        .unwrap();
        node.submit(a, ch, Command::FenceSignal { value: 2 })
            .unwrap();

        assert_eq!(node.fence_value(a, ch).unwrap(), 0);
        node.tick(1_000_000);
        assert_eq!(node.fence_value(a, ch).unwrap(), 2);
    }
}
