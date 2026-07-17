//! Weighted fair scheduling of vGPUs over one physical GPU.
//!
//! # The problem
//!
//! N tenants, one command front-end. Each tenant's profile carries a
//! weight, and the contract is *proportional share*: over any window where
//! a set of tenants all have work queued, tenant i receives
//! `weight_i / Σ weights` of the GPU's cycles. Round-robin cannot express
//! weights; strict priority starves; lottery scheduling is only fair in
//! expectation. The clean deterministic answer — the same one Linux CFS
//! uses for CPUs and NVIDIA's vGPU "best effort with weights" scheduler
//! approximates in firmware — is **virtual runtime**.
//!
//! # Virtual runtime in one paragraph
//!
//! Charge each tenant a clock that runs *inversely to its weight*: after
//! consuming `c` real cycles, tenant i's virtual clock advances by
//! `c / weight_i`. Always run the tenant with the smallest virtual clock.
//! A heavy tenant's clock ticks slowly, so it gets picked more often; two
//! tenants with equal vruntime have, by definition, received cycles in
//! exact proportion to their weights. Fairness falls out of the invariant
//! "the scheduler equalizes vruntime" — there is no ratio bookkeeping
//! anywhere, which is why the technique generalizes so well.
//!
//! # The sleeper problem
//!
//! A tenant idle for a million cycles keeps an ancient (tiny) vruntime; if
//! it wakes, a naive argmin hands it the GPU exclusively until it "catches
//! up" — historical credit converted into a future monopoly. CFS's fix,
//! which we copy, is: fairness applies only to the *runnable*. On wake, a
//! tenant's vruntime is clamped up to the scheduler's high-water mark
//! (`min_vruntime`), so it competes from *now* — see `set_runnable`.

use std::collections::HashMap;

use crate::types::{Cycles, VgpuId};

/// Fixed-point scale for vruntime so integer division by weight keeps
/// precision (CFS does the same with `NICE_0_LOAD` = 1024). vruntime is
/// u128: worst case is `cycles * SCALE` with weight 1, and 2^64 cycles
/// * 2^10 needs headroom beyond u64.
const SCALE: u128 = 1024;

#[derive(Debug)]
struct Account {
    weight: u32,
    /// Weighted virtual runtime, in `cycles * SCALE / weight` units.
    vruntime: u128,
    /// Unweighted total, for metrics and fairness assertions in tests.
    consumed: Cycles,
    runnable: bool,
}

/// The per-node scheduler. It knows nothing about commands or memory —
/// only identities, weights, runnability, and charges. That narrowness is
/// what will let the same scheduler arbitrate *placement* across nodes in
/// the fabric milestone.
pub struct Scheduler {
    accounts: HashMap<VgpuId, Account>,
    /// High-water mark: the largest vruntime any tenant has been *picked
    /// at*. Monotonic. Used only to clamp wakers (see module docs).
    min_vruntime: u128,
}

impl Scheduler {
    /// Empty scheduler.
    pub fn new() -> Self {
        Self {
            accounts: HashMap::new(),
            min_vruntime: 0,
        }
    }

    /// Register a tenant. New tenants start at the high-water mark, not at
    /// zero — joining the node must not grant a retroactive cycle debt
    /// against incumbents (same clamp as waking, for the same reason).
    pub fn register(&mut self, id: VgpuId, weight: u32) {
        debug_assert!(weight > 0, "profile validation guarantees weight > 0");
        self.accounts.insert(
            id,
            Account {
                weight,
                vruntime: self.min_vruntime,
                consumed: 0,
                runnable: false,
            },
        );
    }

    /// Remove a tenant (on destroy).
    pub fn unregister(&mut self, id: VgpuId) {
        self.accounts.remove(&id);
    }

    /// Update runnability. The false→true edge applies the sleeper clamp;
    /// the true→false edge is just bookkeeping. Idempotent per state.
    pub fn set_runnable(&mut self, id: VgpuId, runnable: bool) {
        if let Some(acc) = self.accounts.get_mut(&id) {
            if runnable && !acc.runnable {
                acc.vruntime = acc.vruntime.max(self.min_vruntime);
            }
            acc.runnable = runnable;
        }
    }

    /// Pick the runnable tenant with minimum vruntime, advancing the
    /// high-water mark to its vruntime.
    ///
    /// Ties break by `VgpuId` — arbitrary but *stable*, so a replayed
    /// trace schedules identically. Determinism is a feature the whole
    /// test suite (and, later, migration debugging) stands on.
    pub fn pick(&mut self) -> Option<VgpuId> {
        let chosen = self
            .accounts
            .iter()
            .filter(|(_, a)| a.runnable)
            .min_by_key(|(id, a)| (a.vruntime, **id))
            .map(|(id, _)| *id)?;
        let v = self.accounts[&chosen].vruntime;
        self.min_vruntime = self.min_vruntime.max(v);
        Some(chosen)
    }

    /// Charge `cycles` of real GPU time to a tenant.
    pub fn charge(&mut self, id: VgpuId, cycles: Cycles) {
        if let Some(acc) = self.accounts.get_mut(&id) {
            acc.vruntime += cycles as u128 * SCALE / acc.weight as u128;
            acc.consumed += cycles;
        }
    }

    /// Total real cycles this tenant has consumed (metrics hook).
    pub fn consumed(&self, id: VgpuId) -> Cycles {
        self.accounts.get(&id).map_or(0, |a| a.consumed)
    }
}

impl Default for Scheduler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drive the scheduler with unit slices and return per-tenant cycles.
    fn run(sched: &mut Scheduler, slices: u64, slice_cycles: Cycles) {
        for _ in 0..slices {
            let id = sched.pick().expect("someone is runnable");
            sched.charge(id, slice_cycles);
        }
    }

    #[test]
    fn equal_weights_split_evenly() {
        let (a, b) = (VgpuId(0), VgpuId(1));
        let mut s = Scheduler::new();
        s.register(a, 1);
        s.register(b, 1);
        s.set_runnable(a, true);
        s.set_runnable(b, true);
        run(&mut s, 1000, 10);
        assert_eq!(s.consumed(a), 5000);
        assert_eq!(s.consumed(b), 5000);
    }

    #[test]
    fn weights_yield_proportional_share() {
        let (a, b) = (VgpuId(0), VgpuId(1));
        let mut s = Scheduler::new();
        s.register(a, 3);
        s.register(b, 1);
        s.set_runnable(a, true);
        s.set_runnable(b, true);
        run(&mut s, 4000, 10);
        // 3:1 exactly, because vruntime equalization is exact with equal
        // slice sizes: 30000:10000.
        assert_eq!(s.consumed(a), 30_000);
        assert_eq!(s.consumed(b), 10_000);
    }

    #[test]
    fn sleeper_does_not_monopolize_on_wake() {
        let (a, b) = (VgpuId(0), VgpuId(1));
        let mut s = Scheduler::new();
        s.register(a, 1);
        s.register(b, 1);
        s.set_runnable(a, true);
        s.set_runnable(b, false); // b sleeps while a burns 100k cycles
        run(&mut s, 100, 1000);
        assert_eq!(s.consumed(a), 100_000);

        s.set_runnable(b, true); // b wakes with an ancient vruntime...
        run(&mut s, 100, 1000);
        // ...but the clamp means it only competes from now: it gets half
        // of the post-wake window, not 100% of it until "caught up".
        let b_share = s.consumed(b);
        assert!(
            (49_000..=51_000).contains(&b_share),
            "b should get ~half the post-wake window, got {b_share}"
        );
    }

    #[test]
    fn late_joiner_starts_at_high_water_mark() {
        let (a, b) = (VgpuId(0), VgpuId(1));
        let mut s = Scheduler::new();
        s.register(a, 1);
        s.set_runnable(a, true);
        run(&mut s, 100, 1000);

        s.register(b, 1); // joins after a has 100k cycles of history
        s.set_runnable(b, true);
        run(&mut s, 100, 1000);
        let b_share = s.consumed(b);
        assert!(
            (49_000..=51_000).contains(&b_share),
            "late joiner gets ~half going forward, got {b_share}"
        );
    }

    #[test]
    fn pick_is_deterministic_under_ties() {
        let mut s = Scheduler::new();
        s.register(VgpuId(2), 1);
        s.register(VgpuId(0), 1);
        s.register(VgpuId(1), 1);
        for id in [0, 1, 2] {
            s.set_runnable(VgpuId(id), true);
        }
        // All vruntimes equal -> lowest id wins, every time.
        assert_eq!(s.pick(), Some(VgpuId(0)));
        assert_eq!(s.pick(), Some(VgpuId(0)));
    }
}
