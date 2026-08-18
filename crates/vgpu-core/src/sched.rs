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
//! # Weights alone cannot express what operators sell
//!
//! Proportional share answers "who gets the GPU *now*", and that is the
//! only question a fair scheduler asks. Two questions operators ask
//! constantly are not expressible in weights at all:
//!
//! * **"This tenant must never exceed 25%, even on an idle GPU."** A
//!   weight cannot say this: weights only bind under contention, so a
//!   weight-1 tenant alone on the card gets 100%. But a customer who
//!   bought a quarter card and sees full-card performance at 3am will
//!   file a bug when their neighbours arrive and it halves — and worse,
//!   *the vendor cannot reproduce it*, because performance now depends
//!   on who else is running. A hard cap trades throughput for the thing
//!   tenants actually want from a tier: **predictability**. NVIDIA ships
//!   exactly this as its "fixed share" scheduler, opposite "best effort".
//! * **"This tenant is guaranteed 30% under any contention."** Weights
//!   give a *ratio*, and a ratio's floor collapses as tenants arrive: a
//!   weight-3 tenant holds 75% against one weight-1 neighbour and 23%
//!   against nine. An SLA is an absolute floor, so it needs one.
//!
//! Both are enforced over a sliding window of `QOS_WINDOW_CYCLES`, which
//! is what makes them checkable at all — an instantaneous "share" is not
//! a measurable quantity. Caps and reservations are opt-in per profile;
//! a profile that sets neither behaves exactly as it did before they
//! existed, which is why every fairness test from milestone 0 still
//! passes unchanged.
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

/// Length of the accounting window for caps and reservations.
///
/// A "share" is only meaningful over an interval, so this constant is
/// the interval. The trade is the usual one for any windowed limiter:
/// shorter windows enforce tightly but let a tenant with bursty demand
/// lose cycles it could have used; longer windows tolerate bursts but
/// let a capped tenant monopolize the early part of a window. 100k
/// cycles is ~1000 default slices — long enough that scheduling noise
/// averages out, short enough that a cap is felt promptly.
pub const QOS_WINDOW_CYCLES: Cycles = 100_000;

/// Fixed-point scale for vruntime so integer division by weight keeps
/// precision (CFS does the same with `NICE_0_LOAD` = 1024). vruntime is
/// u128: worst case is `cycles * SCALE` with weight 1, and 2^64 cycles
/// * 2^10 needs headroom beyond u64.
const SCALE: u128 = 1024;

/// Optional per-tenant quality-of-service limits, both expressed as a
/// percentage of the GPU over one `QOS_WINDOW_CYCLES` window.
///
/// `None` for both means "pure proportional share" — the milestone-0
/// behaviour, and still the default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct QosLimits {
    /// Hard ceiling: the tenant is skipped once it has taken this share
    /// of the current window, *even if the GPU would otherwise idle*.
    /// Buys predictability at the cost of throughput.
    pub max_share_pct: Option<u32>,
    /// Guaranteed floor under contention: while the tenant is below this
    /// share of the window it is scheduled ahead of tenants that are not.
    /// Admission control refuses profiles whose floors would sum past
    /// 100% — a guarantee the node cannot keep must never be sold.
    pub min_share_pct: Option<u32>,
}

#[derive(Debug)]
struct Account {
    weight: u32,
    /// Weighted virtual runtime, in `cycles * SCALE / weight` units.
    vruntime: u128,
    /// Unweighted total, for metrics and fairness assertions in tests.
    consumed: Cycles,
    runnable: bool,
    qos: QosLimits,
    /// Cycles taken inside the current QoS window.
    window_consumed: Cycles,
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
    /// Cycles accounted in the current QoS window, including cycles the
    /// GPU spent *idle* because every runnable tenant was capped out.
    /// Counting forced idleness is what makes a cap a cap: if the window
    /// only advanced when work ran, a capped-out tenant alone on the card
    /// would stall the window and then be handed a fresh budget with no
    /// time having passed — a limiter that limits nothing.
    window_used: Cycles,
}

impl Scheduler {
    /// Empty scheduler.
    pub fn new() -> Self {
        Self {
            accounts: HashMap::new(),
            min_vruntime: 0,
            window_used: 0,
        }
    }

    /// Register a tenant. New tenants start at the high-water mark, not at
    /// zero — joining the node must not grant a retroactive cycle debt
    /// against incumbents (same clamp as waking, for the same reason).
    pub fn register(&mut self, id: VgpuId, weight: u32) {
        self.register_with_qos(id, weight, QosLimits::default());
    }

    /// Register a tenant carrying QoS limits.
    pub fn register_with_qos(&mut self, id: VgpuId, weight: u32, qos: QosLimits) {
        debug_assert!(weight > 0, "profile validation guarantees weight > 0");
        self.accounts.insert(
            id,
            Account {
                weight,
                vruntime: self.min_vruntime,
                consumed: 0,
                runnable: false,
                qos,
                window_consumed: 0,
            },
        );
    }

    /// Share of one window, in cycles, for a percentage.
    fn window_share(pct: u32) -> Cycles {
        QOS_WINDOW_CYCLES * pct as u64 / 100
    }

    /// Has this tenant spent its ceiling for the current window?
    fn capped_out(acc: &Account) -> bool {
        acc.qos
            .max_share_pct
            .is_some_and(|pct| acc.window_consumed >= Self::window_share(pct))
    }

    /// Is this tenant still below its guaranteed floor?
    fn under_floor(acc: &Account) -> bool {
        acc.qos
            .min_share_pct
            .is_some_and(|pct| acc.window_consumed < Self::window_share(pct))
    }

    /// True when work is queued but every runnable tenant has spent its
    /// cap. The caller (the tick loop) must then let the GPU idle and
    /// tell the scheduler how long, via `advance_idle` — cap-induced
    /// idleness is the visible price of predictability.
    pub fn all_runnable_are_capped(&self) -> bool {
        let mut any_runnable = false;
        for acc in self.accounts.values() {
            if acc.runnable {
                any_runnable = true;
                if !Self::capped_out(acc) {
                    return false;
                }
            }
        }
        any_runnable
    }

    /// Account `cycles` of GPU time that no tenant consumed, so a window
    /// full of capped-out tenants still rolls over.
    pub fn advance_idle(&mut self, cycles: Cycles) {
        self.window_used += cycles;
        self.roll_window_if_elapsed();
    }

    fn roll_window_if_elapsed(&mut self) {
        if self.window_used >= QOS_WINDOW_CYCLES {
            self.window_used = 0;
            for acc in self.accounts.values_mut() {
                acc.window_consumed = 0;
            }
        }
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
    /// Tenants over their cap are not candidates at all. Among those that
    /// remain, any tenant still below its guaranteed floor is served
    /// first — the reservation tier — and proportional share decides
    /// within each tier. Two tiers is enough: a floor is a promise that
    /// outranks fairness, and everything above its floor is back to
    /// competing normally.
    pub fn pick(&mut self) -> Option<VgpuId> {
        let eligible = || {
            self.accounts
                .iter()
                .filter(|(_, a)| a.runnable && !Self::capped_out(a))
        };
        let chosen = eligible()
            .filter(|(_, a)| Self::under_floor(a))
            .min_by_key(|(id, a)| (a.vruntime, **id))
            .or_else(|| eligible().min_by_key(|(id, a)| (a.vruntime, **id)))
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
            acc.window_consumed += cycles;
        }
        self.window_used += cycles;
        self.roll_window_if_elapsed();
    }

    /// Cycles this tenant has taken inside the current QoS window
    /// (telemetry: how close it is to its ceiling).
    pub fn window_consumed(&self, id: VgpuId) -> Cycles {
        self.accounts.get(&id).map_or(0, |a| a.window_consumed)
    }

    /// Is this tenant currently held back by its own ceiling? Exposed
    /// because "my job is slow" and "my job is slow *because I bought a
    /// quarter card*" are different support tickets.
    pub fn is_capped_out(&self, id: VgpuId) -> bool {
        self.accounts.get(&id).is_some_and(Self::capped_out)
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
