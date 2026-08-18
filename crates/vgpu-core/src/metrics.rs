//! Per-tenant and per-node telemetry.
//!
//! Everything until now made the fabric *correct*. This module makes it
//! *operable*, which is a different property and is usually bolted on
//! far too late. The questions an operator has at 3am are not the
//! questions a designer has at design time:
//!
//! * "Which tenant is eating this card?" — `cycles_consumed`.
//! * "Is this tenant slow, or is it capped?" — `capped_out`,
//!   `window_consumed`. Those look identical from inside the guest and
//!   have opposite remedies, so only the device can distinguish them.
//! * "Is that card full, or just busy?" — `vram_used` vs `vram_budget`
//!   per tenant, `uncommitted_vram` per node. Memory pressure and
//!   compute pressure need different responses (migrate vs. rebalance).
//! * "Is anything faulting?" — `faults`, and *which* channel died.
//! * "Should I rebalance?" — `busy_cycles` against `idle_cycles`, node
//!   by node.
//!
//! # Counters, not gauges, wherever possible
//!
//! Every field here except the obvious levels (`vram_used`, `queued`,
//! `state`) is a monotonic counter, never a rate. Counters are the right
//! shape for telemetry because they are *idempotent under scraping*: a
//! collector that misses a sample, samples twice, or restarts loses
//! nothing, since rates are derived by differencing at query time. This
//! is why Prometheus counters won and why gauges of "current ops/sec"
//! keep lying. The one thing that must never be a counter is a window
//! measurement, so `window_consumed` is explicitly documented as a level
//! that resets.
//!
//! # Cost
//!
//! Collection is O(tenants), allocates a small `Vec`, and reads state
//! the node already maintains — nothing here is sampled on the hot path.
//! Metrics that cost something to collect get collected rarely and are
//! stale when it matters; metrics that cost nothing get collected every
//! few seconds and are there when you need them.

use crate::types::{ChannelId, Cycles, VgpuId};
use crate::vgpu::VgpuState;

/// Everything the node knows about one tenant, at one instant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TenantMetrics {
    /// Which tenant.
    pub vgpu: VgpuId,
    /// The profile it was admitted under.
    pub profile_name: String,
    /// Lifecycle state — a suspended tenant consuming nothing is healthy;
    /// a running one consuming nothing is a mystery worth chasing.
    pub state: VgpuState,
    /// Total GPU cycles ever charged to this tenant (counter).
    pub cycles_consumed: Cycles,
    /// Commands run to completion (counter).
    pub commands_completed: u64,
    /// Channel-killing faults (counter). Non-zero means a guest bug or a
    /// hostile guest; either way somebody wants to know.
    pub faults: u64,
    /// Bytes moved host→device and device→host by DMA (counters).
    pub bytes_dma_in: u64,
    /// Bytes read back to the host.
    pub bytes_dma_out: u64,
    /// Kernel launches (counter).
    pub kernel_launches: u64,
    /// Bytes currently allocated, and the profile's ceiling. A tenant at
    /// 99% of budget is one allocation away from failing.
    pub vram_used: u64,
    /// The tenant's hard VRAM budget.
    pub vram_budget: u64,
    /// Commands sitting in rings right now (a level, not a counter).
    /// Sustained depth means the tenant is demand-limited by the GPU;
    /// zero means it is limited by itself.
    pub queued_commands: u64,
    /// Live channels.
    pub channels: u32,
    /// Cycles taken inside the current QoS window (a level that resets
    /// each window — see the module docs on counters vs levels).
    pub window_consumed: Cycles,
    /// True when the tenant is being held back by its own QoS ceiling
    /// right now. The single most valuable field here: it converts "the
    /// GPU is slow" into "you bought a quarter card", which is a support
    /// answer rather than a support investigation.
    pub capped_out: bool,
}

/// One node's telemetry: the card, plus every tenant on it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeMetrics {
    /// Card name.
    pub name: String,
    /// Logical time elapsed on this node.
    pub clock: Cycles,
    /// Cycles spent executing tenant work (counter).
    pub busy_cycles: Cycles,
    /// Cycles the GPU sat idle *with work queued*, because every runnable
    /// tenant had spent its QoS ceiling (counter). Distinguished from
    /// plain idleness on purpose: this number is capacity an operator
    /// could sell by raising a cap, whereas an idle-because-nobody-asked
    /// card is capacity nobody wants.
    pub capped_idle_cycles: Cycles,
    /// Total VRAM on the card.
    pub vram_bytes: u64,
    /// VRAM not committed to any profile.
    pub uncommitted_vram: u64,
    /// Faults across all tenants (counter).
    pub faults: u64,
    /// Per-tenant detail, ordered by `VgpuId` for stable diffing.
    pub tenants: Vec<TenantMetrics>,
}

impl NodeMetrics {
    /// Fraction of elapsed logical time spent executing work, in percent.
    /// A node that has never run reports 0 rather than dividing by its
    /// zero clock — `checked_div` states that in one expression.
    pub fn utilization_pct(&self) -> u64 {
        (self.busy_cycles * 100)
            .checked_div(self.clock)
            .unwrap_or(0)
    }
}

/// Running per-tenant counters, owned by the node and updated where the
/// events actually happen.
///
/// Kept as a separate struct rather than fields on `Vgpu` so that the
/// accounting has one home: a counter updated from three places grows a
/// fourth place that forgets, and the resulting numbers are worse than
/// no numbers because people trust them.
#[derive(Debug, Default, Clone)]
pub(crate) struct Counters {
    pub commands_completed: u64,
    pub faults: u64,
    pub bytes_dma_in: u64,
    pub bytes_dma_out: u64,
    pub kernel_launches: u64,
}

/// A fault, as telemetry: which channel died and what killed it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FaultSummaryMetric {
    /// The channel that was killed.
    pub channel: ChannelId,
    /// Human-readable cause.
    pub cause: String,
}
