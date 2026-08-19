//! # vgpu-fabric — the control plane (Milestone 4)
//!
//! The last layer: a fleet of `vgpud` nodes becomes *one pool of GPU
//! capacity*. The fabric answers exactly three questions, and its API is
//! those three questions and their bookkeeping:
//!
//! 1. **Where should this tenant run?** — [`Fabric::place`]: best-fit
//!    bin-packing over profile VRAM budgets.
//! 2. **What is running where?** — [`Fabric::inventory`]: live node
//!    reports plus the tenant registry.
//! 3. **How do I move things?** — [`Fabric::migrate_tenant`] and
//!    [`Fabric::evacuate`] (drain a node for maintenance), both built on
//!    milestone 3's live migration.
//!
//! # Control plane vs data plane
//!
//! The fabric *places* tenants; it does not proxy their work. A guest
//! gets back a [`TenantHandle`] — node address + vGPU id — and talks to
//! that node directly (through `VgpuClient` or the milestone-2 shim).
//! This split is why every serious fleet system looks the same
//! (Kubernetes doesn't proxy your pod's packets; a GPU fabric must not
//! proxy doorbell writes): the control plane is on the slow path where
//! policy lives, and adding a tenant never adds load to placement.
//!
//! # Why placement is tractable at all
//!
//! Milestone 0 chose *fixed profiles* over per-resource dials, promising
//! it would "make placement decidable for a fabric scheduler: it can
//! pack profiles onto cards like Tetris pieces". This crate is where
//! that promise is kept: because a tenant's VRAM demand is a single
//! known number (the profile budget, guaranteed by node admission
//! control), placement is classic bin-packing, and a one-line best-fit
//! rule gives a good, *deterministic* answer. Had tenants carried
//! elastic demands, this file would be a capacity estimator, a load
//! predictor, and a regret minimizer. Constraints chosen early are the
//! reason later layers stay small.
//!
//! # State ownership, honestly
//!
//! The registry (which tenant is on which node) lives in the fabric; the
//! *truth* (what a node is actually running) lives on the nodes. The
//! fabric keeps them consistent by being the only actor that places,
//! migrates, or destroys — the single-writer discipline, the same move
//! the daemon made with its device thread. What this milestone does NOT
//! implement, on purpose: fabric HA/consensus (a second fabric instance
//! would break single-writer), node-failure detection, and reconciling
//! external mutations behind the fabric's back. Those are real problems
//! with real literature (Raft, cell architectures); noting the boundary
//! is the honest move.

use std::collections::BTreeMap;
use std::fmt;
use std::net::SocketAddr;

use vgpu_core::metrics::{NodeMetrics, TenantMetrics};
use vgpu_core::types::{VgpuError, VgpuId};
use vgpu_core::vgpu::VgpuProfile;
use vgpu_proto::{migrate, ClientError, MigrateError, MigrateOptions, VgpuClient};

/// Identifies a node within this fabric.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(pub u32);

/// Fabric-global tenant identity — *stable across migrations*, which is
/// the point: `(NodeId, VgpuId)` changes when a tenant moves; the
/// `TenantId` a client holds does not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TenantId(pub u32);

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "node{}", self.0)
    }
}

impl fmt::Display for TenantId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "tenant{}", self.0)
    }
}

/// Where a tenant's work actually goes: the data-plane coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TenantHandle {
    /// Fabric-stable identity.
    pub tenant: TenantId,
    /// Node currently hosting the vGPU.
    pub node: NodeId,
    /// Address guests connect to for submissions and DMA.
    pub addr: SocketAddr,
    /// The vGPU id on that node.
    pub vgpu: VgpuId,
}

/// One node's live standing plus the fabric's view of it.
#[derive(Debug, Clone)]
pub struct NodeReport {
    /// Fabric id.
    pub node: NodeId,
    /// Card name as the node reports it.
    pub name: String,
    /// Data-plane address.
    pub addr: SocketAddr,
    /// Total VRAM on the card.
    pub vram_bytes: u64,
    /// VRAM not committed to any profile (live, from the node).
    pub uncommitted_vram: u64,
    /// Tenants the fabric has placed here.
    pub tenants: Vec<TenantId>,
}

/// One node's telemetry, joined to fabric-stable tenant identity.
#[derive(Debug, Clone)]
pub struct NodeTelemetry {
    /// Fabric id of the node.
    pub node: NodeId,
    /// Its data-plane address.
    pub addr: SocketAddr,
    /// Everything the node reported about itself.
    pub metrics: NodeMetrics,
    /// Per-tenant metrics paired with the fabric's stable `TenantId`.
    /// `None` means the node is running a vGPU the fabric did not place —
    /// worth surfacing rather than hiding, since unmanaged tenants
    /// consume capacity the fabric is busy promising to someone else.
    pub tenants: Vec<(Option<TenantId>, TenantMetrics)>,
}

/// Control-plane failures.
#[derive(Debug)]
pub enum FabricError {
    /// RPC to a node failed.
    Client(ClientError),
    /// A migration failed (source left intact; registry unchanged).
    Migrate(MigrateError),
    /// No node has enough uncommitted VRAM for this profile.
    NoCapacity {
        /// Bytes the profile requires.
        requested: u64,
        /// The largest uncommitted VRAM any node currently offers.
        best_available: u64,
    },
    /// Unknown node id.
    NoSuchNode(NodeId),
    /// Unknown tenant id.
    NoSuchTenant(TenantId),
    /// Evacuation/migration had no eligible destination node.
    NoDestination,
}

impl fmt::Display for FabricError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Client(e) => write!(f, "node rpc failed: {e}"),
            Self::Migrate(e) => write!(f, "migration failed: {e}"),
            Self::NoCapacity {
                requested,
                best_available,
            } => write!(
                f,
                "no node can host {requested} B (best available: {best_available} B)"
            ),
            Self::NoSuchNode(n) => write!(f, "no such node: {n}"),
            Self::NoSuchTenant(t) => write!(f, "no such tenant: {t}"),
            Self::NoDestination => write!(f, "no eligible destination node"),
        }
    }
}

impl std::error::Error for FabricError {}

impl From<ClientError> for FabricError {
    fn from(e: ClientError) -> Self {
        Self::Client(e)
    }
}

impl From<MigrateError> for FabricError {
    fn from(e: MigrateError) -> Self {
        Self::Migrate(e)
    }
}

/// Fabric result alias.
pub type FabricResult<T> = Result<T, FabricError>;

struct Node {
    addr: SocketAddr,
}

struct Placement {
    node: NodeId,
    vgpu: VgpuId,
    profile: VgpuProfile,
}

/// The control plane. Owns the registry; opens short-lived connections
/// to nodes per operation.
///
/// Per-operation connections rather than held ones is a deliberate
/// control-plane idiom: placement and migration are rare, so connection
/// cost is noise, and the fabric never pins a node's connection slot or
/// holds a stream across a long migration — the data plane's steady
/// traffic belongs to guests.
pub struct Fabric {
    nodes: BTreeMap<NodeId, Node>,
    tenants: BTreeMap<TenantId, Placement>,
    next_node: u32,
    next_tenant: u32,
    migrate_opts: MigrateOptions,
}

impl Fabric {
    /// An empty fabric.
    pub fn new() -> Self {
        Self {
            nodes: BTreeMap::new(),
            tenants: BTreeMap::new(),
            next_node: 0,
            next_tenant: 0,
            migrate_opts: MigrateOptions::default(),
        }
    }

    /// Register a node by address. Verifies it is reachable and speaking
    /// our protocol (a `node_info` roundtrip) before admitting it to the
    /// pool — a fabric must never *discover* mid-placement that a node
    /// was never real.
    ///
    /// **Idempotent by address.** Registering the same node twice returns
    /// the original id rather than minting a second one. Two ids for one
    /// card would make the fabric believe it has twice the VRAM it has,
    /// and capacity that does not exist is worse than no capacity: every
    /// placement decision downstream is computed against a fiction, and
    /// the lie only surfaces as a mysterious admission failure at the
    /// node. Registration is exactly the kind of operation that gets
    /// retried by an operator or a config reload, so it must be safe to
    /// repeat.
    pub fn add_node(&mut self, addr: SocketAddr) -> FabricResult<NodeId> {
        if let Some((id, _)) = self.nodes.iter().find(|(_, n)| n.addr == addr) {
            return Ok(*id);
        }
        self.connect(addr)?.node_info()?;
        let id = NodeId(self.next_node);
        self.next_node += 1;
        self.nodes.insert(id, Node { addr });
        Ok(id)
    }

    /// Live inventory: per-node standing (fresh `node_info`, not cached —
    /// capacity questions deserve current answers) plus the registry.
    pub fn inventory(&mut self) -> FabricResult<Vec<NodeReport>> {
        let mut reports = Vec::with_capacity(self.nodes.len());
        for (&id, node) in &self.nodes {
            let info = self.connect(node.addr)?.node_info()?;
            reports.push(NodeReport {
                node: id,
                name: info.name,
                addr: node.addr,
                vram_bytes: info.vram_bytes,
                uncommitted_vram: info.uncommitted_vram,
                tenants: self
                    .tenants
                    .iter()
                    .filter(|(_, p)| p.node == id)
                    .map(|(t, _)| *t)
                    .collect(),
            });
        }
        Ok(reports)
    }

    /// Fleet-wide telemetry: every node's metrics, with the fabric's
    /// tenant ids attached so a number can be traced back to a customer.
    ///
    /// A node knows it is running `vgpu3`; only the fabric knows `vgpu3`
    /// is `tenant7`, who has migrated twice this week. Joining the two is
    /// the whole reason a control plane collects telemetry rather than
    /// leaving operators to scrape nodes: **the identity an alert needs
    /// is the stable one**, and nodes do not have it.
    pub fn telemetry(&mut self) -> FabricResult<Vec<NodeTelemetry>> {
        let nodes: Vec<(NodeId, SocketAddr)> =
            self.nodes.iter().map(|(id, n)| (*id, n.addr)).collect();
        let mut out = Vec::with_capacity(nodes.len());
        for (id, addr) in nodes {
            let metrics = self.connect(addr)?.metrics()?;
            let tenants = metrics
                .tenants
                .iter()
                .map(|t| {
                    let tenant = self
                        .tenants
                        .iter()
                        .find(|(_, p)| p.node == id && p.vgpu == t.vgpu)
                        .map(|(tid, _)| *tid);
                    (tenant, t.clone())
                })
                .collect();
            out.push(NodeTelemetry {
                node: id,
                addr,
                metrics,
                tenants,
            });
        }
        Ok(out)
    }

    /// The node an operator should look at first: highest utilization
    /// among nodes actually hosting tenants. Returns `None` for an empty
    /// or entirely idle fleet.
    ///
    /// Deliberately advisory rather than automatic. A fabric that
    /// rebalances on its own reading of "hot" will chase transients and
    /// migrate tenants during their busiest minute — the classic
    /// autoscaler failure. Surfacing the candidate and letting a human
    /// (or a policy with hysteresis) decide is the conservative default;
    /// `migrate_tenant` is right there when the decision is made.
    pub fn hottest_node(&mut self) -> FabricResult<Option<(NodeId, u64)>> {
        Ok(self
            .telemetry()?
            .into_iter()
            .filter(|t| !t.tenants.is_empty())
            .map(|t| (t.node, t.metrics.utilization_pct()))
            .max_by_key(|(id, util)| (*util, std::cmp::Reverse(*id))))
    }

    /// Place a tenant: choose a node by **best fit**, admit, start.
    ///
    /// Best fit = the node whose uncommitted VRAM exceeds the request by
    /// the *least* (ties broken by `NodeId` — determinism, as everywhere).
    /// Why best-fit and not first-fit or most-free ("worst fit")? Packing
    /// tightly preserves the largest contiguous capacities for the large
    /// profiles that only few nodes can host: spreading a small tenant
    /// onto the emptiest node is exactly how a fleet ends up with 40%
    /// free VRAM and nowhere to put one big tenant. (Best-fit is not
    /// optimal — bin packing is NP-hard — but it is the classic
    /// good-and-explainable answer, and profiles make even the greedy
    /// rule effective.)
    /// Placement re-tries down the ranking when a node refuses the twin.
    /// The fabric reads capacity and *then* admits, and between those two
    /// steps the node is free to change: another operator, a leftover
    /// client, a node restarted with a smaller card. Treating the first
    /// candidate's refusal as fleet-wide exhaustion would strand a tenant
    /// while capacity sat one node over. Nodes that reject are skipped,
    /// and only genuine exhaustion is reported.
    pub fn place(&mut self, profile: VgpuProfile) -> FabricResult<TenantHandle> {
        let mut refused: Vec<NodeId> = Vec::new();
        let (node, addr, vgpu) = loop {
            let node = self
                .best_fit(profile.vram_bytes, None, &refused)
                .map_err(|e| self.exhausted(e, &refused, profile.vram_bytes))?;
            let addr = self.nodes[&node].addr;
            let mut client = self.connect(addr)?;
            match client.create_vgpu(profile.clone()) {
                Ok(vgpu) => {
                    client.start_vgpu(vgpu)?;
                    break (node, addr, vgpu);
                }
                // The node knows its own capacity better than our reading
                // of it did; believe the node and try the next candidate.
                Err(ClientError::Device(VgpuError::ProfileUnsatisfiable { .. }))
                | Err(ClientError::Device(VgpuError::OutOfVram { .. })) => {
                    refused.push(node);
                }
                Err(e) => return Err(FabricError::Client(e)),
            }
        };

        let tenant = TenantId(self.next_tenant);
        self.next_tenant += 1;
        self.tenants.insert(
            tenant,
            Placement {
                node,
                vgpu,
                profile,
            },
        );
        Ok(TenantHandle {
            tenant,
            node,
            addr,
            vgpu,
        })
    }

    /// The data-plane coordinates for a tenant (current node + vGPU id).
    pub fn handle(&self, tenant: TenantId) -> FabricResult<TenantHandle> {
        let p = self
            .tenants
            .get(&tenant)
            .ok_or(FabricError::NoSuchTenant(tenant))?;
        Ok(TenantHandle {
            tenant,
            node: p.node,
            addr: self.nodes[&p.node].addr,
            vgpu: p.vgpu,
        })
    }

    /// Tear a tenant down and release its slot.
    pub fn destroy(&mut self, tenant: TenantId) -> FabricResult<()> {
        let p = self
            .tenants
            .get(&tenant)
            .ok_or(FabricError::NoSuchTenant(tenant))?;
        let addr = self.nodes[&p.node].addr;
        let vgpu = p.vgpu;
        self.connect(addr)?.destroy_vgpu(vgpu)?;
        self.tenants.remove(&tenant);
        Ok(())
    }

    /// Live-migrate one tenant to a specific node. The registry updates
    /// only after the migration succeeds — on failure the source is
    /// intact (milestone 3's contract) and the fabric's view unchanged.
    pub fn migrate_tenant(&mut self, tenant: TenantId, to: NodeId) -> FabricResult<TenantHandle> {
        if !self.nodes.contains_key(&to) {
            return Err(FabricError::NoSuchNode(to));
        }
        let p = self
            .tenants
            .get(&tenant)
            .ok_or(FabricError::NoSuchTenant(tenant))?;
        let (src_addr, src_vgpu) = (self.nodes[&p.node].addr, p.vgpu);
        let dst_addr = self.nodes[&to].addr;

        let mut src = self.connect(src_addr)?;
        let mut dst = self.connect(dst_addr)?;
        let twin = migrate(&mut src, &mut dst, src_vgpu, &self.migrate_opts)?;

        let p = self.tenants.get_mut(&tenant).expect("checked above");
        p.node = to;
        p.vgpu = twin;
        Ok(TenantHandle {
            tenant,
            node: to,
            addr: dst_addr,
            vgpu: twin,
        })
    }

    /// Drain a node: live-migrate every tenant it hosts to the best-fit
    /// *other* node. Returns how many tenants moved. Tenants move one at
    /// a time with the registry updated after each, so a mid-drain
    /// failure leaves a consistent (partially drained) fabric, never a
    /// lost tenant.
    pub fn evacuate(&mut self, node: NodeId) -> FabricResult<u32> {
        if !self.nodes.contains_key(&node) {
            return Err(FabricError::NoSuchNode(node));
        }
        let residents: Vec<TenantId> = self
            .tenants
            .iter()
            .filter(|(_, p)| p.node == node)
            .map(|(t, _)| *t)
            .collect();
        let mut moved = 0;
        for tenant in residents {
            let bytes = self.tenants[&tenant].profile.vram_bytes;
            let dest = self
                .best_fit(bytes, Some(node), &[])
                .map_err(|_| FabricError::NoDestination)?;
            self.migrate_tenant(tenant, dest)?;
            moved += 1;
        }
        Ok(moved)
    }

    /// Keep the most informative error when every candidate is gone: if
    /// nodes refused us, that is the real story, not "no capacity".
    fn exhausted(&self, e: FabricError, refused: &[NodeId], bytes: u64) -> FabricError {
        if refused.is_empty() {
            e
        } else {
            FabricError::NoCapacity {
                requested: bytes,
                best_available: 0,
            }
        }
    }

    /// Best-fit selection over live capacity, optionally excluding one
    /// node (the one being drained) plus any that already refused.
    fn best_fit(
        &mut self,
        bytes: u64,
        exclude: Option<NodeId>,
        refused: &[NodeId],
    ) -> FabricResult<NodeId> {
        let mut best: Option<(u64, NodeId)> = None; // (slack, node)
        let mut best_available = 0u64;
        let candidates: Vec<(NodeId, SocketAddr)> = self
            .nodes
            .iter()
            .filter(|(id, _)| Some(**id) != exclude && !refused.contains(id))
            .map(|(id, n)| (*id, n.addr))
            .collect();
        for (id, addr) in candidates {
            let free = self.connect(addr)?.node_info()?.uncommitted_vram;
            best_available = best_available.max(free);
            if free >= bytes {
                let slack = free - bytes;
                // Strict < keeps the BTreeMap iteration order (ascending
                // NodeId) as the tie-break: deterministic placement.
                if best.is_none_or(|(s, _)| slack < s) {
                    best = Some((slack, id));
                }
            }
        }
        best.map(|(_, id)| id).ok_or(FabricError::NoCapacity {
            requested: bytes,
            best_available,
        })
    }

    fn connect(&self, addr: SocketAddr) -> FabricResult<VgpuClient> {
        VgpuClient::connect(addr).map_err(|e| FabricError::Client(ClientError::Io(e)))
    }
}

impl Default for Fabric {
    fn default() -> Self {
        Self::new()
    }
}

/// Re-exported so fabric users don't need vgpu-core directly for the
/// common case.
pub use vgpu_core::vgpu::VgpuProfile as Profile;
