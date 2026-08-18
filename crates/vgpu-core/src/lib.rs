//! # vgpu-core — the VGPU-X device model
//!
//! This crate is the foundation layer of VGPU-X: a *device model* for a
//! virtualized GPU, implemented as a deterministic userspace simulation.
//! It answers, in executable form, the question every GPU virtualization
//! stack must answer:
//!
//! > Given one physical GPU and N mutually untrusting tenants, how do you
//! > share **memory** and **compute** such that no tenant can read
//! > another's data or starve another's work?
//!
//! The answer is layered, one module per mechanism:
//!
//! | Module    | Mechanism | Real-world analogue |
//! |-----------|-----------|---------------------|
//! | [`vram`]  | Buddy allocator + scrubbed backing store | `drm_buddy`, VRAM scrubbing in vGPU managers |
//! | [`gmmu`]  | Per-tenant page tables, VA→PA walk | NVIDIA GMMU / AMD GPUVM |
//! | [`cmd`]   | Rings, doorbells, fences, channels | Every command front-end since the 90s |
//! | [`vgpu`]  | Profiles, budgets, lifecycle | NVIDIA vGPU types (`A100-2-10C`…) |
//! | [`sched`] | Weighted virtual-runtime fair scheduler | Linux CFS, vGPU best-effort scheduler |
//! | [`engine`]| Translate-then-touch command execution | Copy/compute engines behind the MMU |
//! | [`node`]  | The mediator that owns everything physical | `nvidia-vgpu-mgr`, Intel GVT-g |
//!
//! Two invariants hold everywhere, and the docs walk through why they are
//! sufficient for isolation:
//!
//! 1. **Guests speak only guest virtual addresses.** No API accepts or
//!    returns a physical address across the guest boundary.
//! 2. **Every byte touched is translated first** through the submitting
//!    vGPU's page tables, and translation failures kill only the faulting
//!    channel.
//!
//! Start with `docs/01-the-landscape.md` in the repository for the
//! conceptual grounding, then `docs/04-walkthrough-memory.md` and
//! `docs/05-walkthrough-execution.md` for line-by-line tours of this code.
//!
//! ## A complete session
//!
//! ```
//! use vgpu_core::prelude::*;
//!
//! // A simulated 16 MiB card that prefers 1000-cycle time slices.
//! let mut node = GpuNode::new(PhysGpuConfig {
//!     name: "sim-a".into(),
//!     vram_bytes: 256 * FRAME_SIZE,
//!     slice_cycles: 1000,
//! });
//!
//! // Admit a tenant: 2 MiB of VRAM, weight 1, up to 2 channels.
//! let tenant = node.create_vgpu(VgpuProfile {
//!     name: "sim-2m.1x".into(),
//!     vram_bytes: 32 * FRAME_SIZE,
//!     compute_weight: 1,
//!     max_channels: 2,
//!     ring_slots: 64,
//!     ..Default::default()   // no QoS caps: pure proportional share
//! }).unwrap();
//! node.start_vgpu(tenant).unwrap();
//!
//! // Guest workflow: allocate, upload, compute, fence, download.
//! let buf = node.alloc_memory(tenant, 4096).unwrap();
//! node.dma_write(tenant, buf, b"hello, device").unwrap();
//!
//! let ch = node.create_channel(tenant).unwrap();
//! node.submit(tenant, ch, Command::KernelLaunch {
//!     name: "noop".into(),
//!     threads: 1,
//!     args: vec![],
//!     program: vgpu_core::isa::busy(500), // a real program now — see `isa`
//! }).unwrap();
//! node.submit(tenant, ch, Command::FenceSignal { value: 1 }).unwrap();
//!
//! node.tick(10_000);                                   // give the GPU time
//! assert_eq!(node.fence_value(tenant, ch).unwrap(), 1); // work completed
//!
//! let mut out = [0u8; 13];
//! node.dma_read(tenant, buf, &mut out).unwrap();
//! assert_eq!(&out, b"hello, device");
//! ```

pub mod cmd;
pub mod engine;
pub mod gmmu;
pub mod isa;
pub mod metrics;
pub mod node;
pub mod sched;
pub mod types;
pub mod vgpu;
pub mod vram;

/// One-line import for the common surface.
pub mod prelude {
    pub use crate::cmd::{ChannelState, Command};
    pub use crate::metrics::{NodeMetrics, TenantMetrics};
    pub use crate::node::{GpuNode, PhysGpuConfig, TickReport};
    pub use crate::sched::QosLimits;
    pub use crate::types::{
        AccessKind, ChannelId, Cycles, GpuVirtAddr, VgpuError, VgpuId, FRAME_SIZE,
    };
    pub use crate::vgpu::{VgpuProfile, VgpuState};
}
