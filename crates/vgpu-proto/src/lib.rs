//! # vgpu-proto — the VGPU-X wire protocol
//!
//! Milestone 1 puts the milestone-0 device model behind a socket. This
//! crate is the *contract* between the two sides:
//!
//! * [`wire`] — byte-level primitives: little-endian scalar codecs, a
//!   decoding cursor that is total (any input → value or `WireError`,
//!   never a panic), and `[u32 length][body]` framing with an allocation
//!   guard against lying length prefixes.
//! * [`msg`] — the vocabulary: [`msg::Request`] mirrors `GpuNode`'s
//!   public surface 1:1, [`msg::Response`] carries results including the
//!   complete `VgpuError` taxonomy with every field intact.
//! * [`client`] — [`client::VgpuClient`], the blocking typed client the
//!   milestone-2 guest shim will be built on.
//!
//! Design rules (spelled out in `docs/06-walkthrough-daemon.md`):
//! version byte first, tags are append-only, decoders must consume input
//! exactly (`Dec::finish`), and the daemon never trusts a length it read
//! from a socket.

pub mod client;
pub mod msg;
pub mod wire;

pub use client::{ClientError, ClientResult, VgpuClient};
pub use msg::{FaultSummary, NodeInfo, Request, Response, TickSummary};
pub use wire::{read_frame, write_frame, WireError, MAX_FRAME_LEN, VERSION};
