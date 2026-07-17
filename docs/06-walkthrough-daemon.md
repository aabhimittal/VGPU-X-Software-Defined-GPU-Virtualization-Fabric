# 6. Walkthrough: The Daemon and the Wire (Milestone 1)

Milestone 0 built a device model you could only call as a library.
Milestone 1 makes it a *service*: `vgpud` owns a `GpuNode` and serves it
over TCP to many concurrent clients. This chapter reads `vgpu-proto` and
`vgpud` the same way chapters 4–5 read the core — every design decision
with its why — because the two hard problems here (a trustworthy wire
format, a concurrency story that doesn't rot) are exactly the ones that
sink real mediator daemons.

## 6.1 What changed in the core, and why: `&'static str` → `String`

Milestone 0's `Command::KernelLaunch { name: &'static str }` was honest
for a library: kernel names were compile-time constants. A wire protocol
breaks that assumption — a name decoded from a socket lives on the heap
with a runtime lifetime, and there is no sound way to conjure `&'static`
from received bytes (short of leaking memory per message). So names and
error strings became `String` across the core, one commit ahead of the
protocol that forced it.

The general lesson is worth stating: **`'static` in a public type is a
claim that the data never crosses a serialization boundary.** Make that
claim only where you mean it.

## 6.2 `wire.rs`: owning the bytes

### Why hand-rolled codecs

Same rationale as the zero-dependency core: the wire format *is*
curriculum. ioctls, virtio rings, and GPU command packets are all
hand-specified byte layouts; the failure modes worth internalizing —
truncation, bad tags, version skew, hostile lengths — only become visible
when you own the bytes. Two properties are non-negotiable and testable:

* **Totality.** `Dec` methods return `Result`, never panic, for *any*
  input. The daemon feeds these functions bytes from untrusted sockets;
  a decoder that can panic is a remote crash primitive.
* **Exact consumption.** `Dec::finish()` errors on trailing bytes.
  Encoder/decoder skew (peer added a field you don't read) must fail
  loudly at the frame, not corrupt the *next* frame's parse.

### The frame

```
[len: u32 LE][body]      body = [VERSION u8][tag u8][fields…]
```

Three deliberate choices:

* **`MAX_FRAME_LEN` before allocation.** The first rule of reading a
  length prefix from a socket: distrust it. A peer claiming a 4 GiB frame
  gets an error, not a 4 GiB `Vec`. (Inside a frame the same holds
  structurally: `Dec::bytes` can never allocate more than the frame it
  is decoding.)
* **Version byte first.** Version skew fails with `VersionMismatch{ours,
  theirs}` at decode, before any field is interpreted. Tags are
  append-only — renumbering is a silent-corruption factory.
* **Clean-EOF vs torn-frame distinction.** `read_frame` returns
  `Ok(None)` only when the peer closes *between* frames; EOF mid-frame is
  an error. Conflating them turns every crash into a "graceful" hang-up
  and hides real bugs.

## 6.3 `msg.rs`: the vocabulary is the security argument

`Request` mirrors `GpuNode`'s public API 1:1 — the daemon adds **no**
operations of its own. That makes the milestone-0 isolation argument
transfer across the network boundary by inspection: if every network
operation is exactly a `GpuNode` operation, and every `GpuNode` operation
is confined to one vGPU's address space and budget, then so is every
network operation. The dispatch function in `vgpud` (`handle`) is a flat
match where every arm is one method call; its brevity is the point.

The other promise kept here: milestone 0 argued for one flat `VgpuError`
enum because "the fabric's control plane must serialize these across a
wire, and a single closed set keeps that mapping honest." The wire has
now arrived. `enc_error`/`dec_error` cover every variant with every field
— a client gets the same typed `PageFault { addr, access }` a local
caller would — and `error_roundtrips_losslessly` pins it. Adding a
variant without a codec arm fails the build (match exhaustiveness), which
is exactly the honesty the flat enum bought.

## 6.4 `vgpud`: the single-owner device thread

The central decision. `GpuNode` is `!Sync` on purpose; the daemon had two
ways to share it:

* `Mutex<GpuNode>` — every connection thread locks, calls, unlocks.
* **One thread owns the node; connections send it jobs over a channel.**

Both serialize (a GPU has one command front-end; mediation *is*
serialization). The difference is where the serialization lives. With a
mutex it is implicit and scattered — every handler is a potential
lock-hold-across-I/O bug, and contention shows up as convoy latency that
profiles as "mutex" rather than as any particular cause. With the
ownership thread, the serialization point is a *place you can read*:

```rust
struct Job { request: Request, reply: mpsc::Sender<Response> }

fn device_loop(config: DaemonConfig, jobs: mpsc::Receiver<Job>, stop: …) {
    let mut node = GpuNode::new(config.gpu);   // sole owner, forever
    loop { /* recv job → handle(&mut node, req) → reply */ }
}
```

The analogy to the hardware is exact and worth savoring: in mediated
passthrough, doorbell writes from every guest funnel through one mediator
context; here, `submit` calls from every connection funnel through one
mpsc channel, whose FIFO order plays the role of PCIe write ordering.
The architecture *is* the domain model.

Two supporting choices:

* **Per-request reply channels.** Each job carries its own
  `mpsc::Sender<Response>` (a rendezvous). Responses cannot be
  cross-delivered between connections even in principle — there is no
  shared reply path to mis-route on.
* **Thread-per-connection.** A node serves tens of tenants, not tens of
  thousands; blocking reads keep each connection's loop linear and
  obvious. Async would buy capacity this layer will never need and cost
  the readability that is this repo's whole point.

### The shutdown lesson (a real deadlock, caught by the test suite)

The first version of `device_loop` in manual-tick mode blocked on
`jobs.recv()` and exited on channel *disconnect* — "when all senders are
gone, we're done." It deadlocked the whole test suite. Why: every idle
connection thread holds a live sender clone while blocked in
`read_frame(socket)`, so `shutdown()` joining the device thread waited
on senders that would only drop when clients hung up — which the tests,
reasonably, did after `shutdown()`. Waiting on disconnect meant waiting
on *clients* to cooperate with *our* shutdown.

The fix is the standard one: the blocking wait becomes `recv_timeout`,
and a shared stop flag is checked on every timeout. General form worth
remembering: **never make shutdown depend on the goodwill of peers you
don't control.** The bug and fix are preserved in `device_loop`'s doc
comment because it is the most instructive twenty lines in the crate.

### Where the wall clock enters

The core remains a deterministic simulation. Wall-clock time enters the
system in exactly one expression: the auto-tick, where "no requests for
`interval`" converts into `node.tick(budget)`. Tests set
`auto_tick: None` and drive time with explicit `Tick` requests — which is
why the daemon integration tests assert exact fence values and cycle
counts just like the unit tests underneath. One test
(`auto_tick_drives_the_gpu`) opts into real time and correspondingly
asserts only *progress*, never counts. Determinism is spent narrowly and
on purpose, at the edge.

## 6.5 `client.rs`: deliberately dumb

`VgpuClient` is strict request→response alternation on one socket: no
pipelining, no background reader, one in-flight call. Real drivers
pipeline aggressively — and milestone 2's shim may need to — but the
frames are already self-delimiting, so pipelining is a client upgrade,
not a protocol change. Correctness-checkable-by-eye first.

The error type keeps three failure layers apart because the right
reaction differs per layer:

| Layer | Variant | Right reaction |
|---|---|---|
| Transport | `Io` | retry / reconnect |
| Protocol | `Protocol`, `UnexpectedResponse` | bug report; never normal |
| Device | `Device(VgpuError)` | ordinary error handling, same as local |

## 6.6 What the integration tests prove

`crates/vgpud/tests/daemon.rs`, each a claim:

| Claim | Test |
|---|---|
| The full guest workflow works over real TCP, with exact results | `end_to_end_session_over_tcp` |
| Device errors cross the wire as typed values, fields intact | `device_errors_survive_the_wire` |
| Isolation holds with tenants racing on concurrent connections | `concurrent_tenants_stay_isolated` |
| The daemon makes progress with no client driving it | `auto_tick_drives_the_gpu` |
| Abrupt client death never harms the daemon or other tenants | `daemon_survives_client_disconnects` |

Next milestone (M2): the guest shim — API interception that turns
`cudaMalloc`-shaped calls into this client's methods, plus a tiny kernel
interpreter so launches finally touch guest memory through the same GMMU.
