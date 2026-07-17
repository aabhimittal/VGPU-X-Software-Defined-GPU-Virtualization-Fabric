# 2. How a GPU Actually Executes Work

The mental model most programmers carry — "the CPU calls the GPU like a
function" — is wrong in every particular, and GPU virtualization is
impossible to understand until it is replaced. This chapter builds the
real model. Every structure described here exists in `vgpu-core`, usually
under the same name.

## 2.1 A GPU is fed, not called

There is no "call the GPU" instruction. The only things a CPU can do to a
PCIe device are: read/write its registers (MMIO), read/write memory the
device can also see, and receive its interrupts. Everything else is
convention layered on those three primitives.

The convention every GPU uses is the **ring buffer**:

```
            producer (driver)                consumer (GPU front-end)
                  │                                    │
                  ▼                                    ▼
   ring:  [ cmd ][ cmd ][ cmd ][ cmd ][      ][      ]
                                       ▲               ▲
                                     tail            head
```

The driver writes command packets into a circular buffer in memory,
advances its `tail` index, and then — the only MMIO in the hot path —
writes the new tail to a **doorbell register**. The GPU's front-end
processor wakes, DMAs packets from `head` to `tail`, executes them, and
advances `head`. Submission is *asynchronous by construction*: the CPU is
free the moment the doorbell rings.

In `cmd.rs`, `Ring` is exactly this structure, including the classic
"capacity − 1" convention that keeps `head == tail` unambiguous as
"empty". `GpuNode::submit` plays the role of the doorbell write.

**Why virtualization cares:** because submission is mediated by *memory*
and a *single register write*, a mediator can leave the ring in guest
hands (fast path untouched) and trap only the doorbell. That trap is the
scheduling point where one physical front-end gets multiplexed across
tenants. This one observation is the entire basis of mediated passthrough.

## 2.2 Completion flows back through fences

The reverse direction is also not a return value. The driver appends a
**fence** command — "when everything before this point is done, write the
value N to this location" — and then polls that location or sleeps until
an interrupt. Fence values are monotonically increasing per queue, so
"fence ≥ N" means "everything I submitted before fence N is complete."

`Command::FenceSignal` and `Channel::completed_fence` model this;
`GpuNode::fence_value` is the guest's poll. CUDA's
`cudaStreamSynchronize`, Vulkan's `VkFence`, and DRM's sync objects are
all this mechanism wearing different clothes.

## 2.3 Channels: the unit of context and of blast radius

A GPU runs many independent command streams. Each stream — a **channel**
(NVIDIA's term; AMD says "queue", the concept is identical) — carries its
own ring, its own fence state, and crucially its own *fault domain*. When
a command dereferences a bad address, the hardware kills *that channel*:
the driver gets a fault notification (an NVIDIA "Xid" error), tears the
channel down, and every other channel keeps running.

`Channel` in `cmd.rs` implements precisely this, including `kill()`
draining queued work — and the integration test
`fault_blast_radius_is_one_channel` proves the containment property
end-to-end.

**Why virtualization cares:** fault containment is what makes
multi-tenancy survivable. A tenant *will* submit a wild pointer
eventually; the design question is who pays. The answer must be: only
that tenant's channel.

## 2.4 The GMMU: the GPU has its own MMU, and it is the isolation boundary

Modern GPUs execute all memory accesses through a **GPU MMU** — per-context
page tables, radix-tree structured, walked by hardware, with TLBs. A
kernel launched on channel X dereferences pointers *in channel X's
virtual address space*. There is no way for the kernel to name a physical
VRAM address at all.

This is the deepest fact in the whole subject:

> **GPU memory isolation is not access control checked per operation.
> It is the inability to even express another tenant's address.**

Tenant A's pointer `0x1_0000` and tenant B's pointer `0x1_0000` walk
different page tables and land in different VRAM frames. There is no
"check" that could be skipped, no flag that could be forgotten — the
translation *is* the security. (CPU process isolation works identically;
GPUs just arrived at it twenty years later.)

`gmmu.rs` implements a two-level version: a page directory of page
tables, 64 KiB pages, valid/writable bits, byte-granular `translate` and
scatter-gather `translate_range`. The engine (`engine.rs`) refuses to
touch a byte of VRAM except through it.

## 2.5 VRAM management: allocation, fragmentation, scrubbing

Under the page tables sits physical VRAM, and three unglamorous problems
that dominate real driver code:

* **Allocation & fragmentation.** VRAM outlives any tenant; after months
  of churn you must still find frames for a new tenant's buffers. Linux
  DRM ships a dedicated **buddy allocator** (`drm_buddy`) for this;
  `vram.rs` implements the same algorithm, and its docs explain the
  split/coalesce machinery and the XOR buddy trick.
* **Sparseness of truth.** Page tables mean a tenant's contiguous virtual
  buffer can be backed by scattered physical frames — so allocators never
  need to defragment; the GMMU glues fragments together. `vgpu.rs`
  exploits this with binary-decomposition allocation (a 5-page buffer =
  a 4-frame block + a 1-frame block, mapped contiguously in VA).
* **Scrubbing.** VRAM is not zeroed by power cycles between tenants. If
  freed frames are handed to the next tenant unscrubbed, tenant B reads
  tenant A's model weights. Real vGPU managers zero VRAM on reassignment;
  `FrameStore::scrub` + the `freed_vram_is_scrubbed_before_reuse` test
  pin the behavior here.

## 2.6 Preemption: the granularity of sharing

Time-slicing tenants requires taking the GPU away from one and giving it
to another. *Where* you can do that is the hardware's preemption
granularity, and it has improved one architectural generation at a time:
draw-call boundaries → command boundaries → thread-block boundaries →
instruction-level preemption (NVIDIA Pascal, 2016).

`vgpu-core` models **command-boundary preemption**: the tick loop
(`node.rs`) never interrupts a command, so a time slice can overrun by up
to one command's cost. The scheduler charges the *actual* cycles including
the overrun, and the vruntime mechanism (chapter 5 of the walkthroughs)
automatically compensates by picking that tenant later next time. This is
exactly the accounting posture of real slice-based schedulers.

## 2.7 The model in one table

| Hardware reality | `vgpu-core` model | Where |
|---|---|---|
| Ring buffer + doorbell | `Ring`, `submit()` | `cmd.rs`, `node.rs` |
| Fence values | `FenceSignal`, `completed_fence` | `cmd.rs` |
| Channel = fault domain | `Channel`, `kill()` | `cmd.rs` |
| GMMU radix page tables | 2-level `AddressSpace` | `gmmu.rs` |
| VRAM buddy allocation | `VramAllocator` | `vram.rs` |
| VRAM scrubbing | `FrameStore::scrub` | `vram.rs` |
| Copy/compute engines | `execute()` | `engine.rs` |
| Slice scheduler in firmware | vruntime `Scheduler` | `sched.rs` |
| The mediator process | `GpuNode` | `node.rs` |

What is deliberately *not* modeled (yet): shader execution (kernels are
opaque cycle costs until the milestone-2 interpreter), TLBs and their
shootdowns, interrupts (completion is observed by polling fences), and
multi-engine parallelism (one execution front-end). Each is noted in the
code where it would attach.

Next: how these pieces compose into VGPU-X, and the two invariants that
carry the isolation argument — [03-architecture.md](03-architecture.md).
