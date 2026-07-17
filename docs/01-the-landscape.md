# 1. The GPU Virtualization Landscape

Before a single line of code, the conceptual foundation — because GPU
virtualization is a place where five *genuinely different* techniques all
get called "vGPU", and confusing them makes every design discussion
incoherent. The question that separates them is always the same:

> **At which interface do you split the stack in two?**

A GPU workload passes through a tall stack of interfaces on its way to
silicon:

```
  application
      │  ①  API calls           (CUDA, Vulkan, OpenGL…)
  user-mode driver / runtime
      │  ②  ioctl / command buffers
  kernel-mode driver
      │  ③  MMIO registers, rings, doorbells, interrupts
  PCIe device
      │  ④  the hardware's own internal partitioning
  silicon (SMs, copy engines, VRAM, MMU)
```

Every virtualization technique picks ONE of those numbered lines, puts the
tenant on top of it, and puts a *mediator* underneath it. Everything about
the technique — performance, isolation strength, feature completeness,
migration story — falls out of that single choice.

## Technique 1: API remoting (split at ①)

Intercept the API itself. The guest links against a fake `libcudart` that
serializes every call (`cudaMalloc`, `cudaMemcpy`, `cudaLaunchKernel`) and
ships it to a server that replays it on a real GPU. Examples: rCUDA,
vCUDA, VirtualGL for OpenGL, and every "GPU-over-network" product.

* **Virtualizes:** a *library*. The guest never sees a device at all.
* **Strengths:** works across a network; no hypervisor involvement; the
  server can pack many clients per GPU with full software control.
* **Weaknesses:** you must reimplement/forward an enormous, evolving API
  surface; anything that bypasses the API (pointer arithmetic inside
  kernels, custom PTX) still works only because the *server's* real stack
  handles it; latency per call.
* **Isolation:** whatever the server process enforces — i.e., software.

## Technique 2: Para-virtualization (split at ②)

The guest runs a *modified* driver that knows it is virtualized and speaks
an idealized transport to the host instead of touching hardware. Example:
`virtio-gpu` (with Venus for Vulkan, or virgl for OpenGL): guest submits
command buffers over a virtio queue; the host's renderer executes them.

* **Virtualizes:** the *driver interface*, redesigned for the purpose.
* **Strengths:** clean, stable contract; guest drivers are simple; good
  fit for desktop virtualization.
* **Weaknesses:** the host still re-encodes work for the real GPU (a
  translation tax); compute support has historically lagged graphics.
* **Isolation:** the host renderer process + whatever the GPU driver gives.

## Technique 3: Mediated passthrough (split at ③) — where VGPU-X lives

Give the guest the *real* hardware interface for the fast path, but trap
the control path. The guest's unmodified driver writes command rings in
memory it owns and reads/writes a small set of virtualized registers; a
mediator traps doorbell writes and register accesses, installs the guest's
GPU page tables, and time-slices the real engines between guests.
Examples: NVIDIA vGPU (GRID), Intel GVT-g, the Linux VFIO/mdev framework.

* **Virtualizes:** the *device programming model*: rings, doorbells,
  page tables, interrupts.
* **Strengths:** near-native fast path (the GPU DMAs the guest's ring
  directly); unmodified guest drivers; fine-grained sharing with software
  policy (weights, slices); migration is possible because the mediator
  can see and serialize all state.
* **Weaknesses:** the mediator is intricate, privileged software; every
  register the guest may touch must be emulated or safely exposed.
* **Isolation:** hardware page tables per guest + software mediation of
  everything else.

This is the interesting layer, and it is the layer VGPU-X's device model
implements: per-tenant GPU page tables, command rings with trapped
doorbells, and a mediator (`GpuNode`) that owns every physical resource
and time-slices compute. Milestone 2's guest shim will add technique 1 on
top — the two compose naturally, which is exactly how real fleets deploy
them.

## Technique 4: SR-IOV / hardware partitioning (split at ④)

The PCIe device itself presents multiple *virtual functions* (VFs), each a
real PCIe endpoint with its own registers, its own rings, its own MMU
context — multiplexed by the card's hardware/firmware. Examples: AMD
MxGPU, Intel Flex/Ponte Vecchio SR-IOV, NVIDIA vGPU on Ampere+ (which
moved its data plane onto SR-IOV VFs).

* **Virtualizes:** the *PCIe device*, in silicon.
* **Strengths:** strongest isolation short of separate cards; near-zero
  mediation overhead; the hypervisor mostly just assigns VFs.
* **Weaknesses:** fixed, coarse partition counts baked into the hardware;
  little room for software policy; feature availability depends entirely
  on the vendor.

## Technique 5: Spatial partitioning — MIG (also ④, different axis)

NVIDIA's Multi-Instance GPU slices the *silicon itself*: an A100/H100 can
be split into up to 7 instances, each with dedicated SM slices, dedicated
L2 cache slices, and dedicated memory controllers. This is not
time-sharing at all — it is space-sharing, with hardware-enforced
performance isolation (no noisy neighbors even at the cache level).

Contrast with time-slicing (what techniques 3 and CUDA's own MPS do):
time-slicing shares 100% of the GPU serially; MIG shares fractions of the
GPU in parallel. Real deployments combine them: MIG instances, each
time-sliced by vGPU profiles.

## The three resources you must virtualize (whatever the split)

Every technique above, at whatever layer, ends up answering the same three
questions. These three answers are the skeleton of VGPU-X's core crate:

1. **Memory** — who may touch which bytes of VRAM?
   → per-tenant translation (page tables) + allocation + scrubbing.
   (`vram.rs`, `gmmu.rs`)
2. **Compute** — whose commands run next, and for how long?
   → submission queues + a scheduler with a fairness contract.
   (`cmd.rs`, `sched.rs`, `node.rs`)
3. **Faults** — when a tenant misbehaves, who pays?
   → fault containment: the blast radius must be the offender's context,
   never the device. (`engine.rs`, channel kill in `node.rs`)

Keep the taxonomy and those three questions in mind and every design
decision in the code has a place to hang. Next: the machine model —
what a GPU actually *is* to the software programming it —
in [02-how-a-gpu-executes.md](02-how-a-gpu-executes.md).
