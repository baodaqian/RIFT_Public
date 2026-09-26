# PVC (Intel Data Center GPU Max 1100) environment plan for RIFT on ACES

Assessment date: **2026-09-21**. Evidence comes from probe jobs run on the ACES
`pvc` partition and from a code audit of checkout `b32c134`. Nothing in the
production CUDA environment, the campaign root or the tracked source was changed.
Probe artifacts live under `/scratch/group/p.cis261724.000/RIFT_pvc_probe/`.

## 1. Verdict

- **Feasible for the RIFT adaptive trainer with moderate effort.** The float64
  coherent range operator (Gaussian gridding + `torch.fft`) runs forward and
  backward on a PVC card with the repo's exactness and adjoint gates satisfied.
- **The PVC environment must use `torch==2.12.1+xpu`, not the CUDA env's 2.6.0
  pin.** `torch 2.6.0+xpu` rejects every float64 or complex matmul/einsum
  ("Double and complex datatype matmul is not supported in oneDNN"), which the
  adaptive point-SH scene uses for its SH basis contraction, and it silently
  executes the operator's complex128 FFT on the CPU. 2.12.1 passes all such
  checks on the device and also provides the XCCL collective backend.
- **Two baselines need re-implemented kernels:** Radar Fields (tinycudann) and
  RadarSplat (gsplat + fused-SSIM CUDA kernels) have no Intel ports. The user's
  goal (2026-09-21) is all six models on PVC, with these two re-implemented as
  faithfully to the originals as possible; that is separate adaptation work with
  its own fidelity gates. SpINR, GeRaF and Sugavanam–Ertin are pure torch.
  Agent coordination for the port: `RIFT_PVC_Adaptation.md`.
- **Workload:** about 3–5 working days for a gated PVC production path for RIFT;
  2–3 more days for the three pure-torch baselines. See section 6.
- **Compute:** the operator costs ~40 ms per B787-like training view (forward +
  backward) and ~17 ms for inference on PVC. Operator-only cost of one B787
  epoch (2400 train + 1000 validation views) is therefore ~113 s, against a
  measured all-in H100 epoch of ~83 s (job 2150481). The planning estimate was
  a PVC epoch of roughly 130–180 s and a 150-epoch scene in about 5.5–7.5 h. The
  production campaign measured 191.5–260.1 s per epoch (mean ≈ 211 s) and
  8 h 47 min–8 h 52 min per completed collection scene, inside one 24 h `pvc` job. A direct H100 microbenchmark of the same operator (job 2152997) is
  still queued on `gpu_debug`. GPU memory is not a constraint (production B787
  uses < 2 GB; the card has 48 GB).

## 2. Verified cluster facts

| Item | Finding | Evidence |
| --- | --- | --- |
| Partition | `pvc`: 30 nodes, 116 `gpu:pvc` GRES, 96 cores / 500 GB per node, 2-day max, open to account `158648339640`, QoS `normal`; many nodes idle, probes started within a minute | `scontrol show partition pvc`, `sinfo` |
| GPU | Intel Data Center GPU Max 1100, 1 tile, 448 EUs, 49136 MB, `has_fp64=1`; 2, 4, 6 or 8 cards per node | job 2152956 on `ac010`, torch device properties |
| Driver stack on nodes | `intel-level-zero-gpu 1.6.33578.77`, `level-zero 1.24.0`, `intel-opencl 25.18.33578.77`, IGC 2.11.43, i915 backport 25.2.57; `ZE_AFFINITY_MASK` is set per allocated GPU by Slurm | job 2152956 |
| OS | RHEL 8.10 (not on Intel's supported list for XPU wheels, but the wheels import and run) | `uname -r`, probes |
| Cluster-provided Intel PyTorch | AIKit 2023.2 (`torch 2.0.1a0 + IPEX 2.0.110+xpu`) and `intelpython/2024.1.0_814` env (`torch 2.1.0.post3 + IPEX 2.1.40+xpu`): both far below RIFT's pins; useful only as proof that the driver works (AIKit ran fp64 matmul and complex128 FFT on the card) | job 2152956 |
| Wheels reachable from login node | `torch 2.6.0 … 2.12.1 +xpu` for cp310 at `https://download.pytorch.org/whl/xpu`; the 2.12.1 wheel pulls the whole oneAPI runtime (`onemkl-sycl-dft/blas/lapack`, `oneccl`, `dpcpp-cpp-rt`) so no oneAPI module is needed | `curl` of the index; venv builds |
| Module pitfall | Loading `intel-compilers/2025.1.1` alongside the XPU wheels breaks `import torch` (`libur_loader.so.0: version LIBUR_LOADER_0.10/0.12 not found`): its older Unified Runtime shadows the wheel's. **Do not load CUDA or Intel compiler modules in PVC jobs.** | job 2152983 |
| torchvision / torchaudio | Not imported anywhere in RIFT; drop them from the PVC spec | `grep` |
| File quota | User scratch is at 194k of 250k files; a torch XPU venv is ~23k files. Put the PVC env on group scratch (`/scratch/group/p.cis261724.000`, 500k-file quota, nearly empty) | `showquota` |

## 3. Probe results on a PVC card (jobs 2153072, 2153073, 2153080, 2153086, 2153132, 2153159)

| Check | torch 2.6.0+xpu | torch 2.12.1+xpu |
| --- | --- | --- |
| `torch.xpu` API used by train.py / launcher (seed_all, rng_state_all, memory stats, mem_get_info, device props) | present | present |
| float64 matmul, complex128 einsum | **FAIL** (oneDNN) | PASS (1e-13) |
| complex128 exp/mul/conj, `index_add_` gridding, fp64 fallback gridding | PASS (1e-15) | PASS |
| `torch.fft.fft/ifft` complex128 (both axes), rfft fp64 | 'PASS' but executed on CPU via silent fallback (see below) | PASS on device (1e-16) |
| autograd through `index_add` + `ifft` (complex128) vs CPU | PASS (3e-16) | PASS (4e-16) |
| `torch.utils.checkpoint` inside the range operator (grad mode) | works (timing test) | works |
| `linalg.inv/norm/cross` fp64, searchsorted/unique/bincount/scatter, grid_sample, interpolate, polar | PASS | PASS |
| `torch.autocast('xpu')` fp16/bf16, AdamW foreach and fused, `torch.save/load` | PASS | PASS |
| Range operator forward vs direct sum (repo gate 1e-9), subset gather | 2.2e-10 / 3.1e-10 PASS | same PASS |
| Range adjoint vs direct adjoint, dot test | 9e-11 / 2e-16 PASS | same PASS |
| Repo Stage C gradient gate on XPU (gradcheck; range-vs-brute pos/w_re/w_im, gate 1e-5) | PASS, 6.2e-8 / 7.6e-8 / 8.1e-8, identical to CPU; XPU-vs-CPU gradients agree to 3e-12 | PASS, same values |
| Repo Stage D adjoint gate on XPU (dot 1e-10, vs direct 1e-5, VJP 1e-10) | PASS, 9e-18 / 3.1e-10 / 3.8e-16 | PASS, 3e-18 / 3.1e-10 / 3.4e-16 |
| Silent XPU-to-CPU op fallback | **complex128 `torch.fft.ifft` falls back to CPU** (`Aten Op fallback from XPU to CPU` at `range_operator.py:239`); the earlier 'exact vs CPU' FFT result was this fallback | no fallback warning observed |
| Distributed backend for XPU | none (`xccl` unavailable, no oneCCL bindings) | `dist.is_xccl_available() == True` |
| Range operator timing, N=131072, 600 freqs, 1t1r, fp64, point_chunk 65536 | fwd+bwd 43 ms/view, inference 18 ms, peak 0.10 GiB | fwd+bwd 40 ms/view, inference 17 ms, peak 0.08 GiB (H100 PCIe, torch 2.6.0+cu124: 19 ms / 4 ms, job 2152997) |
| `torch.Generator(device='xpu')` + `rand(generator=)` | never completed: killed at the 25-minute job limit inside the Intel graphics JIT (libigc mapped, 95 % CPU) | never completed within a 15-minute job |
| Cross-device reproducibility of rendered responses | XPU and CPU agree only to 1.1e-3 relative in a 4x4-array test, on both versions. Cause isolated (job 2153159): `rift/forward_operator.py::get_array_pos` builds array positions in **float32**; XPU and CPU round one element differently by one float32 ulp (4.8e-8 relative), and at the 10 m array radius that is a 2e-3 rad carrier-phase shift. The direct sum shows the same 1.1e-3 gap, the operator matches the direct sum on either device to 2.2e-10, and grad and no-grad paths agree to 1e-16. This is a pre-existing float32 property of the geometry helper, not an XPU defect; H100 runs inherit it too. | same |

The generator hang matters little for RIFT: the trainer uses CPU generators
(`_FREQ_RNG` and the scene's CPU generator); only a smoke runtime builds a device
generator. It is still a warning that some kernel variants never finish JIT
compilation, so PVC jobs must avoid device-side generators, set a persistent SYCL
kernel cache, and treat a silent multi-minute stall at first use as a failure
mode to watch for (section 5, step 1; section 7).

The float32 array-position finding means that PVC and H100 renderings of the
same view can differ at the 1e-3 level for views whose angles round differently.
This is far below the reported metrics (validation relative MSE is a few 1e-3),
so method comparisons are unaffected, but bitwise parity between platforms is not
available. Promoting `get_array_pos` to float64 would remove it; that is a code
change outside this plan and would itself alter every existing result at 1e-3.

## 4. Code coupling audit (what has to change for the RIFT method)

The trainer is already nearly device-agnostic. CUDA appears in the RIFT critical
path only here:

| Location | What it does | Change |
| --- | --- | --- |
| `rift/distributed.py:84-100`, `:187` | picks `cuda:<local_rank>`, chooses `nccl`/`gloo`, buffers for all-reduce | add an XPU branch (`xpu:<local_rank>`, backend `xccl`) |
| `train.py:234` (`set_seed`) | `torch.cuda.manual_seed_all` | route through the accelerator shim |
| `train.py:312-358` (`capture_rng_state`, `restore_rng_state`) | saves `torch_cuda_all`, refuses resume if CUDA topology differs | add `torch_xpu_all`; keep the strict contract per device family |
| `rift/adaptive_training_workflow.py:118` | single-GPU gate on `torch.cuda` | generalize |
| `train_rift_dataset.py:154` | passes `--device cuda` for Radar Fields only | no change for `--method rift` |
| campaign `tools/manage.py:265-266` | preflight asserts CUDA and `'H100' in device name` | PVC launcher variant accepting `Data Center GPU Max` |
| `scripts/validate_range_operator.py` (stages B, E), `scripts/validate_range_operator_inference_cache_transition.py:78-80`, `tests/test_training_architectures.py` (2 refs), `.local-setup/gpu-smoke.sbatch` | CUDA-only gates and smoke | generalize the device choice |

`rift/range_operator.py`, `rift/forward_operator.py`,
`rift/coherent_radar_geometry.py`, `rift/collection_adaptive.py`,
`rift/checkpointing.py`, `rift/config.py` and
`scripts/validate_adaptive_capacity_v2.py` contain no CUDA references.

Baselines: SpINR (`train_spinr_style.py`, 7 CUDA calls + 3 literals, TF32
flags), GeRaF (`train_geraf.py` 11 + 2, `rift/geraf_source_training.py` 5 + 4,
four `torch.amp.autocast("cuda")` sites: three in the vendored renderer and one in
`rift/geraf_source.py`, cudnn flags),
Sugavanam–Ertin (`train_sugavanam_ertin*.py` ~13 + 8, workflow 2). GOTCHA
training (`rift/gotcha_training.py` 3 + 4). Radar Fields and RadarSplat depend
on CUDA-only compiled kernels and are out of scope.

## 5. The plan

**Step 0. Scope.** RIFT adaptive trainer on the six-object RIFT dataset first.
Fresh PVC runs with their own provenance; no cross-device continuation of the
H100 checkpoints (different torch version and RNG family; a continuation would be
state-compatible but not trajectory-identical, which the adaptive-v2 contract
refuses).

**Step 1. Environment (0.5 day). DONE 2026-09-21:** `requirements-pvc.yaml`,
`.local-setup/activate-pvc.sh`, `.local-setup/verify_environment_pvc.py`,
`.local-setup/pvc-verify.sbatch`; env at `/scratch/group/p.cis261724.000/envs/RIFT-PVC`,
verified on `ac025` (job 2153218). Original step text: Add `requirements-pvc.yaml`:
`python=3.10.16`, same build tools, `--extra-index-url
https://download.pytorch.org/whl/xpu`, `torch==2.12.1+xpu`, the same pins for
numpy 1.26.0, scipy, matplotlib, wandb, plyfile, tqdm, pandas, scikit-image,
seaborn, configargparse, PyYAML, imageio, pillow, ninja, rich, pytest; no
torchvision/torchaudio/jaxtyping; open3d from conda-forge as on this host.
Create it on group scratch, e.g. `/scratch/group/p.cis261724.000/envs/RIFT-PVC`.
Add `.local-setup/activate-pvc.sh` that sets `PYTHONNOUSERSITE=1`,
`PYTHONDONTWRITEBYTECODE=1`, `MPLBACKEND=Agg`, `SYCL_CACHE_PERSISTENT=1`,
`SYCL_CACHE_DIR=<scratch>/sycl_cache` (verified to cache JIT-compiled kernels
across jobs), `PYTORCH_DEBUG_XPU_FALLBACK=1`, and loads **no** modules. Add an XPU
variant of `verify_environment.py` (imports, `torch.xpu.is_available()`, fp64
matmul, complex128 FFT, `index_add_`).

**Code-isolation rule (user, 2026-09-21).** PVC work must not modify the
existing CUDA pipeline. New code lives in `rift_pvc/` and `scripts_pvc/`, with
per-dataset entry points carrying a `_pvc` postfix
(`train_rift_dataset_pvc.py`, `train_gotcha_dataset_pvc.py`). The call-site
table in section 4 therefore describes what to copy-and-adapt into `rift_pvc/`,
not edits to `rift/` or `train.py`. PVC smoke jobs are pre-authorized.

**Step 2. Accelerator shim and adapted trainer (1–1.5 days). DONE 2026-09-21** (`rift_pvc/`,
`train_pvc.py`, `train_rift_dataset_pvc.py`, `train_gotcha_dataset_pvc.py`; 33 device tests pass).
Original text: New `rift_pvc/accelerator.py`
exposing `backend()` (`cuda` | `xpu` | `cpu`), `device_for_rank()`,
`manual_seed_all`, `device_count`, `synchronize`, `empty_cache`, memory stats,
`get_rng_state_all`/`set_rng_state_all`, `collective_backend()`
(`nccl` | `xccl` | `gloo`). Apply the table in section 4 to copies under `rift_pvc/` and to the new
`_pvc` entry points; the CUDA files are not touched, so existing tests are
unaffected by construction. Unit
tests for the shim and for RNG capture/restore on each family.

**Step 3. Gates on PVC (1 day). Smoke DONE 2026-09-21:** B787 2-epoch run on PVC (job
2153287) reproduces the H100 trajectory to printed precision at 195 s/epoch vs 70-80 s;
resume verified exact (job 2153305: epoch 3 after `--resume` reproduces the uninterrupted
epoch 3 and the H100 epoch 3 to printed precision). Original text: Run everything with
`PYTORCH_DEBUG_XPU_FALLBACK=1` and treat any `Aten Op fallback from XPU to CPU`
warning as a gate failure (a fallback silently moves work to the CPU). On a
`pvc` allocation: the ported
`validate_range_operator.py` stages A–E, the cache-transition regression,
`tests/test_training_architectures.py` (30 tests) and
`scripts/validate_adaptive_capacity_v2.py`. Then a bounded real run:
`train_rift_dataset.py --object b787 --method rift --num-train 2400 --num-tx 1
--num-rx 1` for 2–3 epochs, comparing per-epoch train/validation relative MSE
against the H100 log of job 2150481 and confirming a checkpoint save/resume on PVC.

**Step 4. Launcher and Slurm (0.5–1 day).** Under `scripts_pvc/`: a PVC campaign root (separate from
`h100_smoke_20260920_6d58645`) with a `manage.py` preflight that accepts
`Data Center GPU Max` and records driver/torch versions; job template
`--partition=pvc --gres=gpu:pvc:1 --cpus-per-task=8 --mem=64G --qos=normal
--account=158648339640`, time limit chosen from the measured epoch time (the
2-day partition limit allows the full 150 epochs in one job if PVC is within
about 6x of H100). Document provenance in the cross-cluster notes.

**Step 5 (optional). Baselines (2–3 days).** SpINR, then Sugavanam–Ertin, then
GeRaF (autocast device type, cudnn/TF32 flags, memory reports). Radar Fields and
RadarSplat: not portable.

**Step 6 (optional). Multi-GPU scene sharding.** `xccl` is available in 2.12;
the current 1t1r runs need < 2 GB, so this is not required.

## 6. Difficulty and workload

| Work item | Difficulty | Effort |
| --- | --- | --- |
| Environment build and verification | low (pip wheels bundle oneAPI; driver present) | 0.5 d |
| Accelerator shim and RIFT-path edits with tests | low–moderate (about a dozen call sites, one RNG contract) | 1–1.5 d |
| Gates and bounded B787 run on PVC | moderate (first real training on XPU; possible op gaps surface here) | 1 d |
| PVC launcher, Slurm templates, docs, provenance | low | 0.5–1 d |
| **RIFT production path total** | | **3–4 d, plan 5 d with buffer** |
| SpINR, Sugavanam–Ertin, GeRaF ports | moderate | 2–3 d |
| Radar Fields, RadarSplat | not feasible without CUDA-kernel ports, and against policy | — |

## 7. Risks and mitigations

1. **Torch version divergence (2.12.1 on PVC vs 2.6.0 on H100).** Results are
   comparable but not bit-identical; record it as separate provenance. Do not
   resume H100 checkpoints on PVC.
2. **First-use JIT compilation and stalls.** Most kernels compile in 1–15 s on
   first use; the device-generator variant never finished in 25 min, and one
   first operator call stalled while another stalled compile shared the node.
   Mitigate with `SYCL_CACHE_PERSISTENT=1` and a scratch `SYCL_CACHE_DIR`
   (verified to populate: 153 files / 919 MB after one run), a warm-up pass, no
   device-side `torch.Generator`, and a launcher watchdog that reports a
   first-epoch stall instead of burning the allocation.
3. **Unsupported ops appearing in full training.** The probe covered the
   operator, gridding, FFT, optimizer and checkpointing, not every scene routine.
   Step 3's bounded run is the gate; keep a CPU-fallback list.
4. **Performance.** PVC operator time is ~40 ms/view, which bounds a B787 epoch
   below by ~113 s versus ~83 s all-in on H100; expect 1.5–2.2x slower per
   epoch. Measure the real epoch time in step 3 before choosing the job time
   limit; the 48 h partition limit leaves ample margin.
5. **OS support.** RHEL 8.10 is outside Intel's documented matrix; it works
   empirically, but driver updates by HPRC could change behaviour. Pin the probe
   as a regression check.
6. **Quota.** Build the env on group scratch to protect the running production
   job's checkpoint writes from the user-scratch file quota.
7. **Launcher contract.** The campaign manager hard-codes H100; a PVC variant is
   required rather than a flag on the existing root.

## 7b. Package D results (2026-09-21)

- RIFT dataset, B787, production recipe on one Max 1100: epochs 1-3 reproduce the
  H100 run to printed precision (recipe draws all randomness from CPU
  generators and uses `--init-scale 0`); 195-200 s per epoch vs 70-80 s on H100;
  exact resume from `checkpoint_latest` verified (jobs 2153287, 2153305).
- GOTCHA Camry HH, built-in RIFT backend: ~6.9 s per pass-sector update on PVC vs
  ~6.8 s on H100 (not GPU-bound); trajectories differ from H100 by construction
  because the scene is initialized from the device RNG (`init_scale=1e-3`). This
  was the per-pulse loop over all pulses with the unit amplitude and random start.
  Later Camry runs use the batched lane (0.271 s per update, job 2155254), and the
  current tree defaults to the backprojection start; see
  [TRAINING_RECIPES.md](TRAINING_RECIPES.md).

## 8. Open items

- H100 timing of the same operator benchmark: DONE (job 2152997, H100 PCIe):
  fwd+bwd 19 ms/view, inference 4 ms/view, vs 40 ms / 17 ms on the Max 1100
  (2.1x / 4.2x). The all-in B787 epoch ratio measured in training is 2.5x.
- JIT stalls: the device-generator check never completed (jobs 2153034 TIMEOUT
  at 25 min, 2153039 at 15 min). One further stall hit the diagnostic's first
  operator call on `ac023` while another stalled JIT process was on the node
  (job 2153154, cancelled); the identical script then ran in seconds on `ac025`
  with `SYCL_CACHE_PERSISTENT=1` (job 2153159), which wrote 153 cached kernel
  files (919 MB). Step 3 must confirm no stall recurs under the cache.
