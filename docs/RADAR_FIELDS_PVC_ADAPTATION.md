# Radar Fields on PVC: alteration ledger (2026-09-21)

Status: **complete 2026-09-22 (RF agent): gates 1-3 passed, gate 2 closed on H100
(2153856, 2154357) and PVC (2155072); gate 4 awaits an H100 run of this checkout.
D1–D4 approved by the user on 2026-09-21; design in section 6, implementation
record in section 7 and in `RIFT_PVC_Adaptation.md` (Package E).**

Note on the production runs: the four completed collection runs (campaign root
`production_20260921_rf_rs_14jobs`) run a snapshot that predates the gate-2
closure, so their checkpoints record the parity status "diagnostic" with replay
job 2154358. The current tree records the closed gate (job 2155072).
This document records every deliberate difference between the CUDA comparison
profile (`docs/RADAR_FIELDS_ADAPTATION.md`, `source-adapted-v3`,
`--model-backend upstream-tcnn`) and its PVC (Intel XPU) counterpart. Nothing in
the CUDA files changes; the PVC code lives in `rift_pvc/` and
`train_radar_fields_pvc.py` (see `AGENTS.md`, PVC section).

**Review follow-up (2026-09-21):** the user accepts unavoidable lower-level
floating-point differences when the model and implementation semantics are
preserved. Numerical parity measurements below are diagnostics; they do not
gate training or recovery. The original tolerance failures and skipped
references remain recorded. Recovery now checks the saved reduction strategy,
and smoke acceptance verifies artifacts/progress and rejects CPU fallbacks.
See `RIFT_PVC_Adaptation.md` section 12 for the independent review and fixes.

## 1. What blocks a direct port

The comparison profile executes the authors' unchanged `RadarField(use_tcnn=True)`
from the pinned release. That class builds three `tcnn.Encoding`/`tcnn.Network`
objects: a HashGrid encoding (16 levels, 2 features per level, `log2_hashmap_size`
19, base resolution 16, `per_level_scale = exp2(log2(512/16)/15)`), a
SphericalHarmonics encoding (TCNN `degree` 4, 16 outputs, inputs mapped `2x-1`
inside TCNN), and three FullyFusedMLPs (ReLU, no output activation, width 64,
one hidden layer each, bias-free, fp16 tensor-core arithmetic with outputs padded
to multiples of 16). tiny-cuda-nn is CUDA-only; there is no SYCL port and its
fully fused kernels are not portable. The authors' own `use_tcnn=False` branch is
**not** a fallback: it swaps the hash grid for a Fourier `PositionalEncoding` and
uses biased `nn.Linear` layers, i.e. a different model. The repository's existing
`--model-backend torch` (`rift/radar_fields.py`) is a recorded ablation backend
with documented deviations (SH convention, initialization, MLP structure); it is
not the comparison implementation either.

## 2. Decisions

**D1 (approved 2026-09-21): replace only the kernel library, not the model.** Provide a
pure-torch, API-compatible `tinycudann` shim (`rift_pvc/tcnn_torch/`) exposing
`Encoding`, `Network`, `NetworkWithInputEncoding` with the same config dicts,
`n_output_dims`, and a single flat `params` parameter per module. Inject it as
`sys.modules["tinycudann"]` **only inside the PVC process**, so the authors'
`RadarField(use_tcnn=True)` code path, `get_params` grouping, BatchNorm placement,
activations and the wrapper `OriginalRadarFieldsModel` run byte-identical.
Rejected alternatives: the authors' non-TCNN branch (different model); the
repository torch backend (documented deviations).

**D2 (approved 2026-09-21): mirror TCNN semantics exactly where they are defined by source.**
The shim must reproduce, from the pinned tiny-cuda-nn source in the campaign
snapshot (`external/tiny-cuda-nn`):
- HashGrid: vertex-grid layout, 0.5 stagger, 8-entry table padding, hashing only
  for levels exceeding `2**19`, coherent primes `(1, 2654435761, 805459861)`,
  trilinear interpolation, per-level resolution schedule, parameter memory
  layout (`include/tiny-cuda-nn/encodings/grid.h`), init uniform `±1e-4`
  (`grid.h:1078`). The repository's `HashGridEncoder(layout="tcnn")` is the
  starting point and must be re-audited line by line against `grid.h`.
- SphericalHarmonics: TCNN's real-SH coefficient table and ordering for
  `degree` 4 with the `2x-1` input mapping (`encodings/spherical_harmonics.h`).
- FullyFusedMLP: bias-free layers, ReLU, output padding to 16 then truncation to
  `n_output_dims`, Xavier-uniform initialization (`src/fully_fused_mlp.cu:890`),
  weight-matrix layout in the flat `params` tensor
  (`networks/fully_fused_mlp.h`), and the installed binary's `param_precision()`
  for grid and network params (record it from the H100 build; expected fp16).
- Module `seed` argument accepted and used to seed the shim's own generator; the
  TCNN pcg32 stream itself cannot be reproduced, so initial values differ in
  value but not in distribution (recorded, see section 4).

**D3 (approved 2026-09-21): precision.** The shim computes in fp32 by default. TCNN's fp16
tensor-core arithmetic is a hardware artifact of the release, not a modeling
choice; fp32 is at least as accurate. A `RIFT_PVC_TCNN_HALF=1` mode runs the MLPs
under `torch.autocast("xpu", torch.float16)` and casts params to fp16 for parity
tests. Reported PVC results state "fp32 shim".

**D4 (approved 2026-09-21): identity.** The PVC checkpoint/plan records
`model_backend: "upstream-tcnn-torchshim"` (never `upstream-tcnn`), the shim
version, precision mode, and the parity-test outcome. `check_model_backend`'s
twin accepts XPU for this backend only; the CUDA env keeps refusing it.

## 3. What stays identical

Authors' `RadarField` class and forward, `OriginalRadarFieldsModel` wrapper,
optimizer/scheduler factories evaluated from the pinned `main.py`, BatchNorm
population rules, released epoch mask, 100-profile sampler, ray averaging,
occupancy helper, sealed roles, `source-adapted-v3` recipe values
(`protocols/radar_fields_rift_*_source_adapted_v3.json`), losses, metrics,
checkpoint schema apart from the backend identity, and the GOTCHA native adapter
(`rift/radar_fields_gotcha.py`, copied only where it names CUDA).

## 4. Implementation checks and numerical diagnostics

1. **Unit parity of each shim component against the source formulas** (CPU,
   `rift_pvc/tests/test_tcnn_torch.py`): hash indices and interpolation weights
   for chosen points and levels against a direct re-evaluation of `grid.h`
   arithmetic; SH values against the TCNN table; MLP forward against explicit
   matrix products with the same padding.
2. **Forward/gradient parity against the real tiny-cuda-nn on an H100**
   (`gpu_debug`, campaign build in `runtime/b787_radar_fields/site`): copy the
   real `tcnn` params into the shim through the mirrored layout, evaluate both on
   identical query batches; forward within fp16 tolerance (relative 1e-3),
   gradients w.r.t. params and inputs within 1e-2 relative were the original
   proposed numerical criteria. Report actual differences and missing
   reference coverage; the follow-up policy above does not require exact
   cross-device arithmetic.
3. **Bounded real-data smoke on PVC** through `train_rift_dataset_pvc.py --method
   radar_fields` (B787, `source-adapted-v3`, reduced `--steps`), no XPU-to-CPU
   fallback warnings, checkpoint save and resume, wall time per step.
4. **Trajectory comparison** with the H100 run once one exists for the current
   checkout (the 2026-09-20 smoke crashed before training on a since-fixed
   geometry bug): the same recipe on both backends, compared by validation
   metrics, not values (initialization streams differ, section 2 D2).

## 5. Open limits

- Initial parameter values differ from a TCNN run with the same seed (pcg32 vs
  torch RNG). Distributions match; recorded in the checkpoint.
- fp16 accumulation-order effects of the fused kernels are not reproduced.
- Weight gradients: `torchshim-2` reduces the per-query partial products in a
  fixed order (block-partial GEMMs), which makes PVC runs bitwise reproducible;
  the values agree with the plain GEMM to float32 rounding (1e-5 relative).
- Throughput: three small MLPs and a hash grid in eager torch on XPU will be
  slower than fused kernels; measured in gate 3. A `torch.compile` path on XPU
  is an optional later step, never a correctness dependency.

## 6. Implementation design (Package E)

**Files.** `rift_pvc/tcnn_torch/__init__.py` (exports `Encoding`, `Network`,
`NetworkWithInputEncoding`, `free_temporary_memory` no-op, `__version__`
`"torchshim-1"` at design time, now `"torchshim-2"` with fixed-order weight
gradients), `rift_pvc/tcnn_torch/hashgrid.py`, `spherical_harmonics.py`,
`fully_fused_mlp.py`, `layout.py` (TCNN parameter-layout map, used by the parity
tests and by any CUDA-to-PVC checkpoint conversion); `rift_pvc/radar_fields_upstream.py`
(imports the unchanged `rift.radar_fields_upstream`, installs the shim as
`sys.modules["tinycudann"]` **before** the first `original_module(...)` call and
only when `accelerator.backend() == "xpu"` or `RIFT_PVC_TCNN_SHIM=1`; rebinds
`check_model_backend` to accept `--device xpu` with backend id
`upstream-tcnn-torchshim`; the CUDA env's behaviour is untouched);
`train_radar_fields_pvc.py` (imports the unchanged `train_radar_fields`, rebinds
its seeding L88-89, the checkpoint RNG payload L1229 and its resume validation
L671-677 (`cuda_rng_state` stays for CUDA; `xpu_rng_state` twin plus
`accelerator_backend`; `cuda_rng_state_verified` semantics kept per backend),
the device list L1165 and the `--device` default L1341, through
`rift_pvc.accelerator`; the `--device cuda` that `train_rift_dataset.py` emits
for `source-adapted-v3` is remapped to `xpu` by the PVC frontend);
`rift_pvc/radar_fields_gotcha.py` (copy of `rift/radar_fields_gotcha.py` with
L223, L308-309, L312 default, L347-348, L360 (`rng_cuda`/`rng_xpu`) adapted);
`rift_pvc/radar_fields_released.py` only if L66/L86 are reached on the
production path (verify: `make_released_trainer` device default and CUDA check).
Audited, not adapted: `scripts/readout_radar_fields_b7873200_native.py`
(L342, L406-410, L492-496; a `scripts_pvc/` twin when a PVC readout is needed)
and `scripts/validate_radar_fields.py` (CUDA RNG round-trip checks, L435-502).

**Shim contract.** `Encoding(n_input_dims, encoding_config, seed=1337, dtype=None)`
and `Network(n_input_dims, n_output_dims, network_config, seed=1337)` are
`nn.Module`s with exactly one `nn.Parameter` named `params` (flat, so the
authors' `get_params` yields five one-tensor groups as with TCNN), attributes
`n_input_dims`, `n_output_dims`, `dtype`, `native_tcnn_module=None`, and
`forward(x)` accepting `[N, n_input_dims]` float32 (TCNN casts inputs to fp32).
`otype` support: `HashGrid`, `SphericalHarmonics`, `FullyFusedMLP`; any other
`otype` raises. Config keys are read exactly as `tcnn_utils.py` writes them
(`n_levels`, `n_features_per_level`, `log2_hashmap_size`, `base_resolution`,
`per_level_scale`, `degree`, `activation`, `output_activation`, `n_neurons`,
`n_hidden_layers`). Output dtype: fp32 (D3 default) or fp16 under
`RIFT_PVC_TCNN_HALF=1`; the padded MLP output is truncated to `n_output_dims`.

**Layout map (`layout.py`).** From `grid.h`: per-level `offset_table`,
resolution schedule, vertex-count vs hash decision, 8-entry padding, feature
interleaving; from `fully_fused_mlp.h`/`network.h`: input padding to a multiple
of 16, hidden width, `padded_output_width`, weight-matrix order and storage
(row-major `[out, in]` per layer, fp16), no biases. `layout.slice(module) ->
{name: (offset, shape)}` must reproduce the TCNN `params` vector ordering so
that `shim.params.copy_(tcnn.params.float())` gives the same function. The
implementer must read these sources; do not infer the layout from behaviour.

**Parity job (gate 2).** `scripts_pvc/parity_tcnn_h100.sbatch` on `gpu_debug`
(`--gres=gpu:h100:1`, CUDA env plus `PYTHONPATH=.../runtime/b787_radar_fields/site`
for the campaign's built tiny-cuda-nn) runs `scripts_pvc/parity_tcnn_dump.py`:
builds the five production modules through `radarfields.nn.models.RadarField`
with the `source-adapted-v3` arguments (`hidden_dim` 64, `feature_dim` 32,
`hash_levels` 16, `hash_final_resolution` 512, `hash_log2_size` 19, SH degree 4,
BatchNorm on), records `param_precision()`/`params.dtype` per module, draws
4096 unit-cube points and unit directions, saves params (fp32), inputs, outputs
and `d(sum(out))/d(params)` and `/d(inputs)` to an NPZ. On PVC,
`rift_pvc/tests/test_tcnn_parity.py` loads the NPZ, copies params through the
layout map, and asserts: encodings forward 1e-6 (fp32) / 1e-3 (fp16 mode),
MLP forward 1e-3, gradients 1e-2 relative, output shapes and padding identical.

**Smoke (gate 3).** `train_rift_dataset_pvc.py --object b787 --method
radar_fields --radar-fields-recipe source-adapted-v3 --num-train 2400 --num-tx
1 --num-rx 1 --pvc-steps 48` (frontend bound: `--steps`, `--eval-every`,
`--checkpoint-every` rewritten, `--checkpoint-name` suffixed `_pvcsmoke`), then
the campaign-style SIGTERM/resume check; report ms/step, no fallback warnings.
GOTCHA: `train_gotcha_dataset_pvc.py --method radar_fields --device xpu` once
`rift_pvc/radar_fields_gotcha.py` exists.

**Identity.** Checkpoint keys `model_backend: "upstream-tcnn-torchshim"`,
`tcnn_shim_version`, `tcnn_shim_precision` (`fp32` | `fp16`), `parity_job`
(H100 job id) written by the twin; readouts copy them. As implemented, the
version, precision and parity record are nested in one `tcnn_shim` checkpoint
entry rather than stored as flat keys. A checkpoint carrying
`upstream-tcnn` is refused by the PVC twin and vice versa.

## 7. Implementation record (2026-09-21)

Delivered: `rift_pvc/tcnn_torch/` (shim; `layout.py` transcribes the offset
table, hashing, MLP matrix order and Xavier bounds from `grid.h`,
`common_device.h`, `fully_fused_mlp.cu`, `cpp_api.cu`,
`network_with_input_encoding.h`, `identity.h`, `encoding.cu`),
`rift_pvc/radar_fields_upstream.py`, `rift_pvc/radar_fields_training.py`,
`train_radar_fields_pvc.py`, `rift_pvc/radar_fields_gotcha.py`, the frontend
rewrites, `scripts_pvc/parity_tcnn_dump.py` + `parity_tcnn_h100.sbatch`,
`scripts_pvc/plan_radar_fields_pvc.py` + `smoke_radar_fields_b787_pvc.sbatch`,
tests `rift_pvc/tests/test_tcnn_torch.py`, `test_tcnn_parity.py`,
`test_radar_fields_pvc.py`, `test_radar_fields_xpu.py`. The full audit table
and the job outcomes are in `RIFT_PVC_Adaptation.md`.

Deviations from section 6, each with its reason:

- `--pvc-steps` was not implemented: `source-adapted-v3` fixes `steps` at
  parse time (`train_radar_fields.py` L1436-1440). Gate 3 keeps the production
  argument list and is wall-clock-bounded (SIGTERM, `checkpoint_latest`,
  identical command with `--resume`); the frontend's `--pvc-smoke` suffixes
  `--checkpoint-name` only.
- `train_radar_fields.main()` is copied into the entry point: its resume block
  (L1675-1691) restores `cuda_rng_state` inline for a CUDA device only and was
  missing from the audit; the copy calls `restore_device_rng_state` and is
  otherwise verbatim (pinned by a structural test).
- Source facts folded into the shim contract: standalone `Encoding` objects
  have no output padding (alignment 0); `Network` is a
  `NetworkWithInputEncoding` over an `Identity` encoding aligned to 16 whose
  padding entries are 1; `params` stay float32 in the torch module and are cast
  to fp16 at forward; the SH kernel writes its padding ones before the
  coefficients (non-JIT path); `encode_angle` has an empty `params`
  Parameter. Float32 `grid_scale` gives 33/65/129/257/513 vertices at levels
  3/6/9/12/15 (10,523,376 grid parameters); asserted against the H100 dump.
- Only what the release uses is mirrored (Linear interpolation, CoherentPrime
  hashing, SH degree ≤ 4, FullyFusedMLP widths 16/32/64/128, no SIREN init);
  other configurations raise instead of approximating.
- Gate-2 tolerance for the fp32 HashGrid against the fp32-precision TCNN twin
  is 1e-4 (device-side `exp2f` ulp of the level scale), SH 1e-6, fp16 forward
  1e-3, gradients 1e-2.
- The loss line of the trainer can print `bim=nan`: `released_batch_loss`
  keeps upstream's undefined singleton-group sample std and `nan_to_num`s it in
  the total (`rift/radar_fields_native.py` L256-279); the loss and gradients
  stay finite. Identical on CUDA; not an XPU effect.

Gate results (details and job accounting in `RIFT_PVC_Adaptation.md`):

- Gate 1 (formula parity, CPU): 16 tests pass; production layout 10,523,376 /
  0 / 4096 / 3072 / 4096 parameters; the release model runs through the shim
  with `atol=0` wrapper parity and identical `get_params` groups.
- Gate 3 (B787, production argument list, PVC): job 2153861 ran all 960 steps
  in 824 s wall (0.59 s/step, four 1000-view validations, 0 XPU→CPU fallbacks):
  validation rel-MSE 29.72 / 31.57 / 36.76 / 41.83 % at steps 240/480/720/960,
  best 29.72 % at step 240; resume from the step-720 checkpoint reproduced the
  final validation to 1e-6 relative. Job 2153877 exercised the interrupt path
  (SIGTERM during the step-480 validation → `checkpoint_latest` → identical
  command with `--resume` → completion at step 960 in 415 s). A full
  `source-adapted-v3` run is ~14 min on one PVC card; the trainer is
  renderer-bound (the release model's fwd+bwd for 32768 queries takes 20.7 ms
  warm on XPU).
- Reproducibility on XPU (`torchshim-1`): two fresh runs with the same seed
  were bit-identical for steps 1-3 and diverged from step 10 (1 % in the loss
  by step 200, 0.4 % absolute in validation rel-MSE at step 960). Probes traced
  it to the MLP weight-gradient GEMMs (K = queries per step, ~1e6): oneDNN
  splits K and accumulates in a nondeterministic order (also for chunked K ≥
  4096 and in fp64), while the shim's grid backward, BatchNorm statistics and
  global reductions are deterministic. `torchshim-2` computes the weight
  gradient as batched partial GEMMs over row blocks of 256 reduced by a
  fixed-order `sum(0)` (`rift_pvc/tcnn_torch/linear.py`,
  `RIFT_PVC_TCNN_WEIGHT_GRAD=bmm:256`), the faithful counterpart of
  tiny-cuda-nn's fixed-order split-K reduction, at ~1 ms per 1e6 rows per
  matrix. With it, two fresh runs (jobs 2153958/2153959) print identical
  losses for 480 compared steps and identical validations, and a resume from
  an interrupt checkpoint reproduces the fresh run's later validations
  exactly (36.990967 % / 41.721461 % at steps 720/960): PVC Radar Fields runs
  are bitwise reproducible and resume is trajectory-identical. Cross-backend
  (H100 vs PVC) comparisons remain by metrics (section 4, gate 4).
- Gate 2 (H100 replay, job 2153856): the real tiny-cuda-nn modules have
  exactly the shim's parameter counts and widths (fp16 params and outputs);
  against fp32-precision TCNN twins the shim's HashGrid agrees to 2.2e-5
  (forward and both gradients ≤ 3e-5) and the SH table to 2.2e-8; the whole
  release model through the shim matches the real model to 1.2e-5 (alpha) and
  5.1e-4 (rd). Differences to the fp16 production modules are TCNN's fp16
  arithmetic: 1.6e-3 grid (1.9e-4 in the shim's fp16 mode), 2.4e-3 fused-MLP
  forward at initialization scale (its 1e-4 inputs are fp16 subnormals),
  ≤ 5e-4 for the other two MLPs, parameter gradients ≤ 3.1e-3, input
   gradients up to 3.9e-2. MLP input gradients propagate into the trainable
   encoding and earlier networks. The 1e-3 fp16 forward
  tolerance of section 4 was an assumption; the parity test asserts 5e-3 /
  1e-2 / 1e-1 (forward / parameter / input gradients) against the fp16
  modules and 1e-4 / 1e-6 against the fp32 twins. The re-dump 2154357 (`mean`
  reduction, amplified records with the grid ×64 so features are fp16-normal)
  completed the picture: all references finite, `rd_net` parameter gradient
  2.7e-3, amplified forward ≤ 5.6e-4 for every module, amplified parameter
  gradients ≤ 8.6e-3 (grid) / ≤ 3.2e-3 (MLPs), end-to-end 2.9e-4 / 5.0e-4.
  The fp16 parity mode now applies TCNN's loss scaling (×128 on dL/dy, ÷128
  on the results, `scale_grad` in `module.py`); with it the fp16-mode gradients
  agree on the amplified records (params ≤ 3.2e-3, inputs ≤ 6.1e-3). At
  initialization scale the fp16 *reference* gradients are subnormal (TCNN
  computes input gradients in fp32, the shim's fp16 mode does not), so the
  parity test asserts gradients on the amplified records and reports them
  at init scale. PVC replay 2155072: all 32 parity tests pass on XPU with the
  same numbers. **Gate 2 closed.**
- GOTCHA smoke and on-card resume: passed (jobs 2153872, 2154228, 2154229);
  validation at step 150 pooled rel-MSE 0.2999 vs the H100 campaign's 0.2966.

Identity written by the twins: checkpoint `args.model_backend` and
`radar_fields_recipe.model_backend` = `upstream-tcnn-torchshim`,
`radar_fields_recipe.encoding` = `original_radarfield_tcnn_torchshim`,
`radar_fields_recipe.tcnn_shim` = `{version, precision}`, top-level
`accelerator_backend`, `xpu_rng_state`, `tcnn_shim` (`{model_backend, version,
precision, tcnn_source_commit, parity}`); GOTCHA checkpoints carry `rng_xpu`,
`accelerator_backend`, `tcnn_shim`. `rift_pvc.tcnn_torch.PARITY` records
numerical diagnostic provenance and coverage, not a strict-equivalence
certificate. Checkpoint recovery validates the saved `tcnn_shim.weight_grad`
along with backend/version/precision, while accepting updated diagnostic
metadata. Existing torchshim-2 checkpoints retain their original recipe and
output identity. Follow-up job **2154384** completed on PVC with **82 tests
passed**, three numerical-reference skips, exact synthetic native GOTCHA
recovery, rejection of changed reduction settings, and correct failure on an
injected fallback warning. No model or arithmetic kernel was changed by that
follow-up.
