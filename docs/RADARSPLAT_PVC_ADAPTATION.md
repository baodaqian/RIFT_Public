# RadarSplat on PVC: alteration ledger (2026-09-21)

Status: **decided. D1, D2, D4, D5 approved by the user on 2026-09-21; D3 (SYCL
kernel port) deferred and not to be started unless absolutely needed.
Implementation design in section 6 and in `RIFT_PVC_Adaptation.md` (Package F).**
This document records every deliberate difference between the CUDA comparison
profile (`docs/RADARSPLAT_FIDELITY.md`, `budget48`: the authors' fork at
`ea9c8f530c708622cc3b1b560436b5557ac6a49b`, its CUDA rasterizer, fused-SSIM,
112000 planar Gaussians, 2000 updates) and its PVC (Intel XPU) counterpart.
Nothing in the CUDA files changes; PVC code lives in `rift_pvc/` and
`train_radarsplat_pvc.py`.

## 1. What blocks a direct port

`rift/radarsplat_release.py` imports the fork's `gsplat.rendering` and calls
`_radar_rasterization`, which runs these fork ops: `cartesian_to_spherical`
(torch, fork-specific), `fully_fused_projection` (CUDA), `isect_tiles` and
`isect_offset_encode` (CUDA), `rasterize_to_pixels` in radar mode with separate
power and occupancy products and per-product 1/255 cutoff (CUDA), and
`spherical_harmonics` (CUDA, degree ≤ 4), followed by the torch filters
`spectral_leakage` and `azimuth_antenna_gain_projection`. The loss uses
`fused_ssim(..., padding="valid")` (CUDA, 444-line kernel). The fork ships torch
reference functions (`gsplat/cuda/_torch_impl.py`, `_torch_impl_radar.py`) for
projection, tile intersection, SH and a **partial** radar rasterizer; that
rasterizer still calls the CUDA op `rasterize_to_indices_in_range_radargs` and
requires `nerfacc`, so the fork's torch path is not self-contained. gsplat has no
XPU build. `docs/RADARSPLAT_FIDELITY.md` forbids substituting the repository's
older independent renderer for the comparator; that rule stands.

## 2. Decisions

**D1 (approved 2026-09-21): Tier 1, a pure-torch mirror of the fork's radar kernels.**
`rift_pvc/radarsplat_xpu_backend.py` provides a `rendering`-compatible module
whose `_radar_rasterization` is the fork's function with each CUDA call replaced
by a torch function that mirrors the corresponding fork kernel **line by line**:
`fully_fused_projection` (from the fork's `_fully_fused_projection`, ortho
camera, `sph=True` path), `isect_tiles`/`isect_offset_encode` (fork torch
versions), `spherical_harmonics` (fork `_spherical_harmonics`, degree ≤ 4), and
a new torch `rasterize_to_pixels` radar mode written from
`rasterize_to_pixels_fwd.cu`/`_bwd.cu` and `rasterize_to_indices_in_range_radargs.cu`:
front-to-back order within tiles, `classic` mode, the fork's power and occupancy
products, `noise_probs`, per-product `1/255` cutoff and early termination rules.
Backward is torch autograd over the mirrored forward; the CUDA backward's
hand-written gradients are the parity reference, not the implementation. The
repository's `additive_gaussian_rasterization` may be reused only for its tile
plan and candidate-pair machinery, never for its compositing semantics.

**D2 (approved 2026-09-21): fused-SSIM.** Use the exact mathematical equivalent already in
the repository, `rift/radarsplat_fidelity.py::release_ssim_index` (11x11,
sigma 1.5, valid padding, unit-range stabilizers), generalized to the batched
NCHW call shape the release loss uses. Gate: agreement with `fused_ssim` on an
H100 to 1e-6 on values and gradients.

**D3 (deferred by the user 2026-09-21; only if absolutely needed): Tier 2, SYCL kernels.** If Tier 1
cannot run 2000 updates of 112000 Gaussians within one 48-hour `pvc` job, or if
the user wants kernel-level faithfulness, migrate the fork's CUDA sources (about
9000 lines under `gsplat/cuda/csrc`, plus `ssim.cu`) with the cluster's
SYCLomatic (`dpct`), build them as a `torch.utils.cpp_extension.SyclExtension`
(available in torch 2.12.1+xpu) with `icpx` from `intel-compilers/2025.1.1` in a
**build-only** shell (that module must never be loaded in the run shell, it
breaks the XPU wheel's runtime loader). Parity gate identical to Tier 1.

**D4 (approved 2026-09-21): gsplat import on PVC.** The fork's `_backend.py` JIT-compiles
CUDA on import. The PVC twin loads `gsplat.cuda._torch_impl*` and `rendering`
without triggering that path (pre-seed `gsplat.cuda._backend._C` with a guard
object, or import the modules by file); it must fail loudly if any CUDA op is
reached. `rift_pvc/radarsplat_release.py::load_xpu_reference` replaces
`load_cuda_reference`; it keeps the source-inventory check and the
already-imported-gsplat rejection.

**D5 (approved 2026-09-21): identity.** Checkpoints, plans and readouts record
`radarsplat_backend: "fork_torch_mirror_xpu_v1"` (or `"fork_sycl_port_v1"` for
Tier 2), the SSIM implementation, and the parity-test outcome. PVC results are
never presented as runs of the released CUDA renderer.

## 3. What stays identical

The fork's initializer and preprocessing ASTs, `_radar_rasterization` control
flow and constants, filters, loss weights and terms, Adam groups and schedule,
2000 updates, `budget48` Gaussian count, dataset conversion ledger, occupancy
donors, multipath provider, sealed roles, recovery schema (plus the backend
identity), and the GOTCHA adapter (`rift/radarsplat_gotcha.py`, copied only where
it names CUDA).

## 4. Gates before any PVC RadarSplat number is reported

1. **Op-level parity on saved inputs.** On an H100 (`gpu_debug`, the campaign's
   `tools/dependencies.sh` recipe builds gsplat and fused-SSIM per job): dump one
   real `budget48` state (splats, pose, K, grid) and the CUDA outputs of every
   op above plus gradients; on PVC, run the torch mirror on the same tensors.
   Tolerances: projection/SH/intersections 1e-5 relative; rasterized products
   1e-4 relative per pixel and identical cutoff masks; gradients 1e-3 relative
   (restated 2026-09-22: 1e-3 relative in L2 per parameter group, with max-norm
   outliers above 1e-3 counted and explained; a single near-zero-gradient Gaussian
   of 112000 differs by 5e-3 of the group maximum on CUDA-side fp32 backward
   arithmetic, see `RIFT_PVC_Adaptation.md` 8.F). Report the numbers here.
2. **Loss parity.** Same batch, full `release_loss`: values within 1e-5.
3. **Bounded real-data smoke on PVC** via `train_rift_dataset_pvc.py --method
   radarsplat --radarsplat-recipe budget48` (B787 1t1r): target-cache
   preparation, then a reduced number of updates, checkpoint save and resume,
   no fallback warnings, wall time per update; extrapolate to 2000 updates and
   decide Tier 2.
4. **Trajectory comparison** with an H100 run of the same checkout (none exists
   yet: the 2026-09-20 smokes died on a faulty GPU node) by metrics.

## 5. Open limits

- Tier 1 throughput is unknown until gate 3; the dense parts of a torch
  rasterizer scale with candidate pairs, not pixels, so the tile plan matters.
- Autograd backward will not match the CUDA backward's accumulation order
  bitwise; gate 1 tolerances bound the difference.
- The fork's `_torch_impl_radar` partial rasterizer is a reference for the
  power/occupancy accumulation, not a drop-in.

## 6. Implementation design (Package F)

**Scope correction (2026-09-21, from the fork source).** In the production
radar branch (`sph=True`, `rasterize_mode="classic"`, `camera_model="ortho"`)
the fork's `_radar_rasterization` already performs the per-pixel accumulation
in torch: it calls `_rasterize_to_radar_pixels` (fork `_torch_impl_radar.py`)
six times for the products (powers, occupancy, noise_probs, opa_refl,
noise_refl, reflectance) with the fork's own torch `accumulate`/`sum_weights`
(the `nerfacc` import is commented out). CUDA is reached only through
`fully_fused_projection` (L153 of the function), `isect_tiles` (L188),
`isect_offset_encode` (L211), `spherical_harmonics` (L235) and, inside
`_rasterize_to_radar_pixels`, `rasterize_to_indices_in_range_radargs`. Tier 1
therefore needs torch mirrors of those five ops only; the first four exist in
the fork as `_fully_fused_projection` (ortho supported), `_isect_tiles`,
`_isect_offset_encode`, `_spherical_harmonics` and must be checked for
argument parity with the wrapper calls (`packed=False`, `calc_compensations`,
`eps2d`, near/far planes, `masks`); the fifth is written in torch from
`rasterize_to_indices_in_range_radargs.cu` (300 lines: for a chunk of
Gaussians in depth order per tile, emit (pixel, gaussian) pairs whose
alpha = min(0.999, opacity·exp(-σ)) ≥ 1/255 within tile bounds, with the
kernel's transmittance/early-stop rule and chunk offsets). The non-radar
`_rasterize_to_pixels`/`rasterize_to_pixels` calls (L350-365) are not on the
production path; the implementer confirms by tracing `sph=True`.

**Files.** `rift_pvc/radarsplat_xpu_backend.py`: `load_xpu_reference(root,
device)` twin of `load_cuda_reference` (inventory check via the unchanged
`verify_reference(root, cuda_dependencies=False)`; the already-imported-gsplat
rejection kept; `sys.path` insert; imports the fork's `gsplat.rendering`; then
rebinds in that module's namespace `fully_fused_projection`, `isect_tiles`,
`isect_offset_encode`, `spherical_harmonics` to the torch mirrors and, in
`gsplat.cuda._wrapper`, `rasterize_to_indices_in_range_radargs` to the new torch
function, so the fork's `_radar_rasterization` and `_rasterize_to_radar_pixels`
run unchanged; returns `(rendering, fused_ssim_torch)`). gsplat import on PVC:
the fork's `_backend.py` sets `_C = None` with a warning when no CUDA toolkit
is found (`cuda_toolkit_available()` false), and the wrappers fetch `_C`
lazily, so a plain import is expected to succeed; the loader asserts
`gsplat.cuda._backend._C is None` and that every rebound name is in place, and
fails loudly if any wrapper reaches `_C`. `rift_pvc/gsplat_torch_ops.py`: the
five mirrors (`fully_fused_projection_xpu`, `isect_tiles_xpu`,
`isect_offset_encode_xpu`, `spherical_harmonics_xpu`,
`rasterize_to_indices_in_range_radargs_xpu`) with the wrapper signatures.
`rift_pvc/fused_ssim_torch.py`: `fused_ssim(img1, img2, padding="same",
train=True)` returning `map.mean()`; 11x11 Gaussian window σ=1.5, C1=0.01²,
C2=0.03², per-channel depthwise `conv2d`; `same` = zero padding 5 as the CUDA
kernel computes the full map, `valid` = map cropped `[5:-5, 5:-5]`; derived from
`rift/radarsplat_fidelity.py::release_ssim_index`. `rift_pvc/radarsplat_release.py`:
imports the unchanged module; `create_scene` L86 default via the shim;
`load_cuda_reference` rebound to `load_xpu_reference`. `rift_pvc/radarsplat_release_training.py`:
imports the unchanged module; rebinds `load_cuda_reference` (L130, L238, L291)
and the `--device` default L270; `train(cache, output, device, rendering,
fused_ssim, profile)` is otherwise unchanged. `train_radarsplat_pvc.py`: imports
the unchanged trainer; L110-112 and L1501-1506 memory payload (`cuda_max_*` on
CUDA, `xpu_max_*` on XPU), L172 `--device` default `accelerator.device()`, L267
availability gate per type, L1241-1242 peak reset via the shim.
`scripts_pvc/prepare_radarsplat_b7873200_targets_pvc.py`: twin of the
first production command (L92-94, L123, L188, L336-337). `rift_pvc/radarsplat_gotcha.py`:
copy with L24 import and L334 `load_xpu_reference`. Readouts
(`scripts/readout_radarsplat_checkpoint.py` device-free;
`scripts/readout_radarsplat_gotcha.py` L18/L34) get `scripts_pvc/` twins when
needed. The PVC frontend maps `scripts/prepare_radarsplat_b7873200_targets.py`
to its twin and passes `--device xpu` (Package D follow-up).

**Parity job (gate 1).** `scripts_pvc/parity_radarsplat_h100.sbatch` on
`gpu_debug` with the campaign's `tools/dependencies.sh` recipe (builds the
fork's CUDA extension and fused-SSIM into a job-local site) runs
`scripts_pvc/parity_radarsplat_dump.py`: takes one real `budget48` B787 target
cache and a fresh `create_scene` state (seed 42), and saves for one training
image the inputs and CUDA outputs of `fully_fused_projection`, `isect_tiles`,
`isect_offset_encode`, `spherical_harmonics`, `rasterize_to_indices_in_range_radargs`,
the six `_rasterize_to_radar_pixels` products, the final power/occupancy
images, `release_loss` and its gradients w.r.t. the splat parameters, and
`fused_ssim` value/gradient on a random pair. `rift_pvc/tests/test_radarsplat_parity.py`
replays them on PVC with the tolerances of section 4.

**Bounded smoke (gate 3).** The release CLI fixes 2000 updates for `budget48`
(no step flag), so the smoke is wall-clock bounded: run the two production
commands from `train_rift_dataset_pvc.py --object b787 --method radarsplat
--radarsplat-recipe budget48 --num-train 2400 --num-tx 1 --num-rx 1` (target
preparation, then fitting), SIGTERM the fitting process group after N minutes,
then `--resume auto` with the identical command; report seconds per view for
preparation and seconds per update for fitting; extrapolate 2000 updates. If the
extrapolation exceeds one 48 h job, report it; D3 is not started without the
user's decision.

**Identity (D5).** The release checkpoint's `identity` dict is compared for
equality on resume, so the backend is **not** added to it. `train_radarsplat_pvc.py`
writes `backend.json` next to the checkpoints (`radarsplat_backend:
"fork_torch_mirror_xpu_v1"`, `ssim: "fused_ssim_torch_v1"`, torch and shim
versions, parity job id) and the PVC frontends/readouts copy it into their
reports. A checkpoint directory without `backend.json` is a CUDA run.

## 7. Implementation record (2026-09-21, RadarSplat agent)

Delivered under `rift_pvc/` and `scripts_pvc/` (nothing in `rift/`, `scripts/`
or the root CUDA trainers changed; the PVC frontend gained one script mapping):

| Deliverable | Content |
| --- | --- |
| `rift_pvc/gsplat_torch_ops.py` | the five mirrors with the wrapper signatures: `fully_fused_projection_xpu` (kernel `fully_fused_projection_fwd.cu` with `proj.cuh` ortho/pinhole/fisheye, `add_blur`, `inverse`, three-sigma radius, near/far/inside culling), `isect_tiles_xpu` (vectorised; the kernel's 64-bit key `cam << (32+tile_bits) \| tile << 32 \| int32 bits of depth`, stable sort on the radix-sorted bit range), `isect_offset_encode_xpu`, `spherical_harmonics_xpu` (`sh_coeffs_to_color_fast` constants, degree capped at 4 like the kernel), `rasterize_to_indices_in_range_radargs_xpu` (per-tile batch ranges, `alpha = min(0.999, opacity·exp(-σ))`, drop `σ<0` or `alpha<1/255`, kernel output order camera → pixel → intersection) |
| `rift_pvc/fused_ssim_torch.py` | `fused_ssim(img1, img2, padding, train)` = mean of the kernel's SSIM map: the `ssim.cu` tap literals, separable x-then-y convolution with zero padding, `valid` crop `[5:-5]`, `img2` detached (the extension returns no gradient for it) |
| `rift_pvc/radarsplat_xpu_backend.py` | `load_xpu_reference` (D4: `gsplat.cuda._backend` pre-seeded with a stub whose `_C` raises naming the op; five rebinds in `gsplat.rendering` and `gsplat.cuda._wrapper`; F-dev1 re-execution), `backend.json` sidecar helpers (D5) |
| `rift_pvc/radarsplat_release.py`, `rift_pvc/radarsplat_release_training.py`, `rift_pvc/radarsplat_gotcha.py` | the unchanged modules with the accelerator-specific names rebound (loader, device defaults, `ReleasedPreprocessing` with the release FFT on the accelerator, train/readout twins, sidecar-before-checkpoint, per-update timing) |
| `train_radarsplat_pvc.py`, `scripts_pvc/prepare_radarsplat_b7873200_targets_pvc.py`, `scripts_pvc/readout_radarsplat_gotcha_pvc.py`, `scripts_pvc/plan_radarsplat_pvc.py` | entry points and launch helper; `train_rift_dataset_pvc.py` maps the prepare script |
| `scripts_pvc/parity_radarsplat_dump.py`, `scripts_pvc/parity_radarsplat_h100.sbatch`, `rift_pvc/tests/test_radarsplat_parity.py` | gate 1 (recorded on an H100, replayed on PVC with the section-4 tolerances; the replay writes `<dump>.replay.json`) |
| `scripts_pvc/smoke_radarsplat_b787_pvc.sbatch` | gate 3 (wall-clock-bounded fit, SIGTERM, `--resume`) |
| tests | `rift_pvc/tests/test_fused_ssim_torch.py`, `test_gsplat_torch_ops.py` (mirrors vs the fork's torch references and vs literal kernel transcriptions), `test_radarsplat_pvc.py` (backend, engine, entry point, frontends), `test_radarsplat_xpu.py` (device) |

**F-dev1 (found while implementing, extends section 6).** Beyond the five CUDA
ops, the fork's *Python* holds five literal CUDA constructors on the production
path: `_radar_rasterization` (`.to('cuda')` ×2), `spectral_leakage` and
`azimuth_antenna_gain_projection` (`kernel = ....cuda()`), and
`boreas/data_processing/play_radar_signal.py::FFT` (`.to("cuda")`, reached by
`ReleasedPreprocessing.background`). The repository's own CUDA tests neutralise
them by monkeypatching `torch.Tensor.cuda`/`.to`. The PVC loader instead
re-executes each function from its own source with the literal replaced by the
input tensor's device (or the accelerator device for the preprocessing FFT) and
asserts the literal counts, so the functions remain the fork's text.

**Documented differences of the mirrors from the kernels** (none changes a
value the fork consumes): culled projections and masked SH colours are finite
zero/garbage instead of uninitialised memory; IEEE arithmetic instead of
`--use_fast_math`/`__expf` (candidates within float rounding of the 1/255
cutoff can differ; gate 1 counts them); the fork's per-Gaussian Python
`_isect_tiles` and its degree-5 `_spherical_harmonics` (uninitialised bases
25–35) are not used.

**Identity on disk.** Every PVC checkpoint directory carries `backend.json`
(written before each release-schema checkpoint by the rebound
`train_radarsplat._atomic_torch_save`); continuing a directory with
`checkpoint_latest.pt` but no sidecar (a CUDA run) is refused unless
`RIFT_PVC_RADARSPLAT_ALLOW_CUDA_RESUME=1`. The release `identity` dict is
untouched.

Gate results are recorded in `RIFT_PVC_Adaptation.md` section 8.F as the jobs
complete.

**Gate 1 outcome (2026-09-21 23:10, jobs 2153858 / 2154359; numbers in
`RIFT_PVC_Adaptation.md` 8.F).** Op level: projection radii identical, means2d
1.6e-9 and conics 1.5e-7 relative; tile keys, offsets and SH identical; the
radar index kernel's pair sets and order identical for all twelve product
calls (1.07–1.35 M pairs each); rasterized products 1.2e-6 relative. End to
end, the loss differs by 5.8e-5 and the parameter gradients by up to 5.6e-3
relative, exceeding section 4's 1e-5/1e-3; the whole difference is reproduced
by running the fork's filter convolutions on CPU over the CUDA-recorded
products, i.e. it is cuDNN's default TF32 convolution arithmetic in the H100
reference. A TF32-off reference (`NVIDIA_TF32_OVERRIDE=0`) states the mirrors' own
end-to-end error: images within 2.9e-5 and the loss within 3e-7 relative on
the first such run (job 2154367, whose radar-index recordings were corrupted
by the GPU it landed on and whose self-checked replacement is job 2154965);
section 4's loss and gradient tolerances are read against fp32 CUDA
arithmetic, with the TF32-on numbers reported alongside as the production
CUDA arithmetic. Reference dumps now self-check the radar index kernel
against the CPU mirror and record the GPU UUID.

**Gate 1 closed (2026-09-22 07:52, jobs 2154965 / 2155075; table in
`RIFT_PVC_Adaptation.md` 8.F).** On the PVC card, against the self-checked
fp32 CUDA reference: every op identical or within 1.5e-7; products ≤ 2.9e-5;
filtered images ≤ 3.1e-5 absolute; loss ≤ 3.4e-7 relative; gradients L2 ≤
7.0e-4 on the isotropic initial scene (one near-zero-gradient Gaussian at
5.6e-3 of the max norm) and ≤ 1.2e-4 L2 / 3.8e-4 max norm on the perturbed
anisotropic scene, for all three fixtures. The production CUDA arithmetic
(cuDNN TF32 convolutions) sits 2.3e-4 absolute / 5.5e-5 relative from both.
Gate 2 (loss parity) is closed by the same numbers. Gate 4 (trajectory vs an
H100 run of this checkout) remains open: no H100 RadarSplat trajectory exists.

Note on the production runs: the six completed collection runs (campaign root
`production_20260921_rf_rs_14jobs`) record parity job 2153858 with status
`diagnostic_original_tf32_reference_exceeds_strict_tolerances` in `backend.json`.
That snapshot's `production_job.sbatch` exported environment overrides that
shadowed the module constants, so the runs did not pick up the gate-1 closure. The
overrides are removed in the current tree (committed in 9416aa9), and
campaign roots prepared after that record the gate-1 values.

