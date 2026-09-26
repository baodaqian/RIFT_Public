# Radar Fields source and acquisition adaptation

Public-release packaging note: this repository includes the referenced source
as ordinary files in `external/RadarFields_reference/`, with the upstream URL
and commit recorded in `SOURCE_PROVENANCE.json`. Statements below about an
untracked local checkout describe the original development repository.

The collection and native GOTCHA selectors now use **`source-adapted-v3`**.
The complete current decision table, root integration and remaining limitations
are in [docs/RADAR_FIELDS_ADAPTATION.md](../docs/RADAR_FIELDS_ADAPTATION.md).
This document retains the audited-v2 engineering record and dependency build
instructions. Its 64-ray / 8,000-step recipe is historical, not the current
comparison default. Bare historical NPZ calls retain `legacy-v1`; named-object
calls use v3. Recipes and dataset-specific checkpoints are not interchangeable.

## Sources and runtime dependency

- [Paper v2](https://arxiv.org/html/2405.04662v2), especially §§3.1–3.4.
- [Supplement](https://light.princeton.edu/wp-content/uploads/2024/07/Radar-Fields-Supplement.pdf),
  especially §§1.1–1.3 (integration, occupancy, architecture and losses).
- [Official source](https://github.com/princeton-computational-imaging/RadarFields/tree/ee76d76570f58b3d8539eafd7df0c188b58af333),
  commit `ee76d76570f58b3d8539eafd7df0c188b58af333`.
- [tiny-cuda-nn source](https://github.com/NVlabs/tiny-cuda-nn/tree/749dd70c5afc5a9dadb85e5652ed65d55e0ba187),
  commit `749dd70c5afc5a9dadb85e5652ed65d55e0ba187` (2.0, JIT left disabled).

`external/RadarFields_reference/` is now an explicit runtime dependency for
`audited-v2` and `source-adapted-v3`. It is an unmodified local checkout; the upstream repository has no
license file at this commit, and its source is not vendored into RIFT's tracked
code. `rift/radar_fields_upstream.py` checks that required files exist without hashing
and imports the actual network, Bayesian occupancy helper, ray averaging and
intensity mapping. There are no runtime downloads or silent fallbacks. On
another installation obtain the official checkout at the exact commit in that
directory and install the pinned tiny-cuda-nn Torch bindings.

`--model-backend upstream-tcnn` is the audited default and requires an allocated
CUDA GPU. The separately recorded `--model-backend torch` is a portable testing
and ablation backend: dense/coherent-prime vertex hash grids, bias-free MLPs,
and physical real SH directions. It is **not numerically interchangeable** with
the release's TCNN SH convention, fused arithmetic, or initialization. CPU tests
do not certify the original CUDA implementation.

## Earlier audited-v2 disposition (superseded for the comparison default)

| Area | Historical behavior | `audited-v2` |
| --- | --- | --- |
| Intensity mapping | Helper default offset 0.05 | Upstream trainer default 1.0, scaler 1.0; empty RCS predicts zero normalized intensity |
| Spatial/angular model | Handwritten encoder/MLPs | Original TCNN `RadarField`, feature mask, BN, sigmoid occupancy and softplus reflectance |
| Range rendering | Fractional voxel splatting divided by occupied cell mass | Native bin-center bistatic ellipsoid samples; ray average includes zero exterior values |
| Occupancy target | Thresholded normalized intensity | Per-profile noise rejection plus original occlusion-aware Bayesian probabilities |
| KL/bimodality | Per-view flattened KL sum; binary groups at 0.5 | Global distributions, KL sum divided by physical frame count, groups at 0.01, sample std |
| BN | Training chunks updated statistics independently | One BN population across the frame batch; frozen statistics during evaluation |
| Training exposure | Random choices could omit train views | Persisted shuffled passes with per-view exposure counts and exact continuation |
| Boundary/readout | Coordinate clamping; generic voxel renderer | Reject out-of-box queries, zero exterior ray samples, shared training/readout renderer |

Two audit qualifications matter. With one physical frame, KL `sum` and
`batchmean` agree. With binary targets, the old 0.5 and 0.01 group cuts agree.
These were not established explanations of the old production result.
Empty/singleton std groups now contribute zero instead of upstream NaNs.

The source uses a noise multiplier of 1.5 and strict `>`; the supplement writes
2 and `>=`. The source's evidence-decay recurrence is retained literally,
including repeated decay by distance from the last stronger return. The source
optimizer is Adam `(0.9, 0.99)`, epsilon `1e-15`; the supplement describes AdamW
`(0.9, 0.999)`. This recipe follows the release. The paper gives both R^-2
(Eq. 6) and R^-4 (Eq. 9); `released`, `code_r2`, and `paper_r4` remain explicit
recipe-bound alternatives, with the release's no-range-law default.

## Earlier audited-v2 acquisition choices

The original sensor is a mechanically scanned azimuth/range Navtech radar.
RIFT has calibrated Tx/Rx positions and coherent frequency responses. The
adapter coherently averages redundant chirps, takes `abs(IFFT(response))**2`,
then uses one fixed training-only peak and declared dB dynamic range. This is
not the vendor's proprietary FFT-image normalization. Test/unused responses
remain sealed; stats retain object and exact training-role identities.

Each pair's native bin at one-way range R is sampled by Tx-origin rays satisfying
`(|x-tx| + |x-rx|)/2 = R`, calculated in float64. Deterministic equal-solid-angle
quadrature covers the registered cube's bounding-sphere cone, with unit gain.
The original `avg_rays` integrates these samples. Samples outside the cube have
zero occupancy/RCS but remain in the averaging denominator. There is no measured
antenna LUT for these archives, so Navtech's beam pattern is not transplanted.
`--ray-samples` (default 64) is recorded in the recipe; quadrature convergence
needs a development-data study. `--granularity` controls compatibility geometry
readout, not audited training integration. Training queries are one batch for
BN correctness; `--query-chunk` limits evaluation queries, not training memory.

MIMO channels are not azimuth beams. Occupancy uses each pair's range median,
not a cross-pair azimuth median that would reject returns shared by the array.
It is formed on the full radial profile before ROI cropping and is invariant
to which other pairs are selected. This is a necessary sensor adaptation,
not a reproduction of Navtech's two-dimensional filter.

Finite-band coherent power has sidelobes and phase-dependent interference.
Noncoherent Radar Fields cannot generally reproduce these cross-terms. Tests
check analytic on/off-bin point spread functions and constructive/destructive
two-reflector cases; they establish the limitation rather than claiming a
beam-average renderer becomes a coherent simulator. Re/Im metrics remain
`N/A by construction`. Whole-ROI metrics retain unsupported bins; supported
and padded scores are diagnostics, with the same zero-RCS reference.

Automotive grounding, mounting-height penalties and pose refinement are
disabled for calibrated object-centered sphere acquisitions. No auxiliary
geometry or geometry initialization is used. Compatibility geometry export is
direction-independent occupancy sampled on the declared grid; common RIFT
geometry thresholds are not the paper's reflectance-masked point extraction.
The later native GOTCHA backend is implemented in `rift/radar_fields_gotcha.py`;
its source/region/role/recipe-bound checkpoints are distinct. See the current
adaptation document for both dataset routes.

## Use and verification

### PACE dependency build

Installed in the existing RIFT environment: `tinycudann==2.0`, from the pinned
commit above. The package inventory comparison found only that addition and
no upgrades/removals; a separate RIFT-RadarField environment was unnecessary.
The local wheel, successful build log and installed-binary hashes are retained
under `external/RadarFields_dependencies/`; `installation.json` records the
toolchain and verification limits. Wheel SHA-256:
`f146cc9f9e38c6f28385a32509743f3240fd79631fd67488816f6f13d563290c`.

The build targets the existing RIFT Python 3.10 / PyTorch `2.6.0+cu124` stack,
using modules `anaconda3`, `nvhpc-cuda/12.4` and `gcc/12.3.0`. Build with
`TCNN_CUDA_ARCHITECTURES=75,80,90` and `MAX_JOBS=2`; TCNN chooses the highest
compatible installed variant (other supported intermediate capabilities use
a lower variant and may report reduced performance). The CUDA 12.4 driver
stubs belong on `LIBRARY_PATH` **only for linking on a GPU-free build node**,
never on `LD_LIBRARY_PATH` during execution. An actual NVIDIA driver/GPU is
required at runtime. JIT fusion remains disabled.

Reproduce from the pinned tiny-cuda-nn checkout with:

```bash
module load anaconda3 nvhpc-cuda/12.4 gcc/12.3.0
conda activate RIFT
LIBRARY_PATH="$CUDA_HOME/lib64/stubs:$LIBRARY_PATH" \
  TCNN_CUDA_ARCHITECTURES=75,80,90 MAX_JOBS=2 \
  python -m pip wheel --no-deps --no-build-isolation --no-index \
  --wheel-dir /tmp/rift-radarfields-wheels \
  /path/to/pinned-tiny-cuda-nn/bindings/torch
python -m pip install --no-deps --no-index \
  /tmp/rift-radarfields-wheels/tinycudann-2.0-cp310-cp310-linux_x86_64.whl
```

Use the same module environment for allocated runtime checks. `import
tinycudann` itself checks GPU availability, so an import failure on a login
node is not an allocated forward/backward test. Full neural validation is the
CUDA-marked test in `tests/test_radar_fields_audited.py`.

### Planning and readout

Metadata-only collection planning:

```bash
python train_rift_dataset.py --object a320 --method radar_fields --dry-run
```

The manager can use a new `--output-root` for a source-adapted run. To continue old
collection checkpoints, select `--radar-fields-recipe legacy-v1` and matching
resume/output arguments. Full fitting and quadrature/convergence studies remain
experiment-manager work; this hardening task does not submit them.

Native collection readout uses an explicit matching configuration:

```bash
python scripts/readout_radar_fields_b7873200_native.py --object a320 \
  --config protocols/radar_fields_rift_source_adapted_v3.json \
  --checkpoint /path/to/radar_fields/checkpoint_final.pth.tar \
  --roles validation --output /path/to/new/readout.json
```

Audited readout also accepts a validation-selected intermediate checkpoint from
the same configured schedule. It verifies the saved full schedule and recipe;
it does not require the best checkpoint to occur at the final optimizer step.
The historical v1 production readout retains its final-step restriction.

`tests/test_radar_fields_audited.py` checks original helper parity, analytic
geometry/power cases, exposure and exact interrupted continuation, sealed
response access, and shared readout. Its original-network forward/backward
and chunk-parity test requires CUDA and explicitly skips without it.
`scripts/validate_radar_fields.py --upstream external/RadarFields_reference`
continues to validate the frozen legacy implementation. Tests and source parity
do not establish real-data reconstruction quality or GPU resource requirements.

Implementation verification on 2026-09-19: 251 tests passed across the audited
RF, metric-domain and three collection API/script suites; one CUDA-only test
was skipped. The legacy validator passed all 94 checks. A320 collection planning
was metadata-only. The install check verified all three binary files against
the wheel; no CUDA forward/backward or benchmark-response fitting was run.
