# GeRaF-SENS source attribution

Copyright (c) 2026 Laboratory of Sensing and Networking Systems, EPFL.
Licensed under the included PolyForm Noncommercial License 1.0.0.

Source: https://github.com/VictorLlu/GeRaF-SENS/tree/38266cb6e194e2f3dcbead614069a7281ffd21a5

The following definitions are copied with their bodies unchanged. Registry
decorators and unused imports are omitted; local dependency bindings are added.
The original source paths and definition AST hashes are in `source_manifest.json`.

- `Embedder`, `get_embedder`, `SDFNetwork`, `SingleVarianceNetwork`,
  `ReflectivePowerNetwork`: `geraf/models/networks/sdf_network.py`.
- `make_predictor`, its activation modules: `geraf/models/model_utils/field.py`.
- `BaseRendering`: `geraf/models/rendering/base.py`.
- Shared helpers, `GeRaFStageBase`, complete `GeRaFStage1` including its loss and
  antenna-bank lifecycle: `geraf/models/rendering/rf_rendering.py`.
- `UniformRaySampler`, `TargetRaySampler`, `DynamicLossMask`:
  `geraf/datasets/transforms/sample.py`.
- `_sample_volume`, `InterpolateMFAtTargets`: transforms/loading.py.
- `SceneToUnitSphere`, `AntennasToUnitSphere`: transforms/transform.py.
- `IterLRScheduler`: `tools/train.py`.

The positional-encoding source credits https://github.com/bmild/nerf.
Unmodified signal-tracer and matched-filter CUDA files are retained as
`*_reference.cu`, with whole-file hashes. They are references, not a claim that
these kernels were executed or their hardware-specific waveform was used on
RIFT/GOTCHA. `stage1_config_reference.py` and `train_reference.py` preserve the complete
original example and runner for auditing effective settings and the absent
step-hook call; they are not invoked by the root trainers. Their whole-file
hashes are also recorded. `registry.py` is a local dependency shim.

## Configuration and acquisition bindings

`rift/geraf_source.py` configures the source stage for the v1 paper: SDF width
256/eight layers/ten PE levels/scalar output, source learned position-only
reflectivity (`feats_dim=0`), and MF magnitude L2. The reflectivity source has
four linear layers, sigmoid output, weight normalization and source log power;
it is not the previous independent softplus network. Numerical constants,
sampling, masking before rendering, boundary correction, stale bank and the
source normal-Jacobian convention are preserved.

The supplied example is named `geraf2_bunnyboxv1_stage1.py`. It is not a
recovered historical v1 configuration. Parameters not established for v1 use
explicit released-example/class fallbacks, including the observed unadvanced
model-step counter and absolute LR floor. No silent bug fix or performance
selection is applied. `model_step_policy=advance` is a named alternate recipe.

`rift/geraf_source_ops.py` binds the source tracer/MF signatures to the actual
native measurement phase, frequencies, paired antennas and phase references.
It retains source sum-path spreading, specular gates, sample-count and
antenna-mean normalization. RIFT uses a numerically checked paired NUFFT;
GOTCHA uses exact native sums. Geometry/phase arithmetic is float64. The Torch
ray interpolation and specular Jacobian port follow the copied CUDA formulas;
bitwise CUDA equivalence is not established. No source radar hardware path bias,
frequency decimation or calibration is guessed for these datasets.

The full choice table, required acquisition differences, missing historical
settings, target-preparation fallbacks and validation limits are documented in
[docs/GERAF_V1_HARDENING.md](../../../docs/GERAF_V1_HARDENING.md).
The earlier `hardened_v1`/`legacy` implementations remain explicitly selectable
for compatibility and are not the source-fidelity comparison defaults.
