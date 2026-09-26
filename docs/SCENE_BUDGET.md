# Selected scene budget

User-selected defaults, 2026-09-20, for both the RIFT collection and native
GOTCHA. These are disclosed discretization/capacity adaptations, not recovered
author settings or evidence of equal physical resolution. Existing acquisition,
roles, network architectures and objectives remain. The 2026-09-21 SpINR
comparison-budget override is 150 full training passes with a 150-epoch cosine
schedule on both datasets. This matches RIFT's passes, not measured compute.

| Method | Training scene / numerical lattice | Geometry readout |
| --- | --- | --- |
| RIFT | G48 initial anchors; adaptive capacity 262144, SH 0–3 | G48 support, no upsampling |
| SpINR | G48 midpoint: exactly 110592 integration points; same signed-real INR | G48 field samples |
| Radar Fields | Released continuous hash-grid network; collection compatibility grid G48 | G48 occupancy |
| GeRaF | G48 lazy MF target lattice; same continuous SDF and reflectivity networks | G48 zero-SDF extraction |
| RadarSplat | 112000 Gaussians (1.27% above 48³); source CUDA model, 2000 updates | G48 Gaussian occupancy-union proxy plus metric Gaussians |
| Sugavanam–Ertin | G40 Stage-1 grid per sub-aperture; same neural SDF | G48 zero-SDF extraction |

SE's G40 is the explicit exception. RadarSplat's acquisition/raster sampling
(33 azimuth/elevation samples and native range policy) is unchanged: it is not
the reconstructed Cartesian scene grid. Likewise, RF hash levels and GeRaF ray
sample counts retain their source settings. Native signal validation preserves
all declared measurements and its method-specific observable.

Collection `extent=0.15` is a half-width: the physical cube is 0.30 m wide.
G48 midpoint pitch is 6.25 mm, and SE G40 pitch is 7.5 mm. Native Camry remains
a 10 m cube, giving 208.33 mm / 250 mm pitches. SDF marching cubes samples the
box endpoints; support readouts use cell centers. Record actual coordinates and
fixed metric tolerances; a finer extraction alone does not establish resolution.

## Configuration and recovery

- Collection SpINR defaults to `--spinr-recipe budget48-direct`, using a new
  direct-bin/G48-midpoint/150-epoch identity and `spinr/budget48-direct-150/`
  output. `budget48-direct-1500` preserves the old 1500-epoch identity and
  `spinr/budget48-direct/` output. `paper-v1-direct`
  explicitly retains G96/GL2; `paper-v1` and `legacy-midpoint` remain distinct.
- Native SpINR defaults to G48/order 1. The explicit configuration is
  [gotcha_spinr_g48_midpoint.json](../protocols/gotcha_spinr_g48_midpoint.json).
  It selects 150 epochs and matching cosine schedule. Historical midpoint
  runs use [gotcha_spinr_g48_midpoint_1500.json](../protocols/gotcha_spinr_g48_midpoint_1500.json).
  Prior G48/GL2 runs use [gotcha_spinr_g48.json](../protocols/gotcha_spinr_g48.json);
  prior G96/GL2 runs must specify both grid 96 and order 2.
- GeRaF defaults to `mf_grid=48`; use [geraf_mf48.json](../protocols/geraf_mf48.json)
  or [geraf_mf48_1t1r.json](../protocols/geraf_mf48_1t1r.json). One float32 NPY
  accumulated grid/head is 442496 bytes including its header; no per-view cubes.
  Explicit MF101 protocols remain available for old identities.
- Collection RadarSplat defaults to `--radarsplat-recipe budget48`; native
  GOTCHA uses `fidelity_profile=budget48`. Direct released CLI:
  `--fidelity-profile budget48`. `upstream` explicitly retains 20000 Gaussians.
  [radarsplat_budget48.json](../protocols/radarsplat_budget48.json) is a native
  `--config`. The pinned upstream manifest/source remain unchanged. New runs
  export `geometry_g48.npz` from the final checkpoint; geometry readout exports
  also include G48 support using a disclosed 3-sigma occupancy-union proxy.
- SE defaults to G40/export G48 and user-selected Gaussian initialization std 0.05, recorded by
  [se_g40_readout48.json](../protocols/se_g40_readout48.json). Prior recipes need
  explicit original `granularity` (or 0 for the range-based rule) and
  `export_grid=96` and their original `initialization_std`; the complete recipe
  must match before response access. Explicit std 1 restores the literal
  initialization identity; see [the SE fidelity ledger](SUGAVANAM_ERTIN_PAPER.md).
- Native RIFT now starts at G48; explicit `--granularity 32` retains G32.

Use fresh output roots for changed recipes. Checkpoint identity gates reject
cross-budget recovery; do not relabel old caches or checkpoints. Geometry uses
the same selected checkpoint as signal scoring. General readouts default to G48;
frozen historical B787 reports keep their original G48→G192 interpolation.

## Qualification boundaries

SpINR midpoint integration on the 10 m native scene remains unqualified. SE's
selected std-0.05 initialization passes the bounded CPU probe on both datasets,
with 256/256 nonzero-gradient points and no saturation. Fitting/convergence remain
unqualified; initialization/convergence gates remain.
Increased Gaussian count needs allocated memory/runtime checks.
The larger SE collection grid and changed SpINR/GeRaF grids invalidate earlier
resource tables for those configurations; the guides mark those estimates stale.
Native GOTCHA SpINR/RF geometry interfaces remain TODOs; this change does not
claim to implement missing evaluators. No production preparation, training,
scheduler action, reserved-test access or publication is authorized by this file.
