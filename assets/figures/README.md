# README figures

These are exports of Figures 1–4, 10, and 13 from the September 2026 ICLR 2027
submission, *Radon Implicit Field Transform: Fully Radar-Native Novel-View
Synthesis and 3D Reconstruction*. Figure numbering is taken from the final
local manuscript's cross-reference record.

| Figure | Published image | Manuscript source |
| --- | --- | --- |
| 1 | `figure_01_rift_overview.png` | `figures/schematic/rift_schematic.pdf` |
| 2 | `figure_02_simulated_performance.png` | `figures/rift_dataset/rift_dataset_performance_scatter.pdf` |
| 3 | `figure_03_gotcha_performance.png` | `figures/gotcha/gotcha_performance_scatter.pdf` |
| 4 | `figure_04_gotcha_pipeline.png` | `figures/gotcha/gotcha_rift_pipeline.pdf` |
| 10 | `figure_10_simulated_reconstructions.png` | Compiled manuscript, page 31; label `fig:homemade-rift-view-3` |
| 13 | `figure_13_gotcha_reconstructions.png` | Compiled manuscript, page 36; label `fig:gotcha-camry-view-3` |

The original source paths are relative to `manuscripts/iclr27/` in the RIFT
development repository. Figures 10 and 13 use the 36-page build
`check_20260925_2223_abstract`, completed on 25 September 2026 at 22:30 CDT.
Their layout comes from `sec/12_dataset_visualizations_appendix.tex`.

The PDFs were rendered to PNG at 432 dpi with Poppler 23.09.0. Figures 1–4 use
their full standalone PDF bounds. Figures 10 and 13 use the figure regions of
the compiled pages, omitting the manuscript header, line numbers, page number,
and caption; captions are provided in the root README. Crop rectangles in PDF
points, measured from the page's upper-left corner, are `(136, 78, 469, 502)`
for Figure 10 and `(104, 78, 501, 369)` for Figure 13. No scientific panels,
method rows, labels, or color bars were removed or regenerated.

The Figure 4 parking-lot photograph is from Casteel et al., *A Challenge
Problem for 2D/3D Imaging of Targets from a Volumetric Data Set in an Urban
Environment*, Proc. SPIE 6568, 65680D (2007), Figure 2,
[doi:10.1117/12.731457](https://doi.org/10.1117/12.731457).

The Figure 13 reference meshes are by Nieve5677 (Toyota Camry, CC BY 4.0),
Lone Wolf (Nissan Sentra, 3D Warehouse General Model License), and teenlin3
(Hyundai Santa Fe, CC BY 4.0). They were scaled and registered for the paper;
the figure shows their rendered views and lattice support. Full model links,
attribution, and registration details are in
[`docs/GOTCHA_REFERENCE_MESHES.md`](../../docs/GOTCHA_REFERENCE_MESHES.md).
The source meshes are not included in this repository.
