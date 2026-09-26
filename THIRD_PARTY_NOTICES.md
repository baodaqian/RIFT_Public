# Third-party sources

This source snapshot retains upstream authorship, notices, and license files.
The entries below identify the incorporated components; they do not replace
their terms or grant additional rights. No new project-wide license is applied.

| Component | Upstream / revision | Included location and terms |
| --- | --- | --- |
| GeRaF-SENS | [VictorLlu/GeRaF-SENS](https://github.com/VictorLlu/GeRaF-SENS), `38266cb6e194e2f3dcbead614069a7281ffd21a5` | `rift/vendor/geraf_sens/`, `rift_pvc/vendor/geraf_sens/`; PolyForm Noncommercial 1.0.0, with adjacent `LICENSE` and `NOTICE.md` files |
| Radar Fields | [princeton-computational-imaging/RadarFields](https://github.com/princeton-computational-imaging/RadarFields), `ee76d76570f58b3d8539eafd7df0c188b58af333` | `external/RadarFields_reference/`; the copied upstream revision contains no separate license file; no additional permissions are asserted here |
| RadarSplat | [umautobots/radarsplat](https://github.com/umautobots/radarsplat), `ea9c8f530c708622cc3b1b560436b5557ac6a49b` | `external/radarsplat_reference/ea9c8f530c708622cc3b1b560436b5557ac6a49b/`; upstream `LICENSE` and nested component notices are retained |
| GLM and the RadarSplat gsplat fork | Revisions recorded by RadarSplat and `SOURCE_PROVENANCE.json` | Retained within the RadarSplat reference tree, including their license/copyright notices |
| SPGL1 Python port | [drrelyea/spgl1](https://github.com/drrelyea/spgl1), `405ca805a2d56d783a0445e834c801c5b7c2263a` | `rift_pvc/vendor/spgl1/`; LGPL 2.1, with adjacent `LICENSE` and `NOTICE.md` |
| FRTM reader and PyArgus | Copied from the working RIFT tree | `external/read_frtm/`; original author headers retained, including PyArgus's GPLv3 notices; no separate upstream commit was recorded |

SpINR, Sugavanam–Ertin, and SH-SAS implementations in this tree are described as
independent implementations in their source and adaptation documentation, not
as the respective authors' released code. The PVC Radar Fields and RadarSplat
backends are adaptations with separate backend identities. Original CUDA
reference files are included for provenance and shared code, not as a claim
that CUDA kernels execute on PVC.

Historical source-hash inventories are preserved as documentation. This
publication does not calculate or enforce new source-hash checks.
