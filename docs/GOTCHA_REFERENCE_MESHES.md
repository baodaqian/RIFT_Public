# GOTCHA reference meshes: sources and citations

The GOTCHA Volumetric SAR Data Set supplies no 3D ground truth for its vehicles. Each GOTCHA target's 3D reference is
a **same-generation stand-in model** from a public 3D repository, identified by the user, scaled per axis to the
published specification of that generation, and registered to the vehicle's position in the data. None is a scan of
the GOTCHA vehicle. **All three models must be cited in the paper** (user, 2026-09-24).

| GOTCHA target | Stand-in model | Author | Repository | Licence | Generation / spec used |
|---|---|---|---|---|---|
| Toyota Camry (target "B"; `camry_box_v2`) | "Toyota Camry (Mk4)(XV20) 1997" | Nieve5677 | Sketchfab, <https://sketchfab.com/3d-models/toyota-camry-mk4xv20-1997-196829bf4cbc4b8cb26c50964f35ff27> | CC Attribution 4.0 | XV20 (1997–2001): 4.765 × 1.785 × 1.430 m |
| Hyundai Santa Fe (C4; `santafe_box_v1`) | "2004 Hyundai Santa- Fe" | teenlin3 | Sketchfab, <https://sketchfab.com/3d-models/b33c872e79b24508a1b9213e3537651c> | CC Attribution 4.0 | First generation SM (2000–2006; 2004 facelift): 4.500 × 1.845 × 1.730 m |
| Nissan Sentra (C2; `sentra_box_v1`) | "Nissan Sentra 2005" | Lone Wolf | 3D Warehouse (Trimble), <https://3dwarehouse.sketchup.com/model/e4a15cca51107dd035da01f298003d56/Nissan-Sentra-2005> | 3D Warehouse General Model License | B15 (2000–2006): 4.508 × 1.709 × 1.410 m |

Specifications (length × width without mirrors × height) are from the Wikipedia generation infoboxes
(Toyota Camry (XV20); Hyundai Santa Fe, first generation; Nissan Sentra, fifth generation B15). The Santa Fe height uses
1,730 mm (Wikipedia lists 1,675 mm, and 1,730 mm for China) because the model carries roof rails.

## How each mesh is used

- **Camry:** `scripts_pvc/prepare_gotcha_camry_mesh.py`. Registered to the four footprint corners of the GOTCHA
  ground-truth workbook, then moved by the calibration array's documentation-to-data offset (the `data_frame` variant,
  the user's choice on 2026-09-23). Outputs: `data/meshes/camry_xv20_data_frame*`.
- **Santa Fe and Sentra:** `scripts_pvc/prepare_gotcha_target_mesh.py`. There is no workbook footprint for these cars
  (the workbook is not available), so each is centred on its TRAIN-only radar locate estimate with the estimated
  heading (`scripts_pvc/gotcha_target_locate.py`; the same estimator is 0.28 m and 4° from the Camry's registered box).
  Wheels are on the lot's ground height. Both front/back orientations are written and a data check chooses one.
  The Sentra arrived as a Collada export and was flattened with `scripts_pvc/gotcha_target_dae_to_obj.py`.
  Outputs: `RUN_ROOT/gotcha_new_targets_20260924/meshes/{santafe_2004,sentra_b15}_box_v1/`.
- Vehicle positions for the new targets start from Casteel et al., "A challenge problem for 2D/3D imaging of targets from
  a volumetric data set in an urban environment", Proc. SPIE 6568, 65680D (2007), Fig. 1. The vehicle identities
  (Nissan Sentra, Hyundai Santa Fe) are the user's.

## Suggested BibTeX (verify fields before use)

```bibtex
@misc{nieve5677_camry_xv20,
  author       = {Nieve5677},
  title        = {Toyota Camry (Mk4)(XV20) 1997},
  howpublished = {Sketchfab, \url{https://sketchfab.com/3d-models/toyota-camry-mk4xv20-1997-196829bf4cbc4b8cb26c50964f35ff27}},
  note         = {3D model, CC BY 4.0},
}
@misc{teenlin3_santafe_2004,
  author       = {teenlin3},
  title        = {2004 Hyundai Santa- Fe},
  howpublished = {Sketchfab, \url{https://sketchfab.com/3d-models/b33c872e79b24508a1b9213e3537651c}},
  note         = {3D model, CC BY 4.0},
}
@misc{lonewolf_sentra_2005,
  author       = {{Lone Wolf}},
  title        = {Nissan Sentra 2005},
  howpublished = {3D Warehouse, \url{https://3dwarehouse.sketchup.com/model/e4a15cca51107dd035da01f298003d56/Nissan-Sentra-2005}},
  note         = {3D model, 3D Warehouse General Model License},
}
@inproceedings{casteel2007gotcha,
  author    = {Casteel, Jr., Curtis H. and Gorham, LeRoy A. and Minardi, Michael J. and Scarborough, Steven M. and Naidu, Kiranmai D. and Majumder, Uttam K.},
  title     = {A challenge problem for {2D/3D} imaging of targets from a volumetric data set in an urban environment},
  booktitle = {Algorithms for Synthetic Aperture Radar Imagery XIV},
  series    = {Proc. SPIE},
  volume    = {6568},
  pages     = {65680D},
  year      = {2007},
}
```

The Casteel et al. author list is checked against the paper's first page (Curtis H. Casteel, Jr; LeRoy A. Gorham;
Michael J. Minardi; Steven M. Scarborough; Kiranmai D. Naidu; Uttam K. Majumder; AFRL Sensors Directorate). A copy is at
`RUN_ROOT/gotcha_new_targets_20260924/Casteel2007_GOTCHA_challenge_problem.pdf`.

Record: docs/RIFT_GOTCHA_Tune.md A75–A77.
