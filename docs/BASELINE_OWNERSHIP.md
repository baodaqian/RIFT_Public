# Baseline implementation ownership

The user initially separated RadarSplat and Sugavanam–Ertin frontends during
concurrent implementation, then explicitly requested consolidation after both
owners finished. The canonical dataset CLIs are now `train_rift_dataset.py` and
`train_gotcha_dataset.py`. The four method-specific dataset scripts were removed
at the user's request after merge checks. HH is the default and intended GOTCHA
comparison channel.

| Responsibility | RadarSplat | Sugavanam–Ertin |
| --- | --- | --- |
| Collection command/recipe module | `rift/radarsplat_collection.py` | `rift/sugavanam_ertin_collection.py` |
| Collection recipe tests | `tests/test_radarsplat_collection.py` | `tests/test_sugavanam_ertin_collection.py` |
| Maintained model trainer | `train_radarsplat.py` | `train_sugavanam_ertin.py` |
| GOTCHA backend | `rift/radarsplat_gotcha.py` | `rift/sugavanam_ertin_paper_workflow.py` |
| Fidelity/adaptation ledger | `docs/RADARSPLAT_FIDELITY.md` | `docs/SUGAVANAM_ERTIN_PAPER.md` |

Model, recipe, conversion and checkpoint implementations remain separate. The
common CLIs delegate to these modules; merging frontends does not justify
embedding model budgets in the dispatcher or altering scientific choices.
Shared GOTCHA detailed planning lives in `rift/gotcha_baseline_planning.py` and
imports only the selected owner's implementation. Capability discovery still
uses the maintained trainers' literal `GOTCHA_BACKEND` declarations.

The two methods can be selected together through either dataset CLI. Sealed
public loaders, object/source identities and method-specific resume rules are
preserved. RIFT retains RadarSplat's dependency check before target preparation;
both datasets expose SE's bounded initialization diagnostic. All output paths
are checked before starting the first selected method/object.

Integration tests now cover the canonical interfaces, joint dispatch and error
propagation. Model tests and synthetic fixtures remain in their owner files;
they do not require merging the two numerical implementations. Coordinate future
shared API changes in `temp_crosstalk.md`.

See [dataset frontend consolidation](DATASET_FRONTEND_CONSOLIDATION.md) for the
removed-script mapping, config/resume interfaces, validation and remaining
baseline limitations. Real fitting remains experiment-manager-owned.
