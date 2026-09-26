"""Metadata planning shared by the merged RadarSplat/SE GOTCHA frontends.

Model recipes and execution remain in their respective backends. Imports are
lazy so selecting one baseline does not import the other baseline's engine.
"""
from pathlib import Path


METHODS = ("radarsplat", "sugavanam_ertin")


def validate_config(method, config):
    """Reject invalid owner options before opening the native dataset."""
    if method == "radarsplat":
        from .radarsplat_gotcha import adapter_config
        return dict(adapter_config(config), fidelity_profile=config.get("fidelity_profile", "budget48"))
    if method == "sugavanam_ertin":
        from .sugavanam_ertin_paper_workflow import make_recipe
        make_recipe("gotcha_native", config)
        return dict(config)
    raise ValueError(f"No merged baseline planner for {method!r}")


def make_plan(method, dataset, output_dir, config):
    """Preserve both former frontends' detailed metadata-only reports."""
    output = Path(output_dir)
    if method == "radarsplat":
        from .radarsplat_gotcha import CONTROL_FILE, GOTCHAPowerCache, planning
        caches = {pol: GOTCHAPowerCache(dataset, pol, output/pol/"targets", config)
                  for pol in dataset.polarizations}
        return dict(recipe=planning(dataset, config), resume_file=str(output/CONTROL_FILE),
            targets_per_head={pol: dict(train=len(cache.train_indices),
                validation=len(cache.validation_indices), range_bins=cache.grid["n_range"],
                azimuth_bins=cache.grid["n_azimuth"], elevation_bins=cache.grid["n_elevation"])
                for pol, cache in caches.items()},
            cuda_execution_validated=False, response_payload_read=False)
    if method == "sugavanam_ertin":
        from .sugavanam_ertin_acquisition import GOTCHAAcquisition
        from .sugavanam_ertin_paper_workflow import make_recipe, plan
        _, _, report = plan(GOTCHAAcquisition(dataset), make_recipe("gotcha_native", config))
        return dict(se=report, fidelity_status="published_initialization_unresolved_not_benchmark_ready")
    raise ValueError(f"No merged baseline planner for {method!r}")


def check_initialization(dataset, entry):
    """SE's existing bounded CPU diagnostic; no responses, fitting or writes."""
    if entry["method"] != "sugavanam_ertin":
        raise ValueError("Initialization diagnostic requires Sugavanam–Ertin")
    from .sugavanam_ertin_paper import PaperSDF, initialization_audit
    from .sugavanam_ertin_paper_workflow import _model_config
    recipe = entry["se"]["recipe"]
    model = PaperSDF(**_model_config(recipe, dataset.region.half_extent_m))
    return initialization_audit(model, dataset.region.half_extent_m, seed=recipe["seed"])
