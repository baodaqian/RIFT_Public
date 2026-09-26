"""Versioned RF acquisition adaptations; no response access or Torch imports."""

from __future__ import annotations

import math


LEGACY_RECIPE = "legacy-v1"
AUDITED_RECIPE = "audited-v2"
SOURCE_RECIPE = "source-adapted-v3"
AUDITED_FIELDS = (
    "model_backend",
    "ray_samples", "occupancy_noise_multiplier", "occupancy_decay_bins",
    "occupancy_probability_offset", "occupancy_probability_scale",
)


# What source-adapted-v3 means: the released RadarField model, output mapping and
# occupancy recurrence (external/RadarFields_reference configs/radarfields.ini,
# radar.py; TCNN SH "degree" 4 = bands 0..3 = sh_degree 3), plus this profile's
# declared adaptation constants. The profile accepts no other value; engineering
# variants belong to audited-v2. Every campaign run used exactly these values.
SOURCE_LOCKS = dict(
    model_backend="upstream-tcnn", hidden_dim=64, feature_dim=32, sh_degree=3,
    sigmoid_tightness=1.0, no_batch_norm=False, hash_levels=16, hash_features=2,
    hash_base_resolution=16, hash_final_resolution=512, hash_log2_size=19,
    occupancy_noise_multiplier=1.5, occupancy_decay_bins=10.0,
    occupancy_probability_offset=-0.15, occupancy_probability_scale=2.0,
    intensity_offset=1.0, intensity_scaler=1.0,
    dynamic_range_db=60.0, range_margin=0.05, range_guard_cells=2,
)


def recipe_name(args):
    name = getattr(args, "recipe", LEGACY_RECIPE)
    if name not in (LEGACY_RECIPE, AUDITED_RECIPE, SOURCE_RECIPE):
        raise ValueError(f"unknown Radar Fields recipe {name!r}")
    return name


def native_recipe(args):
    return recipe_name(args) in (AUDITED_RECIPE, SOURCE_RECIPE)


def recipe_contract(args):
    name = recipe_name(args)
    if name == LEGACY_RECIPE:
        return {"schema": "rift_radar_fields_recipe_v1", "recipe": name}
    controls = {key: getattr(args, key) for key in AUDITED_FIELDS}
    if isinstance(controls["ray_samples"], bool) or int(controls["ray_samples"]) != controls["ray_samples"] or controls["ray_samples"] < 1:
        raise ValueError("ray_samples must be a positive integer")
    if controls["model_backend"] not in ("upstream-tcnn", "torch"):
        raise ValueError("model_backend must be upstream-tcnn or torch")
    for key, value in controls.items():
        if key == "model_backend":
            continue
        if not math.isfinite(float(value)):
            raise ValueError(f"{key} must be finite")
    for key in ("occupancy_noise_multiplier", "occupancy_decay_bins", "occupancy_probability_scale"):
        if controls[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if name == SOURCE_RECIPE:
        for key, value in SOURCE_LOCKS.items():
            if hasattr(args, key) and getattr(args, key) != value:
                raise ValueError(f"source-adapted-v3 fixes {key}={value!r}; use audited-v2 for engineering variants")
    result = {
        "schema": "rift_radar_fields_recipe_v2", "recipe": name,
        "encoding": ("original_radarfield_tcnn" if controls["model_backend"] == "upstream-tcnn"
                     else "torch_vertex_dense_or_coherentprime_physical_sh_v1"),
        "renderer": "bistatic_bin_center_ray_mean_v1",
        "angular_measure": "deterministic_uniform_solid_angle_scene_cap",
        "antenna_gain": "unit_within_scene_cap_no_measured_pattern",
        "exterior": "zero_outside_registered_cube_denominator_includes_all_rays",
        "occupancy": "per_pair_range_median_released_bayesian_recurrence_v1",
        "loss": "global_occupancy_distribution_batchmean_sample_std_0p01",
        "normalization": "fixed_train_peak_60db_or_declared_dynamic_range",
        "optimizer": "released_adam_0p9_0p99_eps1e-15",
        "grounding_and_height_priors": "disabled_in_object_acquisition",
        "pose_refinement": "disabled_calibrated_dataset_geometry",
        "view_sampling": "permuted_train_coverage_v1",
        **controls,
    }
    if name == SOURCE_RECIPE:
        result.update(schema="rift_radar_fields_recipe_v3",
            angular_measure="released_random_pitch_yaw_and_central_ray_over_scene_ROI",
            loss="released_KL_finite_part_at_fixed_exterior_zeros_and_sample_std_nan_to_num",
            optimizer="released_Adam_parameter_groups_0p9_0p99_eps1e-15",
            training_schedule="released_800_lr_clock_ceil_complete_epochs_epoch_sine_mask",
            view_sampling="torch_SubsetRandomSampler_permutation",
            profile_sampling="released_sorted_randint_with_replacement_100",
            fft_target="normalized_db_then_global_floor_0p1525_missing_preprocessing_script",
            source_iters=800)
    return result


def validate_recipe_checkpoint(checkpoint, args):
    """Reject a scientific recipe change before loading data, including v1."""
    current = recipe_contract(args)
    saved = checkpoint.get("radar_fields_recipe")
    saved_name = checkpoint.get("args", {}).get("recipe", LEGACY_RECIPE)
    if saved_name != current["recipe"]:
        raise ValueError("Radar Fields recipe mismatch; historical checkpoints require --recipe legacy-v1")
    if saved is None and current["recipe"] == LEGACY_RECIPE:
        return
    if saved != current:
        raise ValueError("Radar Fields recipe/renderer/occupancy configuration mismatch")
