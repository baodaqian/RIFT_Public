import unittest
import importlib.util
from pathlib import Path
import sys

import numpy as np

_HELPER_PATH = Path(__file__).resolve().parents[1] / "rift" / "public_radar_tuning.py"
_SPEC = importlib.util.spec_from_file_location("rift_public_radar_tuning_pure", _HELPER_PATH)
_HELPERS = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _HELPERS
_SPEC.loader.exec_module(_HELPERS)

ComplexAccumulators = _HELPERS.ComplexAccumulators
accumulation_windows = _HELPERS.accumulation_windows
angular_coverage_hole_deg = _HELPERS.angular_coverage_hole_deg
closed_form_gain = _HELPERS.closed_form_gain
epoch_group_order = _HELPERS.epoch_group_order
interleaved_group_split = _HELPERS.interleaved_group_split
score_gain = _HELPERS.score_gain
select_balanced_direction_fps = _HELPERS.select_balanced_direction_fps
select_group_stratified_fps = _HELPERS.select_group_stratified_fps
validate_embedded_partition = _HELPERS.validate_embedded_partition


class PublicRadarTuningTests(unittest.TestCase):
    def test_interleaved_split_is_group_safe_uniform_and_deterministic(self):
        # Deliberately variable group sizes exercise the canonical Camry edge.
        sizes = np.resize(np.array([28, 32, 36]), 20)
        groups = np.repeat(np.arange(20), sizes)
        azimuth = np.concatenate(
            [np.linspace(g * 18.0, (g + 1) * 18.0, size, endpoint=False)
             for g, size in enumerate(sizes)]
        )
        roles, summary = interleaved_group_split(groups, azimuth, seed=42)
        replay, replay_summary = interleaved_group_split(groups, azimuth, seed=42)
        for role in ("train", "validation", "test"):
            np.testing.assert_array_equal(roles[role], replay[role])
        self.assertEqual(summary, replay_summary)
        self.assertEqual(summary["groups_by_role"], {
            "train": 16, "validation": 2, "test": 2
        })
        self.assertEqual(summary["groups"]["validation"], [0, 10])
        self.assertEqual(summary["groups"]["test"], [5, 15])
        validate_embedded_partition(
            groups.size,
            roles["train"],
            roles["validation"],
            roles["test"],
            groups,
        )
        for role, indices in roles.items():
            ordered_groups = groups[indices]
            for group in np.unique(ordered_groups):
                locations = np.flatnonzero(ordered_groups == group)
                self.assertEqual(locations[-1] - locations[0] + 1, locations.size)

    def test_interleaved_gotcha_variable_group_counts_are_exact(self):
        validation_role_groups = list(range(0, 360, 10))
        test_role_groups = list(range(5, 360, 10))
        validation_groups = set(validation_role_groups)
        test_groups = set(test_role_groups)
        train_groups = [
            group for group in range(360)
            if group not in validation_groups and group not in test_groups
        ]
        # 304 groups have 118 views; allocate the extras so the sealed role
        # counts match the authoritative pass-2 acquisition exactly.
        extra_groups = (
            set(validation_role_groups[:30])
            | set(test_role_groups[:31])
            | set(train_groups[:243])
        )
        sizes = np.asarray(
            [118 if group in extra_groups else 117 for group in range(360)]
        )
        self.assertEqual(int(sizes.sum()), 42424)
        groups = np.repeat(np.arange(360), sizes)
        within = np.concatenate([np.arange(size) for size in sizes])
        azimuth = (groups + (within + 0.5) / sizes[groups]) % 360.0
        roles, summary = interleaved_group_split(groups, azimuth, seed=42)
        self.assertEqual(
            tuple(roles[role].size for role in ("train", "validation", "test")),
            (33939, 4242, 4243),
        )
        self.assertEqual(
            tuple(
                summary["groups_by_role"][role]
                for role in ("train", "validation", "test")
            ),
            (288, 36, 36),
        )

    def test_balanced_direction_fps_supports_multiple_group_rounds(self):
        azimuth = np.arange(0.0, 360.0, 10.0)
        points = []
        groups = []
        for group, az in enumerate(azimuth):
            for elevation in (30.0, 40.0, 50.0, 60.0):
                az_r, el_r = np.radians([az, elevation])
                points.append([
                    np.cos(el_r) * np.cos(az_r),
                    np.cos(el_r) * np.sin(az_r),
                    np.sin(el_r),
                ])
                groups.append(group)
        points = np.asarray(points)
        groups = np.asarray(groups)
        train = np.arange(points.shape[0])
        selected = select_balanced_direction_fps(train, points, groups, count=100)
        replay = select_balanced_direction_fps(train, points, groups, count=100)
        np.testing.assert_array_equal(selected, replay)
        _, counts = np.unique(groups[selected], return_counts=True)
        self.assertEqual((int(counts.min()), int(counts.max())), (2, 3))
        self.assertEqual(np.unique(selected).size, 100)
        self.assertLess(
            angular_coverage_hole_deg(points, points[selected]),
            angular_coverage_hole_deg(points, points[train[:100]]),
        )

    def test_embedded_partition_rejects_group_leakage(self):
        summary = validate_embedded_partition(
            6,
            np.array([0, 1, 2, 3]),
            np.array([4, 5]),
            np.array([], dtype=np.int64),
            np.array([0, 0, 1, 1, 2, 2]),
        )
        self.assertEqual(summary["train"], 4)
        with self.assertRaisesRegex(ValueError, "leaks"):
            validate_embedded_partition(
                6,
                np.array([0, 1, 2]),
                np.array([3, 4, 5]),
                np.array([], dtype=np.int64),
                np.array([0, 0, 1, 1, 2, 2]),
            )

    def test_group_epoch_order_is_contiguous_and_resume_exact(self):
        groups = np.repeat(np.arange(8), [3, 2, 4, 1, 3, 2, 5, 4])
        first = epoch_group_order(groups, seed=42, epoch=5)
        replay = epoch_group_order(groups, seed=42, epoch=5)
        next_epoch = epoch_group_order(groups, seed=42, epoch=6)
        np.testing.assert_array_equal(first, replay)
        self.assertFalse(np.array_equal(first, next_epoch))
        np.testing.assert_array_equal(np.sort(first), np.arange(groups.size))
        ordered_groups = groups[first]
        for group in np.unique(groups):
            locations = np.flatnonzero(ordered_groups == group)
            self.assertEqual(locations[-1] - locations[0] + 1, locations.size)

    def test_exact_1800_update_budget(self):
        order = np.arange(20736)
        windows = accumulation_windows(order, 1800)
        sizes = np.asarray([window.size for window in windows])
        self.assertEqual(len(windows), 1800)
        self.assertEqual(int((sizes == 12).sum()), 936)
        self.assertEqual(int((sizes == 11).sum()), 864)
        np.testing.assert_array_equal(np.concatenate(windows), order)

    def test_stratified_fps_is_deterministic_and_improves_coverage(self):
        # Four elevation rings; each angular sector is one acquisition group.
        azimuth = np.arange(0.0, 360.0, 10.0)
        elevation = np.array([30.0, 40.0, 50.0, 60.0])
        points = []
        elevations = []
        groups = []
        for group, az in enumerate(azimuth):
            for el in elevation:
                az_r, el_r = np.radians([az, el])
                points.append(
                    [
                        np.cos(el_r) * np.cos(az_r),
                        np.cos(el_r) * np.sin(az_r),
                        np.sin(el_r),
                    ]
                )
                elevations.append(el)
                groups.append(group)
        points = np.asarray(points)
        elevations = np.asarray(elevations)
        groups = np.asarray(groups)
        train = np.arange(points.shape[0])
        selected = select_group_stratified_fps(
            train, points, groups, elevations, count=32
        )
        replay = select_group_stratified_fps(
            train, points, groups, elevations, count=32
        )
        np.testing.assert_array_equal(selected, replay)
        self.assertEqual(np.unique(groups[selected]).size, 32)
        for el in elevation:
            self.assertEqual(int(np.isclose(elevations[selected], el).sum()), 8)
        ordered = train[:32]
        self.assertLess(
            angular_coverage_hole_deg(points, points[selected]),
            angular_coverage_hole_deg(points, points[ordered]),
        )

    def test_closed_form_gain_and_scoring(self):
        rng = np.random.default_rng(7)
        prediction = rng.normal(size=100) + 1j * rng.normal(size=100)
        truth_gain = 1.7 * np.exp(0.4j)
        measured = truth_gain * prediction
        accumulators = ComplexAccumulators(
            cross=complex(np.vdot(prediction, measured)),
            predicted_power=float(np.vdot(prediction, prediction).real),
            measured_power=float(np.vdot(measured, measured).real),
            sample_count=prediction.size,
        )
        fitted = closed_form_gain(accumulators)
        self.assertAlmostEqual(fitted.real, truth_gain.real, places=12)
        self.assertAlmostEqual(fitted.imag, truth_gain.imag, places=12)
        self.assertLess(score_gain(accumulators, fitted)["relative_mse"], 1.0e-14)
        self.assertAlmostEqual(
            score_gain(accumulators, 0.0j)["relative_mse"], 1.0, places=12
        )


if __name__ == "__main__":
    unittest.main()
