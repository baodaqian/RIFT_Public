import math
import unittest

import torch

from rift.gotcha_domain import (
    GOTCHA_DOMAIN_SCHEMA,
    SPEED_OF_LIGHT_M_S,
    gotcha_full_domain_metadata,
    gotcha_nonwrapping_chunk_mask,
    validate_gotcha_frequency_grid,
    validate_gotcha_full_domain,
    validate_gotcha_planar_square_views,
)


class GotchaDomainTests(unittest.TestCase):
    REFERENCE_RANGE_M = 60.0
    UNAMBIGUOUS_RANGE_M = 110.5
    FREQUENCY_STEP_HZ = SPEED_OF_LIGHT_M_S / (2.0 * UNAMBIGUOUS_RANGE_M)

    def mask(self, positions, viewpoints):
        return gotcha_nonwrapping_chunk_mask(
            torch.as_tensor(positions, dtype=torch.float64),
            torch.as_tensor(viewpoints, dtype=torch.float64),
            reference_range_m=self.REFERENCE_RANGE_M,
            frequency_step_hz=self.FREQUENCY_STEP_HZ,
        )

    def test_metadata_freezes_square_plane_and_validates_core_window(self):
        metadata = gotcha_full_domain_metadata(
            reference_range_m=self.REFERENCE_RANGE_M,
            frequency_step_hz=self.FREQUENCY_STEP_HZ,
        )
        self.assertEqual(metadata["schema"], GOTCHA_DOMAIN_SCHEMA)
        self.assertEqual(metadata["x_bounds_m"], [-50.0, 50.0])
        self.assertEqual(metadata["y_bounds_m"], [-50.0, 50.0])
        self.assertEqual(metadata["z_reference_m"], 0.0)
        self.assertEqual(metadata["scene_center_m"], [0.0, 0.0, 0.0])
        self.assertEqual(metadata["core_radius_m"], 50.0)
        self.assertAlmostEqual(
            metadata["unambiguous_one_way_range_m"],
            self.UNAMBIGUOUS_RANGE_M,
        )
        self.assertGreater(self.REFERENCE_RANGE_M - 50.0, 0.0)
        self.assertLess(
            self.REFERENCE_RANGE_M + 50.0,
            metadata["unambiguous_one_way_range_m"],
        )

    def test_entire_core_is_valid_for_every_view(self):
        positions = [
            [0.0, 0.0, 0.0],
            [30.0, 40.0, 0.0],
            [-30.0, 40.0, 0.0],
            [50.0, 0.0, 0.0],
            [-50.0, 0.0, 0.0],
        ]
        viewpoints = [
            [100.0, 0.0, 0.0],
            [-100.0, 0.0, 0.0],
            [0.0, 100.0, 20.0],
            [80.0, -60.0, 30.0],
        ]
        mask = self.mask(positions, viewpoints)
        self.assertEqual(tuple(mask.shape), (len(viewpoints), len(positions)))
        self.assertTrue(bool(mask.all().item()))

    def test_square_corner_validity_depends_on_view(self):
        corner = [[50.0, 50.0, 0.0]]
        viewpoints = [
            [1000.0, 0.0, 0.0],
            [-1000.0, 0.0, 0.0],
        ]
        mask = self.mask(corner, viewpoints)
        self.assertEqual(mask[:, 0].tolist(), [True, False])

    def test_planar_square_preflight_uses_exact_distance_extrema(self):
        viewpoints = torch.tensor(
            [[1000.0, 0.0, 20.0], [-1000.0, 0.0, 20.0]],
            dtype=torch.float64,
        )
        summary = validate_gotcha_planar_square_views(
            viewpoints,
            reference_range_m=self.REFERENCE_RANGE_M,
            frequency_step_hz=SPEED_OF_LIGHT_M_S / (2.0 * 130.0),
        )
        self.assertEqual(summary["view_count"], 2)
        self.assertGreater(summary["minimum_lower_window_margin_m"], 0.0)
        self.assertGreater(summary["minimum_upper_window_margin_m"], 0.0)
        self.assertEqual(
            summary["mask_realization"],
            "identity_after_exact_planar_extrema_preflight",
        )

        with self.assertRaisesRegex(ValueError, "not wholly"):
            validate_gotcha_planar_square_views(
                viewpoints,
                reference_range_m=50.1,
                frequency_step_hz=SPEED_OF_LIGHT_M_S / (2.0 * 101.0),
            )

    def test_range_window_boundaries_are_strict(self):
        position = torch.tensor([[50.0, 50.0, 0.0]], dtype=torch.float64)
        viewpoint = torch.tensor([100.0, 100.0, 0.0], dtype=torch.float64)
        delta = (
            torch.linalg.vector_norm(position[0] - viewpoint)
            - torch.linalg.vector_norm(viewpoint)
        ).item()
        reference_range_m = -delta
        unambiguous_range_m = reference_range_m + 60.0
        frequency_step_hz = SPEED_OF_LIGHT_M_S / (2.0 * unambiguous_range_m)
        at_lower_boundary = gotcha_nonwrapping_chunk_mask(
            position,
            viewpoint,
            reference_range_m=reference_range_m,
            frequency_step_hz=frequency_step_hz,
        )
        just_inside = gotcha_nonwrapping_chunk_mask(
            position,
            viewpoint,
            reference_range_m=reference_range_m + 1.0e-12,
            frequency_step_hz=frequency_step_hz,
        )
        self.assertFalse(bool(at_lower_boundary.item()))
        self.assertTrue(bool(just_inside.item()))

        with self.assertRaisesRegex(ValueError, "lower open"):
            validate_gotcha_full_domain(
                reference_range_m=50.0,
                frequency_step_hz=SPEED_OF_LIGHT_M_S / 300.0,
            )
        with self.assertRaisesRegex(ValueError, "upper open"):
            validate_gotcha_full_domain(
                reference_range_m=60.0,
                frequency_step_hz=SPEED_OF_LIGHT_M_S / (2.0 * 110.0),
            )

    def test_invalid_shapes_values_and_domain_points_are_rejected(self):
        valid_positions = torch.zeros((2, 3), dtype=torch.float64)
        valid_viewpoint = torch.tensor([100.0, 0.0, 0.0], dtype=torch.float64)
        kwargs = {
            "reference_range_m": self.REFERENCE_RANGE_M,
            "frequency_step_hz": self.FREQUENCY_STEP_HZ,
        }
        with self.assertRaisesRegex(ValueError, r"shape \[N, 3\]"):
            gotcha_nonwrapping_chunk_mask(
                torch.zeros(3, dtype=torch.float64), valid_viewpoint, **kwargs
            )
        with self.assertRaisesRegex(ValueError, r"shape \[3\] or \[V, 3\]"):
            gotcha_nonwrapping_chunk_mask(
                valid_positions, torch.zeros((2, 2), dtype=torch.float64), **kwargs
            )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            bad = valid_positions.clone()
            bad[0, 0] = torch.nan
            gotcha_nonwrapping_chunk_mask(bad, valid_viewpoint, **kwargs)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            gotcha_nonwrapping_chunk_mask(
                valid_positions,
                torch.tensor([math.inf, 0.0, 0.0], dtype=torch.float64),
                **kwargs,
            )
        with self.assertRaisesRegex(ValueError, "outside GOTCHA"):
            gotcha_nonwrapping_chunk_mask(
                torch.tensor([[50.01, 0.0, 0.0]], dtype=torch.float64),
                valid_viewpoint,
                **kwargs,
            )
        with self.assertRaisesRegex(ValueError, "z=0"):
            gotcha_nonwrapping_chunk_mask(
                torch.tensor([[0.0, 0.0, 0.01]], dtype=torch.float64),
                valid_viewpoint,
                **kwargs,
            )
        with self.assertRaisesRegex(ValueError, "exactly"):
            gotcha_nonwrapping_chunk_mask(
                valid_positions,
                valid_viewpoint,
                scene_center_m=(1.0, 0.0, 0.0),
                **kwargs,
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_gotcha_full_domain(
                reference_range_m=math.nan,
                frequency_step_hz=self.FREQUENCY_STEP_HZ,
            )
        with self.assertRaisesRegex(ValueError, "positive"):
            validate_gotcha_full_domain(
                reference_range_m=self.REFERENCE_RANGE_M,
                frequency_step_hz=0.0,
            )

    def test_float32_source_frequency_jitter_is_magnitude_derived(self):
        nominal_spacing_hz = 1_464_320.0
        frequencies_hz = torch.tensor(
            [
                9_600_000_000.0,
                9_601_464_320.0,
                9_602_929_664.0,
                9_604_393_984.0,
            ],
            dtype=torch.float64,
        )
        nominal_unambiguous_range_m = SPEED_OF_LIGHT_M_S / (
            2.0 * nominal_spacing_hz
        )
        summary = validate_gotcha_frequency_grid(
            frequencies_hz,
            nominal_spacing_hz=nominal_spacing_hz,
            unambiguous_range_m=nominal_unambiguous_range_m,
        )
        self.assertEqual(summary["frequency_count"], 4)
        self.assertEqual(summary["nominal_spacing_hz"], 1_464_320.0)
        self.assertEqual(summary["min_spacing_hz"], 1_464_320.0)
        self.assertEqual(summary["max_spacing_hz"], 1_465_344.0)
        self.assertEqual(summary["max_abs_spacing_deviation_hz"], 1_024.0)
        self.assertEqual(summary["float32_source_step_tolerance_hz"], 1_024.0)
        self.assertEqual(summary["conservative_spacing_hz"], 1_465_344.0)
        self.assertLess(
            summary["endpoint_affine_residual_fraction"],
            summary["endpoint_affine_tolerance_fraction"],
        )
        self.assertAlmostEqual(
            summary["nominal_unambiguous_range_m"],
            nominal_unambiguous_range_m,
        )
        self.assertAlmostEqual(
            summary["conservative_unambiguous_range_m"],
            SPEED_OF_LIGHT_M_S / (2.0 * 1_465_344.0),
        )

    def test_frequency_jitter_bound_scales_with_frequency_magnitude(self):
        low_frequency_grid = torch.tensor(
            [1_000_000.0, 1_000_100.0625, 1_000_200.0625],
            dtype=torch.float64,
        )
        summary = validate_gotcha_frequency_grid(
            low_frequency_grid,
            nominal_spacing_hz=100.0,
        )
        self.assertEqual(summary["float32_source_step_tolerance_hz"], 0.0625)
        self.assertEqual(summary["max_abs_spacing_deviation_hz"], 0.0625)

        with self.assertRaisesRegex(ValueError, "source-quantization bound"):
            validate_gotcha_frequency_grid(
                torch.tensor(
                    [1_000_000.0, 1_000_100.125, 1_000_200.125],
                    dtype=torch.float64,
                ),
                nominal_spacing_hz=100.0,
            )

    def test_invalid_frequency_grids_and_metadata_are_rejected(self):
        valid = torch.tensor(
            [9.0e9, 9.001e9], dtype=torch.float32
        ).to(dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            validate_gotcha_frequency_grid(
                valid.reshape(1, 2), nominal_spacing_hz=1.0e6
            )
        with self.assertRaisesRegex(ValueError, "at least two"):
            validate_gotcha_frequency_grid(
                valid[:1], nominal_spacing_hz=1.0e6
            )
        with self.assertRaisesRegex(ValueError, "non-finite"):
            validate_gotcha_frequency_grid(
                torch.tensor([9.0e9, math.inf]), nominal_spacing_hz=1.0e6
            )
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            validate_gotcha_frequency_grid(
                torch.tensor([9.0e9, 9.0e9]), nominal_spacing_hz=1.0e6
            )
        with self.assertRaisesRegex(ValueError, "round-trip"):
            validate_gotcha_frequency_grid(
                torch.tensor(
                    [1_000_000.01, 1_000_100.01], dtype=torch.float64
                ),
                nominal_spacing_hz=100.0,
            )
        x_band_start = 9_600_000_000.0
        bad_steps = [1_465_344.0] * 50 + [1_464_320.0] * 50
        bad_affine = [x_band_start]
        for step in bad_steps:
            bad_affine.append(bad_affine[-1] + step)
        with self.assertRaisesRegex(ValueError, "endpoint linspace"):
            validate_gotcha_frequency_grid(
                torch.tensor(bad_affine, dtype=torch.float64),
                nominal_spacing_hz=1_464_320.0,
            )
        with self.assertRaisesRegex(ValueError, "positive"):
            validate_gotcha_frequency_grid(valid, nominal_spacing_hz=0.0)
        with self.assertRaisesRegex(ValueError, "disagrees"):
            validate_gotcha_frequency_grid(
                valid,
                nominal_spacing_hz=1.0e6,
                unambiguous_range_m=100.0,
            )


if __name__ == "__main__":
    unittest.main()
