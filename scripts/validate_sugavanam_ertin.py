#!/usr/bin/env python
"""CPU contract gates for the Sugavanam--Ertin reimplementation."""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rift.sugavanam_ertin import (  # noqa: E402
    METHOD_NAME,
    FourierFeatureSDF,
    estimate_pca_normals,
    load_scattering_cloud,
    project_to_zero_level,
    spatial_gradient,
    uniformize_points,
)
from compare_sugavanam_ertin_geometry import surface_calibration_metadata  # noqa: E402
from export_sugavanam_ertin_calibrated import (  # noqa: E402
    CALIBRATION_POLICY,
    calibrated_output_dir,
    calibrate_level,
    export_surface,
    sha256,
)


def gate(name, condition, detail=""):
    if not condition:
        raise AssertionError(f"FAIL {name}: {detail}")
    print(f"PASS {name}{': ' + detail if detail else ''}")


def synthetic_scatter_checkpoint(path):
    g, extent = 12, 0.15
    pitch = 2 * extent / g
    axis = torch.linspace(-extent + pitch / 2, extent - pitch / 2, g)
    pos = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), -1)
    radius = pos.norm(dim=-1)
    mag = torch.exp(-0.5 * ((radius - 0.07) / 0.012) ** 2)
    phase = 3.0 * pos[..., 0]
    torch.save(
        {
            "epoch": 2,
            "scene_repr": "grid",
            "extent": extent,
            "granularity": g,
            "model_state_dict": {
                "w_re": mag * torch.cos(phase),
                "w_im": mag * torch.sin(phase),
                "active_mask": torch.ones_like(mag, dtype=torch.bool),
                "grid_positions": pos,
            },
        },
        path,
    )


class SphereSDF(torch.nn.Module):
    def forward(self, xyz):
        return xyz.norm(dim=-1) - 0.07


class ShiftedSphereSDF(torch.nn.Module):
    def forward(self, xyz):
        return xyz.norm(dim=-1) - 0.07 - 0.025


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    gate("identity", "Sugavanam" in METHOD_NAME and "Ertin" in METHOD_NAME)

    with tempfile.TemporaryDirectory() as td:
        ck = os.path.join(td, "scatter.pth.tar")
        synthetic_scatter_checkpoint(ck)
        cloud = load_scattering_cloud(ck, threshold_fraction=0.2, max_points=500)
        gate("scattering-centre extraction", 20 < len(cloud.points) <= 500, f"N={len(cloud.points)}")
        gate("checkpoint provenance", cloud.source_epoch == 2 and cloud.granularity == 12)
        gate("finite PCA normals", np.isfinite(cloud.normals).all())
        gate("unit PCA normals", np.allclose(np.linalg.norm(cloud.normals, axis=1), 1, atol=1e-5))

        model = FourierFeatureSDF(extent=0.15, n_fourier=9, hidden_dim=32, n_layers=5, seed=42)
        xyz = torch.randn(24, 3) * 0.05
        sdf, grad = spatial_gradient(model, xyz.clone(), create_graph=True)
        gate("SDF output shape", sdf.shape == (24,))
        gate("spatial-gradient shape", grad.shape == (24, 3))
        loss = sdf.abs().mean() + (grad.norm(dim=-1) - 1).abs().mean()
        loss.backward()
        gate("second-order loss gradients", all(p.grad is not None for p in model.parameters()))

        q0 = torch.randn(128, 3) * 0.04
        err0 = SphereSDF()(q0).abs().mean()
        q1 = project_to_zero_level(SphereSDF(), q0, extent=0.15, max_step=0.03)
        err1 = SphereSDF()(q1).abs().mean()
        gate("Newton zero-level projection", float(err1) < float(err0) * 0.05,
             f"{float(err0):.3g}->{float(err1):.3g}")

        dup = torch.zeros(32, 3)
        dup[:, 0] = torch.linspace(-0.01, 0.01, 32)
        spread0 = dup.std(dim=0).norm()
        spread1 = uniformize_points(dup, bandwidth=0.01, k=8).std(dim=0).norm()
        gate("iso-point repulsion", float(spread1) > float(spread0))

        state = model.state_dict()
        clone = FourierFeatureSDF(**model.config())
        clone.load_state_dict(state)
        gate("checkpoint architecture roundtrip", torch.equal(model(xyz), clone(xyz)))

        # Geometry truth cannot leak through the training API because no such
        # parameter exists in the model or cloud loader signatures.
        forbidden = {"stl", "mesh", "lidar", "ground_truth"}
        names = set(load_scattering_cloud.__code__.co_varnames) | set(model.forward.__code__.co_varnames)
        gate("geometry-truth exclusion", forbidden.isdisjoint(names), str(sorted(names & forbidden)))

        directions = torch.randn(129, 3)
        directions = directions / directions.norm(dim=-1, keepdim=True)
        centre_points = directions * torch.linspace(0.068, 0.072, 129)[:, None]
        shifted = ShiftedSphereSDF()
        calibration = calibrate_level(shifted, centre_points.numpy(), torch.device("cpu"), chunk=23)
        gate("stage-1 L1 isolevel calibration", abs(calibration["raw_isolevel"] + 0.025) < 1e-6,
             f"level={calibration['raw_isolevel']:.6g}")
        centred = shifted(centre_points).detach().numpy() - calibration["raw_isolevel"]
        gate("calibrated centre median is zero", abs(float(np.median(centred))) < 1e-6)

        source = {
            "checkpoint": "/synthetic/checkpoint_final.pth.tar",
            "sha256": "a" * 64,
            "step": 5000,
            "best_loss": 0.125,
        }
        cal_dir = os.path.join(td, "calibrated")
        surface = export_surface(
            shifted, cal_dir, 0.15, 24, 256, 0, torch.device("cpu"), calibration, source,
            "b" * 64, chunk=128,
        )
        with np.load(surface, allow_pickle=False) as z:
            gate("calibrated surface export", len(z["vertices"]) > 0 and len(z["surface_points"]) == 256)
            gate("calibrated surface provenance", str(z["calibration_policy"].item()) == CALIBRATION_POLICY and
                 str(z["source_checkpoint_role"].item()) == "checkpoint_final" and
                 abs(float(z["raw_isolevel"]) - calibration["raw_isolevel"]) < 1e-6)

        raw_dir = os.path.join(td, "raw")
        scatter_dir = os.path.join(td, "scatter")
        os.makedirs(raw_dir)
        os.makedirs(scatter_dir)
        final_ckpt = os.path.join(raw_dir, "checkpoint_final.pth.tar")
        scatter_ckpt = os.path.join(scatter_dir, "checkpoint_final.pth.tar")
        scatter_sha = "b" * 64
        torch.save(
            {
                "method": METHOD_NAME,
                "step": 5000,
                "ground_truth_geometry_used": False,
                "scatter_checkpoint_sha256": scatter_sha,
            },
            final_ckpt,
        )
        npz_path = os.path.join(td, "calibration_meta.npz")
        np.savez_compressed(
            npz_path,
            method=np.asarray(METHOD_NAME),
            sign_flipped=np.bool_(False),
            orientation=np.int8(1),
            raw_isolevel=np.float64(-0.025),
            calibration_policy=np.asarray(CALIBRATION_POLICY),
            source_checkpoint=np.asarray(final_ckpt),
            source_checkpoint_sha256=np.asarray(sha256(final_ckpt)),
            source_checkpoint_role=np.asarray("checkpoint_final"),
            scatter_checkpoint_sha256=np.asarray(scatter_sha),
        )
        with np.load(npz_path, allow_pickle=False) as z:
            meta = surface_calibration_metadata(z, npz_path)
        gate("calibrated evaluator validates retained final", meta["source_role"] == "checkpoint_final")

        bad_meta = os.path.join(td, "bad_calibration_meta.npz")
        np.savez_compressed(
            bad_meta,
            method=np.asarray(METHOD_NAME),
            sign_flipped=np.bool_(False),
            orientation=np.int8(1),
            raw_isolevel=np.float64(-0.025),
            calibration_policy=np.asarray(CALIBRATION_POLICY),
            source_checkpoint=np.asarray(os.path.join(raw_dir, "checkpoint_best.pth.tar")),
            source_checkpoint_sha256=np.asarray(sha256(final_ckpt)),
            source_checkpoint_role=np.asarray("checkpoint_final"),
            scatter_checkpoint_sha256=np.asarray(scatter_sha),
        )
        failed_source = False
        try:
            with np.load(bad_meta, allow_pickle=False) as z:
                surface_calibration_metadata(z, bad_meta)
        except ValueError:
            failed_source = True
        gate("evaluator rejects non-final provenance path", failed_source)

        failed_output_root = False
        try:
            calibrated_output_dir(os.path.join(raw_dir, "calibrated"), final_ckpt, scatter_ckpt)
        except ValueError:
            failed_output_root = True
        gate("calibrated output cannot nest under raw sources", failed_output_root)

        failed = False
        try:
            bad = dict(calibration)
            bad["raw_isolevel"] = 1.0
            export_surface(shifted, os.path.join(td, "bad"), 0.15, 24, 32, 0, torch.device("cpu"), bad,
                           source, "b" * 64, chunk=128)
        except RuntimeError:
            failed = True
        gate("no-crossing export fails closed", failed and not os.path.exists(os.path.join(td, "bad", "surface_reconstruction.npz")))

    print("PASS 20/20 Sugavanam--Ertin CPU contract gates")


if __name__ == "__main__":
    main()
