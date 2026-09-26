"""Metadata-first SE paper-recipe adapters; no historical loader relaxation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import torch

from .sugavanam_ertin_paper import unit_vectors, Subapertures


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def array_digest(*arrays):
    h = hashlib.sha256()
    for array in arrays:
        a = np.ascontiguousarray(array)
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


class CollectionAcquisition:
    """All sealed pairs/frequencies; first-order Fourier acquisition extension.

    SE supplies no MIMO implementation. Use the bistatic spatial frequency
    k*(u_tx+u_rx), which reduces to its monostatic Fourier model. Do not add
    exact near-field curvature to improve this comparator's measurement model.
    """
    kind = "rift_collection"
    extent = .15

    def __init__(self, *, object_name=None, dataset_root=None, npz_path=None, manifest=None):
        from .rift_dataset import (DEFAULT_ROOT, resolve_object_inputs, load_object_contract,
                                   evaluation_role_indices)
        from .npz_dataset import build_freqs
        npz_path, manifest = resolve_object_inputs(object_name=object_name,
            dataset_root=DEFAULT_ROOT if dataset_root is None else dataset_root,
            npz_path=npz_path, role_manifest_path=manifest)
        self.arrays, self.contract = load_object_contract(npz_path, manifest)
        self.keys = {r: evaluation_role_indices(self.contract, r).tolist() for r in ("train", "validation")}
        self.train_sample_counts = [int(np.prod(self.arrays["response_shape"][1:]))]*len(self.keys["train"])
        self.directions = {r: unit_vectors(self.arrays["viewpoint_positions"][ids]) for r, ids in self.keys.items()}
        self.freqs = build_freqs(self.arrays["meta"])
        from .config import cc
        self.range_resolution_m = cc/(2*float(self.arrays["meta"]["radar_bandwidth_hz"]))
        self.path = Path(npz_path)
        self.signature = self._stat()
        with zipfile.ZipFile(npz_path) as archive:
            header = archive.getinfo("response.npy")
            response_identity = dict(crc32=header.CRC, size=header.file_size)
        self.identity = dict(kind=self.kind, contract=self.contract,
            acquisition_sha256=array_digest(self.freqs, self.arrays["tx_pos"], self.arrays["rx_pos"],
                                            self.arrays["viewpoint_positions"]),
            response=response_identity, metadata=self.arrays["meta"],
            operator="first_order_bistatic_fourier_native_reference_origin_spreading_v2",
            paper_scope="bistatic_extension_not_specified_by_SE",
            target="full_native_complex", chirps="coherent_mean")
        self.identity = json.loads(json.dumps(self.identity))
        # Scalar conditioning only: reference-origin RMS spreading on TRAIN.
        ids = self.keys["train"]
        tx = np.linalg.norm(self.arrays["tx_pos"][ids], axis=-1)
        rx = np.linalg.norm(self.arrays["rx_pos"][ids], axis=-1)
        kernel = (4*np.pi)**-2 / (tx[:, :, None]+rx[:, None, :])**2
        self.kernel_scale = float(np.sqrt(np.mean(kernel**2)))

    def _stat(self):
        s = self.path.stat()
        return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns

    def observations(self, key, *, role):
        from .npz_dataset import get_npz_response_view
        if role not in self.keys or key not in self.keys[role]:
            raise PermissionError("SE observation is outside its declared train/validation role")
        if self._stat() != self.signature:
            raise ValueError("Collection source changed after metadata validation")
        raw = get_npz_response_view(self.arrays, int(key))
        target = np.asarray(raw, dtype=np.complex128).mean(axis=2).transpose(2, 1, 0)
        if not np.isfinite(target).all():
            raise ValueError("Nonfinite native complex observation")
        yield dict(response=target, tx=self.arrays["tx_pos"][key], rx=self.arrays["rx_pos"][key], freqs=self.freqs)

    def render(self, points, weights, observation, *, point_chunk, pair_chunk):
        from .config import cc
        tx = torch.as_tensor(observation["tx"], device=points.device, dtype=torch.float64)
        rx = torch.as_tensor(observation["rx"], device=points.device, dtype=torch.float64)
        rt, rr = tx.norm(dim=-1), rx.norm(dim=-1)
        directions = (rx[:, None]/rr[:, None, None]+tx[None]/rt[None, :, None]).reshape(-1, 3)
        reference = (rr[:, None]+rt[None, :]).flatten()
        amplitude = (4*torch.pi)**-2/reference.square()/self.kernel_scale
        result = fourier_forward(points, weights, observation["freqs"], directions,
            reference, amplitude, cc=cc, point_chunk=point_chunk, pair_chunk=pair_chunk)
        return result.reshape(len(observation["freqs"]), len(rx), len(tx))


class GOTCHAAcquisition:
    """HH fields shared across elevation passes; declared native sector pulses.

    Fit full complex phase history. Same-delay and out-of-ROI clutter remain a
    modeling residual; this adapter does not pretend to isolate vehicle returns.
    """
    kind = "gotcha_native"
    kernel_scale = 1.

    def __init__(self, dataset):
        if tuple(dataset.polarizations) != ("hh",):
            raise ValueError("SE paper-v1 currently supports HH only")
        self.dataset = dataset
        self.extent = dataset.region.half_extent_m
        from .gotcha_dataset import C
        self.range_resolution_m = C/(2*max(float(np.ptp(s.frequencies_hz)) for s in dataset.shards.values()))
        original_keys = {r: dataset.viewpoints(r) for r in ("train", "validation")}
        self.native_view_counts = {r: len(keys) for r, keys in original_keys.items()}
        self.keys, self.pulse_ids, self.train_sample_counts = {}, {}, []
        self.directions = {}
        self.direction_statistics = None
        for role, keys in original_keys.items():
            self.keys[role] = []
            local_centres = []
            statistics = []
            for p, sector in keys:
                shard = dataset.shards[p, "hh"]
                rows = shard.sector_rows[sector]
                xyz = np.stack([shard.arrays[k][rows].astype(np.float64) for k in ("x", "y", "z")], -1)
                local = dataset.region.to_local(xyz)
                u = unit_vectors(local)
                bins = Subapertures.bin_ids(u, 72, 1)
                # An ingress sector can straddle a 5-degree boundary. Assign
                # every native pulse itself, without changing its sealed role.
                for angular_bin in np.unique(bins):
                    selected = bins == angular_bin
                    key = (p, sector, int(angular_bin))
                    self.keys[role].append(key)
                    self.pulse_ids[key] = set(shard.arrays["pulse_index"][rows[selected]].tolist())
                    local_centres.append(u[selected].mean(0))
                    az = np.arctan2(u[selected, 1], u[selected, 0])
                    el = np.arcsin(np.clip(u[selected, 2], -1, 1))
                    statistics.append([np.cos(az).sum(), np.sin(az).sum(), el.sum(), len(az)])
                    if role == "train":
                        self.train_sample_counts.append(int(selected.sum())*len(shard.frequencies_for_role('train')))
            self.directions[role] = unit_vectors(local_centres)
            if role == "train":
                self.direction_statistics = np.asarray(statistics, dtype=np.float64)
        self.identity = dict(kind=self.kind, dataset_identity=dataset.identity, contract=dataset.contract,
            target="full_native_complex", autofocus="source_channel_owned_once_by_ingress",
            operator="monostatic_far_field_fourier_native_reference_v2",
            angular_assignment="each_native_pulse_5_degree_azimuth_all_passes",
            pulse_partition_sha256=digest([{ "key": k, "pulses": sorted(v)} for k, v in self.pulse_ids.items()]),
            viewpoint=("pass_sector_same_fixed_cap_train_validation_test" if dataset.training_pulse_selection
                       else "pass_sector_all_native_pulses"), frequencies="ragged_exact")
        if dataset.frequency_selection:
            from rift.gotcha_frequency_selection import recipe_fields
            self.identity.update(recipe_fields(dataset),
                target="native_complex_same_selection_rule_train_validation_test",
                frequencies="role_specific_ragged_exact_native_bins")

    def observations(self, key, *, role):
        if role not in self.keys or key not in self.keys[role]:
            raise PermissionError("SE observation is outside its declared train/validation role")
        shard = self.dataset.shards[key[0], "hh"]
        for row in shard.sector_rows[key[1]]:
            if int(shard.arrays["pulse_index"][row]) in self.pulse_ids[key]:
                # The shared reader still enforces roles, source identity and
                # source-owned autofocus before touching the selected response.
                yield shard.read(int(row))

    def render(self, points, weights, observation, *, point_chunk, pair_chunk):
        from .gotcha_dataset import C
        rotation = torch.as_tensor(self.dataset.region.rotation_local_to_native, device=points.device, dtype=torch.float64)
        translation = torch.as_tensor(self.dataset.region.translation_m, device=points.device, dtype=torch.float64)
        antenna = torch.as_tensor(observation.position_m, device=points.device, dtype=torch.float64)
        relative = (antenna-translation) @ rotation
        r = relative.norm()
        directions = 2*(relative/r)[None]
        reference = (2*(r-observation.reference_range_m)).reshape(1)
        return fourier_forward(points, weights, observation.frequencies_hz.copy(), directions,
            reference, torch.ones_like(reference), cc=C, point_chunk=point_chunk, pair_chunk=1)[:, 0]


def fourier_forward(points, weights, frequencies, directions, reference, amplitude, *,
                    cc, point_chunk, pair_chunk):
    """SE Eq. 2 in the native phase reference, no quadratic range terms.

    Antenna-facing u gives exp(+ik*u.x) for native exp(-ik*R). The paper's
    negative Fourier sign uses propagation-facing -u. The known reference
    phase/amplitude converts back to native observation units, without a gain.
    """
    from torch.utils.checkpoint import checkpoint
    k = 2*torch.pi*torch.as_tensor(frequencies, device=points.device, dtype=torch.float64)/cc
    blocks = []
    for start in range(0, len(reference), pair_chunk):
        d, r, a = (v[start:start+pair_chunk] for v in (directions, reference, amplitude))
        local_chunk = min(point_chunk, max(1, 1048576//(len(k)*len(r))))
        result = weights.new_zeros((len(k), len(r)))
        # All block variables are explicit inputs: backward cannot capture a
        # later loop iteration's directions or reference range.
        def block(p, w, direction, origin_range, scale):
            path = origin_range[None]-p.double() @ direction.T
            kernel = torch.exp(-1j*k[:, None, None]*path[None])*scale[None, None]
            return (kernel*w[None, :, None]).sum(1)
        for p, w in zip(points.split(local_chunk), weights.split(local_chunk)):
            if torch.is_grad_enabled() and (p.requires_grad or w.requires_grad):
                result = result+checkpoint(block, p, w, d, r, a, use_reentrant=False)
            else:
                result = result+block(p, w, d, r, a)
        blocks.append(result)
    return torch.cat(blocks, dim=1)


def response(observation):
    return observation["response"] if isinstance(observation, dict) else observation.response


def training_statistics(acquisition):
    """Scalar RMS and full measurement counts using TRAIN only; no cached cubes."""
    energy, total, counts, energies = 0., 0, [], []
    for key in acquisition.keys["train"]:
        count = 0
        local_energy = 0.
        for observation in acquisition.observations(key, role="train"):
            y = np.asarray(response(observation))
            local_energy += float(np.vdot(y.reshape(-1), y.reshape(-1)).real)
            count += y.size
        if count == 0:
            raise ValueError("Empty training viewpoint")
        counts.append(count)
        energies.append(local_energy)
        energy += local_energy
        total += count
    rms = float(np.sqrt(energy/total))
    if counts != acquisition.train_sample_counts:
        raise ValueError("Observed training sample counts differ from native acquisition headers")
    if not np.isfinite(rms) or rms <= 0:
        raise ValueError("Training response RMS must be finite and positive")
    return dict(schema="se_train_rms_v1", identity=digest(acquisition.identity),
                rms=rms, samples=total, per_view_samples=counts, per_view_energy=energies,
                kernel_scale=acquisition.kernel_scale, fitting_exposure=False)


def data_objective(acquisition, points, weights, indices, statistics, recipe, *, gradient=False):
    """Full sub-aperture mean squared residual, streamed per native observation."""
    denominator = sum(statistics["per_view_samples"][i] for i in indices)
    x = weights.detach().requires_grad_(gradient)
    loss = 0.
    g = torch.zeros_like(x) if gradient else None
    for i in indices:
        for observation in acquisition.observations(acquisition.keys["train"][i], role="train"):
            with torch.set_grad_enabled(gradient):
                pred = acquisition.render(points, x, observation, point_chunk=recipe["point_chunk"], pair_chunk=recipe["pair_chunk"])
                target = torch.as_tensor(response(observation), device=points.device, dtype=torch.complex128)/statistics["rms"]
                term = .5*(pred-target).abs().square().sum()/denominator
                if gradient:
                    g += torch.autograd.grad(term, x)[0].detach()
            loss += float(term.detach())
    return (loss, g) if gradient else loss


@torch.no_grad()
def validation_readout(acquisition, points, fields, assignments, statistics, recipe):
    numerator, denominator, count = 0., 0., 0
    for i, key in enumerate(acquisition.keys["validation"]):
        weights = fields[int(assignments[i])].to(points.device)
        for observation in acquisition.observations(key, role="validation"):
            pred = acquisition.render(points, weights, observation, point_chunk=recipe["point_chunk"], pair_chunk=recipe["pair_chunk"])
            target = torch.as_tensor(response(observation), device=points.device, dtype=torch.complex128)/statistics["rms"]
            numerator += float((pred-target).abs().square().sum())
            denominator += float(target.abs().square().sum())
            count += target.numel()
    if denominator <= 0 or not np.isfinite([numerator, denominator]).all():
        raise ValueError("Invalid full-native validation readout")
    return dict(global_complex_rel_mse=numerator/denominator, squared_error=numerator,
                target_energy=denominator, samples=count, views=len(assignments),
                output="stage1_subaperture_diagnostic_not_SDF_NVS")
