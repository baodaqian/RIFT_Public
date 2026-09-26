#!/usr/bin/env python
"""Build a shared pulse-compressed sonar cache from the public transients.

The defaults reproduce the public simulated-scene command: 20 kHz bandwidth,
20x transient binning, two bounce channels, and 20 dB signal SNR.  Operations
are batched and vectorized, avoiding the release script's 54,000 Python FFT
loop while retaining the same LFM and matched-filter definitions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.signal
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def analytic_signal(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[-1]
    spectrum = torch.fft.fft(x, dim=-1)
    h = torch.zeros(n, device=x.device, dtype=x.dtype)
    h[0] = 1
    if n % 2 == 0:
        h[n // 2] = 1
        h[1 : n // 2] = 2
    else:
        h[1 : (n + 1) // 2] = 2
    return torch.fft.ifft(spectrum * h, dim=-1)


def lfm(fs: float, f_start: float, f_stop: float, duration: float, tukey: float) -> np.ndarray:
    times = np.linspace(0.0, duration - 1.0 / fs, num=int(duration * fs))
    waveform = scipy.signal.chirp(times, f_start, duration, f_stop, phi=0)
    return waveform * scipy.signal.windows.tukey(len(waveform), tukey)


def padded_analytic_fft(waveform: np.ndarray, length: int, device: torch.device) -> torch.Tensor:
    padded = torch.zeros(length, dtype=torch.float64, device=device)
    padded[: waveform.shape[0]] = torch.as_tensor(waveform, dtype=torch.float64, device=device)
    return torch.fft.fft(analytic_signal(padded), dim=-1)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system-data", required=True)
    parser.add_argument("--transients", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--reed-root", default="external/Reed_SAS_reference")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--max-pings", type=int, default=0, help="0 uses all 54,000")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bin-upsample", type=int, default=20)
    parser.add_argument("--bounces", type=int, default=2)
    parser.add_argument("--snr-db", type=float, default=20.0)
    parser.add_argument("--bandwidth-khz", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bin_upsample <= 0 or args.batch_size <= 0 or args.bounces <= 0:
        raise ValueError("bin-upsample, batch-size, and bounces must be positive")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    old_manifest = None
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            old_manifest = json.load(handle)
        if old_manifest.get("complete") is True:
            print(f"Complete cache already exists at {output}; refusing to overwrite.")
            return

    reed_root = Path(args.reed_root).resolve()
    sys.path.insert(0, str(reed_root))
    from rift.sas_dataset import atomic_json, restricted_load_system_data, schema_dict

    system_data = restricted_load_system_data(args.system_data)
    transient = np.load(args.transients, mmap_mode="r", allow_pickle=False)
    if transient.ndim != 3 or transient.shape[-1] < args.bounces:
        raise ValueError(f"unexpected transient shape {transient.shape}")
    num_pings = transient.shape[0] if args.max_pings <= 0 else min(args.max_pings, transient.shape[0])
    crop = schema_dict(system_data["crop_settings"])
    sys_params = schema_dict(system_data["sys_params"])
    wfm_params = schema_dict(system_data["wfm_params"])
    geometry = schema_dict(system_data["geometry"])
    num_bins = int(crop["num_samples"])
    if transient.shape[1] != num_bins * args.bin_upsample:
        raise ValueError(
            f"transient bins {transient.shape[1]} != {num_bins} * {args.bin_upsample}"
        )
    if abs(args.bandwidth_khz - 20.0) > 1e-9:
        raise ValueError("the reportable SH-SAS simulation profile currently supports 20 kHz only")

    contract = {
        "complete": False,
        "scene": args.scene,
        "frontend": "reed_lfm_matched_filter",
        "source_transients": str(Path(args.transients).resolve()),
        "source_system_data": str(Path(args.system_data).resolve()),
        "reed_commit": "0de52a000dd8d054ae70eaf44db4e099308126de",
        "num_pings": int(num_pings),
        "num_bins": int(num_bins),
        "bin_upsample": int(args.bin_upsample),
        "bounces": int(args.bounces),
        "snr_db": float(args.snr_db),
        "bandwidth_khz": float(args.bandwidth_khz),
        "seed": int(args.seed),
        "paper_simulation_contract": {
            "center_frequency_hz": 20000.0,
            "bandwidth_hz": 20000.0,
            "sampling_frequency_hz": float(sys_params["sampling_frequency"]),
            "snr_db": float(args.snr_db),
        },
        "pulse_completed": 0,
        "matched_completed": 0,
        "positive_power_sum": 0.0,
        "positive_count": 0,
    }
    if old_manifest is not None:
        immutable = (
            "scene",
            "source_transients",
            "source_system_data",
            "num_pings",
            "num_bins",
            "bin_upsample",
            "bounces",
            "snr_db",
            "bandwidth_khz",
            "seed",
        )
        mismatches = [key for key in immutable if old_manifest.get(key) != contract.get(key)]
        if mismatches:
            raise RuntimeError(f"incomplete cache contract mismatch in {mismatches}; use a new output")
        manifest = old_manifest
        print(
            f"Resuming cache: pulse={manifest.get('pulse_completed', 0)}/{num_pings}, "
            f"matched={manifest.get('matched_completed', 0)}/{num_pings}",
            flush=True,
        )
    else:
        manifest = contract
    atomic_json(manifest, manifest_path)

    device = torch.device(args.device)
    fs = float(sys_params["sampling_frequency"])
    duration = float(wfm_params["t_dur"])
    window_ratio = float(wfm_params["win_ratio"])
    base_waveform = lfm(fs, 30000.0, 10000.0, duration, window_ratio)
    high_waveform = scipy.signal.resample(base_waveform, base_waveform.shape[0] * args.bin_upsample)
    high_kernel = padded_analytic_fft(high_waveform, transient.shape[1], device)
    base_kernel = padded_analytic_fft(base_waveform, num_bins, device)

    clean_path = output / "raw_clean.npy"
    if clean_path.exists():
        clean = np.load(clean_path, mmap_mode="r+", allow_pickle=False)
        if clean.shape != (num_pings, num_bins) or clean.dtype != np.float32:
            raise RuntimeError("existing raw_clean.npy has the wrong cache contract")
    else:
        clean = np.lib.format.open_memmap(
            clean_path, mode="w+", dtype=np.float32, shape=(num_pings, num_bins)
        )
    positive_power_sum = float(manifest.get("positive_power_sum", 0.0))
    positive_count = int(manifest.get("positive_count", 0))
    pulse_start = int(manifest.get("pulse_completed", 0))
    start_time = time.time()
    with torch.no_grad():
        for start in range(pulse_start, num_pings, args.batch_size):
            stop = min(start + args.batch_size, num_pings)
            impulse_np = np.asarray(transient[start:stop, :, : args.bounces]).sum(axis=-1)
            impulse = torch.as_tensor(impulse_np, device=device, dtype=torch.float64)
            convolved = torch.fft.ifft(torch.fft.fft(impulse, dim=-1) * high_kernel, dim=-1).real
            downsampled = convolved[:, :: args.bin_upsample].to(torch.float32).cpu().numpy()
            clean[start:stop] = downsampled
            positive = downsampled[downsampled > 1.0e-6]
            positive_power_sum += float(np.square(positive, dtype=np.float64).sum())
            positive_count += int(positive.size)
            clean.flush()
            manifest.update(
                {
                    "pulse_completed": stop,
                    "positive_power_sum": positive_power_sum,
                    "positive_count": positive_count,
                }
            )
            atomic_json(manifest, manifest_path)
            if start == 0 or stop == num_pings or stop % 1024 == 0:
                print(f"pulse synthesis {stop}/{num_pings}", flush=True)
    clean.flush()
    if positive_count == 0:
        raise RuntimeError("no positive signal samples exceeded the authors' 1e-6 SNR gate")
    signal_watts = positive_power_sum / positive_count
    noise_watts = 10.0 ** ((10.0 * np.log10(signal_watts) - args.snr_db) / 10.0)
    noise_std = float(np.sqrt(noise_watts))

    weights_path = output / "weights.npy"
    if weights_path.exists():
        weights = np.load(weights_path, mmap_mode="r+", allow_pickle=False)
        if weights.shape != (num_pings, num_bins) or weights.dtype != np.complex64:
            raise RuntimeError("existing weights.npy has the wrong cache contract")
    else:
        weights = np.lib.format.open_memmap(
            weights_path, mode="w+", dtype=np.complex64, shape=(num_pings, num_bins)
        )
    matched_start = int(manifest.get("matched_completed", 0))
    with torch.no_grad():
        for start in range(matched_start, num_pings, args.batch_size):
            stop = min(start + args.batch_size, num_pings)
            # Batch-addressed RNG makes a resumed slice bit-identical without
            # serializing a large NumPy generator state.
            rng = np.random.default_rng(args.seed + start)
            noise = rng.normal(0.0, noise_std, size=(stop - start, num_bins)).astype(np.float32)
            raw = torch.as_tensor(np.asarray(clean[start:stop]) + noise, device=device, dtype=torch.float64)
            compressed = torch.fft.ifft(
                torch.fft.fft(analytic_signal(raw), dim=-1) * torch.conj(base_kernel), dim=-1
            )
            weights[start:stop] = compressed.to(torch.complex64).cpu().numpy()
            weights.flush()
            manifest["matched_completed"] = stop
            atomic_json(manifest, manifest_path)
            if start == 0 or stop == num_pings or stop % 1024 == 0:
                print(f"matched filtering {stop}/{num_pings}", flush=True)
    weights.flush()

    radii = np.linspace(float(crop["min_dist"]), float(crop["max_dist"]), num_bins).astype(np.float32)
    np.savez(
        output / "geometry.npz",
        tx_coords=np.asarray(system_data["tx_coords"][:num_pings], dtype=np.float32),
        rx_coords=np.asarray(system_data["rx_coords"][:num_pings], dtype=np.float32),
        radii=radii,
        corners=np.asarray(geometry["corners"], dtype=np.float32),
        voxels=np.asarray(geometry["voxels"], dtype=np.float32),
        grid_shape=np.asarray(
            [geometry["num_x"], geometry["num_y"], geometry["num_z"]], dtype=np.int64
        ),
    )
    manifest.update(
        {
            "complete": True,
            "noise_std": noise_std,
            "signal_watts_authors_gate": signal_watts,
            "elapsed_seconds": time.time() - start_time,
            "weights_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        }
    )
    atomic_json(manifest, manifest_path)
    print(f"Wrote complete shared SAS cache to {output}", flush=True)


if __name__ == "__main__":
    main()
