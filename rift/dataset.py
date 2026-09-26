"""Loading AEDT-simulated radar CSV data.

Each CSV holds one viewpoint (a fixed (dphi, dtheta)) with one row per
frequency and magnitude/phase columns per Tx/Rx pair.
"""
import glob
import os
import random
import re

import numpy as np
import torch
from torch.utils.data import Dataset


def extract_index_from_filename(filename):
    """Extract the frame index from a filename of the form 'boxN.pkl'."""
    base = os.path.basename(filename)
    match = re.match(r'box(\d+)\.pkl', base)
    if match:
        return int(match.group(1))
    raise ValueError(f"Filename {filename} does not match the pattern 'boxN.pkl'")


def list_and_select_frame_indices(directory, num_files=40, exclude_files=None):
    """Randomly select pkl-format frame files and return their frame indices."""
    if exclude_files is None:
        exclude_files = []

    all_files = glob.glob(os.path.join(directory, "box*.pkl"))
    available_files = [file for file in all_files if file not in exclude_files]

    if len(available_files) < num_files:
        raise ValueError("Not enough files available to select the requested number.")

    selected_files = random.sample(available_files, num_files)
    selected_indices = [extract_index_from_filename(f) for f in selected_files]

    return selected_indices, selected_files


def list_and_select_files(directory, num_files=40, exclude_files=None):
    """Randomly select CSV viewpoint files from an AEDT data directory."""
    if exclude_files is None:
        exclude_files = []
    all_files = glob.glob(f"{directory}/*.csv")
    available_files = [file for file in all_files if file not in exclude_files]
    return random.sample(available_files, num_files)


class CSVSimulationDataset(Dataset):
    def __init__(self, file_paths, device="cpu"):
        self.device = device
        # Keep the exact role membership as lightweight provenance.  Training
        # still reads/parses the same files in the same order; adaptive-v2
        # checkpoints use this list only to reject a changed split on resume.
        self.file_paths = [str(file_path) for file_path in file_paths]
        self.simulation_data = []
        self.viewpoint_thetas = []

        for file_path in self.file_paths:
            simulation_item = self.process_csv_file(file_path)
            self.simulation_data.append(simulation_item)
            dtheta_tensor = simulation_item[2]
            self.viewpoint_thetas.append(dtheta_tensor[0].item())

    def __len__(self):
        return len(self.simulation_data)

    def __getitem__(self, idx):
        freqs_tensor, dphi_tensor, dtheta_tensor, magnitude_tensor, phase_tensor = self.simulation_data[idx]
        return freqs_tensor, dphi_tensor, dtheta_tensor, magnitude_tensor, phase_tensor

    def process_csv_file(self, csv_file_path):
        data = np.genfromtxt(csv_file_path, delimiter=',', skip_header=1, dtype=float)

        freqs_tensor = torch.tensor(data[:, 0], device=self.device, dtype=torch.float)
        dphi_tensor = torch.tensor(data[:, 1], device=self.device, dtype=torch.float)
        dtheta_tensor = torch.tensor(data[:, 2], device=self.device, dtype=torch.float)

        magnitude_data = data[:, 3::2]
        phase_data = data[:, 4::2]

        magnitude_tensor = torch.tensor(magnitude_data, device=self.device, dtype=torch.float)
        phase_tensor = torch.tensor(phase_data, device=self.device, dtype=torch.float)

        return freqs_tensor, dphi_tensor, dtheta_tensor, magnitude_tensor, phase_tensor
