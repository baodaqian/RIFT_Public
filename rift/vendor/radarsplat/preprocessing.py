"""RadarSplat preprocessing excerpts, unchanged from the official implementation.

Authors: Pou-Chun Kung and the RadarSplat authors / umautobots.
Source: boreas/data_processing/play_radar_signal.py
Commit: ea9c8f530c708622cc3b1b560436b5557ac6a49b
https://github.com/umautobots/radarsplat
License: CC BY-NC-SA 4.0; see LICENSE in this directory.
Only two pure functions are retained; data loading, plotting and CUDA imports
are omitted. No mathematical changes were made to these functions.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d

def apply_saturation_mask(polar_image, saturate_azi_mask, smoothing_sigma, occ_thres=0.1):
    """
    Replace values outside the decay region with 0 for rows where polar_fft[:, 0] > threshold.

    Args:
    - polar_image (numpy array): 2D array of radar polar intensities (shape: [rows, range_pixels]).
    - polar_fft (numpy array): 2D FFT magnitude spectrum of the polar image.
    - saturate_azi_mask: Saturation mask for selecting affected rows.

    Returns:
    - modified_polar (numpy array): Modified polar image with values outside decay regions set to 0.
    - occ_polar (numpy array): Occupany polar image with values after decay regions set to 0.5 (unknown).
    """
    modified_polar = polar_image.copy()
    occ_polar = polar_image.copy()
    occ_polar[occ_polar>=occ_thres]=np.clip(occ_polar+0.5, 0, 1)[occ_polar>=occ_thres]
    occ_polar[occ_polar<occ_thres]=0.0

    for row_idx in np.where(saturate_azi_mask)[0]:  # Get indices where saturation occurs
        max_idx, decay_region = find_decay_region(polar_image[row_idx], sigma=smoothing_sigma)

        # Zero out values outside the decay region
        modified_polar[row_idx, :decay_region[0]] = 0  # Left side
        modified_polar[row_idx, decay_region[1] + 1:] = 0  # Right side
        
        occ_polar[row_idx, :decay_region[0]] = 0.5
        occ_polar[row_idx, decay_region[1] + 1:] = 0.5

    return modified_polar, occ_polar

def find_decay_region(intensity_values, sigma=1.0):
    """
    Find the region where intensity values continuously decay after the maximum point.

    Args:
    - intensity_values (numpy array): 1D array of intensity values.

    Returns:
    - max_index (int): Index of the maximum intensity value.
    - decay_region (tuple): (start_index, end_index) of the decay region.
    """
    
    smoothed_values = gaussian_filter1d(intensity_values, sigma=sigma)

    # Step 1: Find max intensity value and its index
    max_index = np.argmax(smoothed_values)
    
    # Step 2: Find left boundary (backward search)
    start_index = max_index
    while start_index > 0 and smoothed_values[start_index - 1] <= smoothed_values[start_index]: # (Frank) bug fix 0225. TODO: Rerun experiments
        start_index -= 1
    
    # Step 3: Find right boundary (forward search)
    end_index = max_index
    while end_index < len(smoothed_values) - 1 and smoothed_values[end_index + 1] <= smoothed_values[end_index]:
        end_index += 1
    
    # Step 4: Return the max index and decay region
    return max_index, (start_index, end_index)
