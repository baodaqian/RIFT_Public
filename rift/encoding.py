"""Scene grid sampling and positional encoding for the INR scene model."""
import torch


def prepare_model_input(grid):
    return grid.view(-1, 3)


def positional_encoding(x, num_frequencies, include_input=True):
    if len(x.shape) == 1:
        x = x.unsqueeze(0)
    frequencies = 2.0 ** torch.arange(num_frequencies, dtype=torch.float32, device=x.device)
    encodings = []
    for freq in frequencies:
        encodings.append(torch.sin(freq * x))
        encodings.append(torch.cos(freq * x))
    if include_input:
        encodings = [x] + encodings
    return torch.cat(encodings, dim=-1)


def generate_dynamic_grid(granularity, extent, device, jitter: bool = False):
    """Voxel grid over [-extent, extent]^3, one sample per cell per axis.

    jitter=True: random sample within each cell (legacy behavior -- note
    this grid is built once per training run, not resampled per epoch, so
    it's a single fixed random realization, not true stratified sampling).
    jitter=False (default): cell midpoints, i.e. a plain uniformly-spaced
    grid. Required by the NUFFT forward operator (rift/nufft_forward_operator.py),
    which needs regular spacing for torch.fft.fftn.
    """
    intervals = torch.linspace(-extent, extent, granularity + 1, device=device)

    def sample_from_intervals(intervals, granularity, device):
        sampled_points = []
        for i in range(granularity):
            low, high = intervals[i], intervals[i + 1]
            sampled_point = torch.rand(1, device=device) * (high - low) + low
            sampled_points.append(sampled_point)
        return torch.cat(sampled_points)

    if jitter:
        x_coords = sample_from_intervals(intervals, granularity, device)
        y_coords = sample_from_intervals(intervals, granularity, device)
        z_coords = sample_from_intervals(intervals, granularity, device)
    else:
        x_coords = y_coords = z_coords = (intervals[:-1] + intervals[1:]) / 2

    coords = torch.cartesian_prod(x_coords, y_coords, z_coords)
    return coords.view(granularity, granularity, granularity, 3)


def total_variation_3d(x):
    dx = x[1:, :-1, :-1] - x[:-1, :-1, :-1]
    dy = x[:-1, 1:, :-1] - x[:-1, :-1, :-1]
    dz = x[:-1, :-1, 1:] - x[:-1, :-1, :-1]
    grad_mag = torch.sqrt(dx**2 + dy**2 + dz**2)
    return torch.sum(grad_mag)
