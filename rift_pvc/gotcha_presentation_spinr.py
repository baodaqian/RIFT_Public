"""SpINR (native GOTCHA) reconstruction on the presentation grid, relative and K-calibrated.

Plugs into ``rift_pvc.gotcha_presentation`` (see its docstring for the two
directories). Reads the checkpoint only: SpINR is a position-only field, so no
look direction, shard or response is needed.

- relative: the project's SpINR readout (``scripts/readout_spinr.py`` via
  ``rift.spinr_fidelity.field_readout``): sigma = initial_output_scale * field at
  the G48 cell midpoints, shown as sigma^2 (its magnitude squared, a power-like
  quantity for the dB scale).
- physical: the native render (``rift.spinr_native.NativeKernel.render``) is
  y = sum_n field_n * w_n * s * exp(-i 4 pi f (d_n - r0) / c) / d_n^2, with w_n the
  quadrature node's physical volume (m^3) and s the per-polarization
  ``initial_scales`` value. K's definition y / K = sqrt(RCS) / (2 d)^2 gives each
  node sqrt(RCS_n) = 4 s w_n field_n / K, independent of range (the kernel's d^2
  and K's (2d)^2 differ by the constant 4) and of look direction (no angular
  term). A cell's sqrt(RCS) is the sum of its nodes' signed amplitudes (its share
  of SpINR's own quadrature sum, intra-cell phase ignored); RCS is its square. At
  the pinned G48 midpoint recipe each cell holds exactly one node.
"""
from __future__ import annotations

from pathlib import Path

import torch

from rift_pvc.gotcha_presentation import K, PRESENTATION_GRID, MethodPresentation, deposit_points

FORMAT = 'spinr_gotcha_native_v1'


def load_head(checkpoint, polarization):
    """The saved SpinrStyleINR head, its fixed output scale and support half-extent."""
    from rift.spinr_style import SpinrStyleINR
    if checkpoint.get('method_format') != FORMAT:
        raise ValueError('not a native GOTCHA SpINR checkpoint')
    extent = float(checkpoint['dataset_contract']['region']['half_extent_m'])
    prefix = f'{polarization}.'
    state = {k[len(prefix):]: v for k, v in checkpoint['model_state_dict'].items() if k.startswith(prefix)}
    if not state:
        raise ValueError(f'checkpoint has no {polarization} head')
    model = SpinrStyleINR(support_m=extent)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, float(checkpoint['initial_scales'][polarization]['value']), extent


@torch.no_grad()
def _field(model, points, tile=4096):
    return torch.cat([model(p).to(torch.float64) for p in points.split(tile)])


@torch.no_grad()
def node_amplitudes(model, scale, recipe, extent):
    """Training quadrature nodes, weights (m^3) and each node's signed sqrt(RCS) (m)."""
    from rift.spinr_style import gauss_legendre_cell_grid
    nodes, weights = gauss_legendre_cell_grid(recipe['grid_size'], nodes_per_cell=recipe['nodes_per_cell'],
                                              support_m=extent, dtype=torch.float64)
    field = _field(model, nodes, recipe.get('neural_point_tile', 4096))
    return nodes, weights, 4.0 * scale * weights * field / K


def spinr_presentation(checkpoint_path, *, shard_root=None, polarization='hh', grid=PRESENTATION_GRID):
    from rift.spinr_fidelity import field_readout
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    recipe = checkpoint['recipe']
    model, scale, extent = load_head(checkpoint, polarization)
    _, sigma = field_readout(model, grid_size=grid, support_m=extent, initial_output_scale=scale,
                             neural_point_tile=recipe.get('neural_point_tile', 4096), device='cpu')
    relative = sigma.square().reshape(grid, grid, grid).numpy()
    source = dict(checkpoint=str(Path(checkpoint_path).resolve()), epoch=checkpoint.get('epoch'),
                  cursor=checkpoint.get('cursor'), updates=checkpoint.get('updates'),
                  best_validation=checkpoint.get('best_val'), best_epoch=checkpoint.get('best_epoch'),
                  initial_output_scale=scale, quadrature=dict(grid_size=recipe['grid_size'],
                                                              nodes_per_cell=recipe['nodes_per_cell']),
                  half_extent_m=extent)
    rcs, physical_quantity = None, None
    if polarization == 'hh':
        nodes, _, amplitude = node_amplitudes(model, scale, recipe, extent)
        rcs = deposit_points(nodes, amplitude, extent, grid) ** 2
        physical_quantity = ('RCS per G48 cell (m^2): square of the summed signed node amplitudes; '
                             'isotropic, so identical for every look direction')
        source.update(conversion='sqrt(RCS_node) = 4 s w field / K (kernel 1/d^2 vs K (2d)^2)',
                      look_directions='not needed: position-only field, range-independent conversion')
    return MethodPresentation('spinr', polarization, relative,
                              'SpINR readout sigma^2, sigma = initial_output_scale * field at G48 midpoints '
                              '(arbitrary units)', rcs, physical_quantity, source)
