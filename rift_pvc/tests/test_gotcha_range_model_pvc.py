"""PVC batched GOTCHA kernel with the declared amplitude law: exact against autograd of the per-pulse renderer."""
import pytest
import torch

import train_gotcha_dataset_pvc as cli
from rift_pvc import gotcha_training as pvc
from rift_pvc.gotcha_batched import batched_native_forward

ARGV = ['--granularity', '2', '--max-points', '15', '--sh-degree', '1']


def problem(seed=0):
    g = torch.Generator().manual_seed(seed)
    points = (torch.rand(7, 3, generator=g, dtype=torch.float64) - .5) * .06
    weights = torch.complex(torch.randn(3, 7, generator=g, dtype=torch.float64),
                            torch.randn(3, 7, generator=g, dtype=torch.float64))
    antennas = torch.tensor([[20., 1., 2.], [19., 4., 3.], [-15., 12., 6.]], dtype=torch.float64)
    refs = torch.linalg.vector_norm(antennas, dim=-1) + torch.tensor([.3, -.2, .1], dtype=torch.float64)
    freqs = torch.linspace(9e9, 10e9, 16, dtype=torch.float64)
    freqs[1::2] += 128
    target = torch.complex(torch.randn(3, 16, generator=g, dtype=torch.float64),
                           torch.randn(3, 16, generator=g, dtype=torch.float64))
    return points, weights, antennas, refs, freqs, target


def loss(out, target):
    return (out - target).abs().square().sum()


@pytest.mark.parametrize('range_model', ['sum2', 'unit'])
def test_batched_forward_and_backward_match_per_pulse_autograd(range_model):
    points, weights, antennas, refs, freqs, target = problem()
    scale = 1e5 if range_model == 'sum2' else 1.   # keep the fit residual comparable to the target
    target = target / scale
    x_loop, w_loop = points.clone().requires_grad_(), weights.clone().requires_grad_()
    outs = [pvc.native_forward(x_loop, w_loop[p], antennas[p], freqs, float(refs[p]), point_chunk=3,
                               range_model=range_model) for p in range(3)]
    per_pulse_x = [torch.autograd.grad(loss(o, t), x_loop, retain_graph=True)[0] for o, t in zip(outs, target)]
    loss(torch.stack(outs), target).backward()

    x, w = points.clone().requires_grad_(), weights.clone().requires_grad_()
    pulse_grad_d = torch.zeros(3, 7, dtype=torch.float64)
    out = batched_native_forward(x, w, antennas, refs, freqs, point_chunk=3, pulse_grad_d=pulse_grad_d,
                                 range_model=range_model)
    loss(out, target).backward()

    torch.testing.assert_close(out.detach(), torch.stack(outs).detach(), rtol=1e-12, atol=0)
    torch.testing.assert_close(w.grad, w_loop.grad, rtol=1e-10, atol=0)
    torch.testing.assert_close(x.grad, x_loop.grad, rtol=1e-10, atol=1e-12*float(x_loop.grad.abs().max()))
    unit = points[None] - antennas[:, None]
    unit = unit / torch.linalg.vector_norm(unit, dim=-1, keepdim=True)
    for p in range(3):
        torch.testing.assert_close(pulse_grad_d[p, :, None] * unit[p], per_pulse_x[p], rtol=1e-10,
                                   atol=1e-12*float(per_pulse_x[p].abs().max()))


def test_sum2_differs_from_unit_by_the_physical_range_law():
    points, weights, antennas, refs, freqs, _ = problem()
    sum2 = batched_native_forward(points, weights, antennas, refs, freqs, point_chunk=4, range_model='sum2')
    unit = batched_native_forward(points, weights * (1 / ((4*torch.pi)**2 * ((2*torch.linalg.vector_norm(
        points[None] - antennas[:, None], dim=-1))**2 + 1e-9))), antennas, refs, freqs, point_chunk=4)
    torch.testing.assert_close(sum2, unit, rtol=1e-12, atol=0)
    with pytest.raises(ValueError):
        batched_native_forward(points, weights, antennas, refs, freqs, point_chunk=4, range_model='product')


def test_pvc_frontend_declares_the_same_law(monkeypatch):
    monkeypatch.setenv('RIFT_ACCELERATOR', 'cpu')
    for method in ('rift', 'rift_grid', 'isotropic'):
        assert pvc.recipe_from_args(cli.parse_args(ARGV), method)['range_model'] == 'sum2'
    assert 'range_model' not in pvc.recipe_from_args(cli.parse_args(ARGV), 'mfbp')
    assert pvc.recipe_from_args(cli.parse_args(ARGV + ['--range-model', 'unit']), 'rift')['range_model'] == 'unit'
