import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from chrono.losses import cka_loss, hsic_loss  # noqa: E402


def test_cka_is_scale_free_and_bounded():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(200, 16, generator=g); z = torch.nn.functional.one_hot(torch.randint(0, 5, (200,), generator=g), 5).float()
    a, b = cka_loss(x, z), cka_loss(10 * x, z)
    assert 0 <= float(a) <= 1 and abs(float(a) - float(b)) < 1e-4
    assert abs(float(cka_loss(x, x)) - 1) < 1e-4


def test_cka_detects_dependence_hsic_barely_does():
    g = torch.Generator().manual_seed(1)
    z_idx = torch.randint(0, 5, (256,), generator=g); z = torch.nn.functional.one_hot(z_idx, 5).float()
    x_dep = torch.randn(256, 32, generator=g) + 2.0 * torch.nn.functional.one_hot(z_idx, 32)[:, :32].float()
    x_ind = torch.randn(256, 32, generator=g)
    assert float(cka_loss(x_dep, z)) > 5 * float(cka_loss(x_ind, z))
    # the raw HSIC magnitudes are tiny: this is why lambda 1..10 did nothing in C6
    assert float(hsic_loss(x_dep, z)) < 0.1
