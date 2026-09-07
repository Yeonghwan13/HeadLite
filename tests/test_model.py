import torch
import pytest
from headlite import HeadLite, MODEL_CONFIG


def inputs(batch=2):
    return (torch.randn(batch, 3, 300), torch.randn(batch, 3, 300),
            torch.rand(batch, 40, 300), torch.randn(batch, 6))


def test_contract():
    m = HeadLite()
    assert sum(p.numel() for p in m.parameters()) == 1_732_146
    assert len(m.state_dict()) == 115
    assert len(list(m.parameters())) == 76
    assert len(list(m.buffers())) == 39
    assert m.combined_dim == 448
    d = [(n, x.p) for n, x in m.named_modules() if isinstance(x, torch.nn.Dropout)]
    assert d == [("dense_mean.3", MODEL_CONFIG["dropout_rate"]),
                 ("dense_mean.7", MODEL_CONFIG["dropout_rate"])]


@pytest.mark.parametrize("batch", [1, 2, 4])
def test_forward(batch):
    m = HeadLite().eval()
    with torch.inference_mode():
        mean, var, emb = m(*inputs(batch), return_embedding=True)
    assert mean.shape == var.shape == (batch, 1)
    assert emb.shape == (batch, 448)
    assert torch.isfinite(mean).all() and torch.isfinite(var).all()
    assert (var >= 0).all()


def test_branch_lengths():
    m = HeadLite().eval(); a, g, p, _ = inputs()
    with torch.inference_mode():
        assert m.conv1d_acc(a).shape == (2, 128, 213)
        assert m.conv1d_gyr(g).shape == (2, 128, 213)
        assert m.conv1d_prs(p).shape == (2, 192, 213)


def test_seed_before_construction():
    torch.manual_seed(77); a = HeadLite()
    torch.manual_seed(77); b = HeadLite()
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())


def test_backward_and_dropouts():
    m = HeadLite().train(); calls = []
    handles = [x.register_forward_hook(lambda *args, n=n: calls.append(n))
               for n, x in m.named_modules() if isinstance(x, torch.nn.Dropout)]
    try:
        mean, var = m(*inputs(4))
        (mean.square().mean() + var.mean()).backward()
        assert calls == ["dense_mean.3", "dense_mean.7"]
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    finally:
        for h in handles: h.remove()


def test_train_batch_one_rejected():
    with pytest.raises(ValueError, match="BatchNorm"):
        HeadLite()(*inputs(1))


@pytest.mark.parametrize("problem", ["length", "channels", "meta", "batch", "nan", "dtype", "empty"])
def test_invalid_inputs(problem):
    a, g, p, m = inputs()
    if problem == "length": a = a[:, :, :-1]
    elif problem == "channels": g = g[:, :2]
    elif problem == "meta": m = m[:, :5]
    elif problem == "batch": p = p[:1]
    elif problem == "nan": a[0, 0, 0] = float("nan")
    elif problem == "dtype": g = g.double()
    elif problem == "empty": a, g, p, m = a[:0], g[:0], p[:0], m[:0]
    with pytest.raises(ValueError): HeadLite().eval()(a, g, p, m)


def test_state_dict_roundtrip(tmp_path):
    a = HeadLite().eval(); x = inputs()
    path = tmp_path / "generated_weights.pt"
    torch.save(a.state_dict(), path)
    b = HeadLite().eval()
    b.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    with torch.inference_mode():
        aa, av = a(*x); bb, bv = b(*x)
    assert torch.equal(aa, bb) and torch.equal(av, bv)
