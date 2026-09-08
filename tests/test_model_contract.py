"""The fixed architecture, checked against written-down numbers.

These expectations are literals, deliberately not read back from MODEL_CONFIG. A test that
recomputes its own expectation from the value under test cannot detect a change to that value,
which is the whole point of pinning a published architecture.

The numbers come from the article's description of HeadLite and were confirmed by measuring the
implementation: three valid convolutions of kernel 30 take the 300-point stride grid to 271, 242
and 213; the three branch projections and the metadata encoder give 3 x 128 + 64 = 448.
"""
import torch

from headlite import MODEL_CONFIG, HeadLite

PARAMETERS = 1_732_146
FIVE_NETWORKS = 8_660_730
STATE_ENTRIES = 115
PARAMETER_TENSORS = 76
BUFFERS = 39
DROPOUT_P = 0.21469034765110603
DROPOUT_MODULES = ("dense_mean.3", "dense_mean.7")
CONV_TAIL = 213
FUSION_WIDTH = 448

EXPECTED_CONFIG = {
    "acc_gyr_dim1": 32,
    "prs_dim1": 48,
    "dense_dim": 256,
    "dropout_rate": DROPOUT_P,
    "meta_feature_dim": 6,
    "meta_encode_dim": 64,
    "proj_dim": 128,
}

CONV_SHAPES = {
    "conv1d_acc.0.weight": (32, 3, 30),
    "conv1d_acc.3.weight": (64, 32, 30),
    "conv1d_acc.6.weight": (128, 64, 30),
    "conv1d_gyr.0.weight": (32, 3, 30),
    "conv1d_gyr.3.weight": (64, 32, 30),
    "conv1d_gyr.6.weight": (128, 64, 30),
    "conv1d_prs.0.weight": (48, 40, 30),
    "conv1d_prs.3.weight": (96, 48, 30),
    "conv1d_prs.6.weight": (192, 96, 30),
}

HEAD_SHAPES = {
    "dense_mean.0.weight": (256, FUSION_WIDTH),
    "dense_mean.4.weight": (128, 256),
    "dense_mean.8.weight": (1, 128),
    "dense_var.0.weight": (256, FUSION_WIDTH),
    "dense_var.3.weight": (128, 256),
    "dense_var.6.weight": (1, 128),
    "meta_encoder.net.0.weight": (64, 6),
    "meta_encoder.net.3.weight": (64, 64),
    "acc_proj.0.weight": (128, 128),
    "gyr_proj.0.weight": (128, 128),
    "prs_proj.0.weight": (128, 192),
}


def inputs(batch=2):
    return (torch.randn(batch, 3, 300), torch.randn(batch, 3, 300),
            torch.rand(batch, 40, 300), torch.randn(batch, 6))


def test_configuration_matches_the_published_values():
    assert dict(MODEL_CONFIG) == EXPECTED_CONFIG


def test_dropout_probability_is_the_exact_published_value():
    """The article rounds this to two places; the implementation keeps the full value."""
    assert MODEL_CONFIG["dropout_rate"] == DROPOUT_P
    assert round(DROPOUT_P, 2) == 0.21


def test_dropout_sits_only_in_the_mean_head():
    model = HeadLite()
    found = [(name, module.p) for name, module in model.named_modules()
             if isinstance(module, torch.nn.Dropout)]
    assert tuple(name for name, _ in found) == DROPOUT_MODULES
    assert all(p == DROPOUT_P for _, p in found)
    assert not [n for n, _ in found if n.startswith("dense_var")]


def test_parameter_and_state_counts():
    model = HeadLite()
    assert sum(p.numel() for p in model.parameters()) == PARAMETERS
    assert 5 * PARAMETERS == FIVE_NETWORKS
    assert len(model.state_dict()) == STATE_ENTRIES
    assert len(list(model.named_parameters())) == PARAMETER_TENSORS
    assert len(list(model.named_buffers())) == BUFFERS
    assert PARAMETER_TENSORS + BUFFERS == STATE_ENTRIES


def test_branch_and_head_shapes():
    state = HeadLite().state_dict()
    for key, shape in {**CONV_SHAPES, **HEAD_SHAPES}.items():
        assert key in state, f"{key} is missing from the state dictionary"
        assert tuple(state[key].shape) == shape, f"{key}: {tuple(state[key].shape)} != {shape}"


def test_convolution_tail_and_fusion_width():
    """Measured, not assumed: the tail length and the fused width come out of a forward pass."""
    model = HeadLite().eval()
    tails = []
    handles = [module.register_forward_hook(
        lambda mod, inp, out, store=tails: store.append(out.shape[-1]) or None)
        for module in model.modules() if isinstance(module, torch.nn.Conv1d)]
    with torch.inference_mode():
        _, _, embedding = model(*inputs(), return_embedding=True)
    for handle in handles:
        handle.remove()
    assert tails == [271, 242, 213] * 3, tails
    assert tails[2] == CONV_TAIL
    assert embedding.shape[-1] == FUSION_WIDTH


def test_variance_head_ends_in_softplus():
    model = HeadLite().eval()
    assert isinstance(model.dense_var[-1], torch.nn.Softplus)
    with torch.inference_mode():
        _, variance = model(*inputs())
    assert torch.isfinite(variance).all()
    assert (variance > 0).all(), (
        "Softplus is positive for every finite input here; this is an observation on this input, "
        "not a guarantee that no input can underflow to zero")


def test_training_mode_needs_more_than_one_sample():
    """BatchNorm cannot compute batch statistics from a single sample."""
    import pytest
    model = HeadLite().train()
    with pytest.raises(ValueError, match="two samples"):
        model(*inputs(1))
    mean, variance = model(*inputs(2))
    assert mean.shape == variance.shape == (2, 1)


def test_evaluation_mode_accepts_one_sample():
    model = HeadLite().eval()
    with torch.inference_mode():
        mean, variance = model(*inputs(1))
    assert mean.shape == variance.shape == (1, 1)
