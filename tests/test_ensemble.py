import torch
import pytest
from headlite import HeadLite, HeadLiteEnsemble


class Constant(torch.nn.Module):
    def __init__(self, value):
        super().__init__(); self.register_buffer("value", torch.tensor(float(value)))
    def forward(self, acc, gyr, prs, meta):
        mean = self.value.expand(acc.shape[0], 1)
        return mean, torch.ones_like(mean)


class Wrong(torch.nn.Module):
    """A member that returns something the wrapper must refuse."""
    def __init__(self, value, mode):
        super().__init__()
        self.register_buffer("value", torch.tensor(float(value)))
        self.mode = mode
    def forward(self, acc, gyr, prs, meta):
        b = acc.shape[0]
        if self.mode == "one_row":            # ignores the batch entirely
            m = self.value.reshape(1, 1)
        elif self.mode == "three_dim":
            m = self.value.reshape(1, 1, 1).expand(b, 1, 1)
        elif self.mode == "wide":
            m = self.value.expand(b, 2)
        elif self.mode == "nan":
            m = torch.full((b, 1), float("nan"))
        elif self.mode == "inf":
            m = torch.full((b, 1), float("inf"))
        elif self.mode == "integer":
            return torch.zeros(b, 1, dtype=torch.long), torch.ones(b, 1)
        elif self.mode == "bare":
            return self.value.expand(b, 1)     # not a (mean, variance) pair
        elif self.mode == "dict":
            return {"mean": self.value.expand(b, 1)}
        else:
            raise AssertionError(self.mode)
        return m, torch.ones(b, 1)


def inputs(batch=3):
    return (torch.randn(batch, 3, 300), torch.randn(batch, 3, 300),
            torch.rand(batch, 40, 300), torch.randn(batch, 6))


def test_prediction_average():
    e = HeadLiteEnsemble([Constant(i) for i in range(1, 6)]).eval()
    x = torch.zeros(3, 1)
    y = e(x, x, x, x)
    assert y.shape == (3, 1)
    assert torch.equal(y, torch.full((3, 1), 3.0))
    assert not any(m.training for m in e.models)


@pytest.mark.parametrize("count", [0, 1, 4, 6])
def test_wrong_size(count):
    with pytest.raises(ValueError, match="five"):
        HeadLiteEnsemble([Constant(i) for i in range(count)])


def test_duplicate_object():
    m = Constant(1)
    with pytest.raises(ValueError, match="same model"):
        HeadLiteEnsemble([m] * 5)


def test_distinct_objects_with_equal_weights_are_allowed():
    """Equal weights alone do not make a member wrong, so this stays accepted."""
    e = HeadLiteEnsemble([Constant(2) for _ in range(5)]).eval()
    x = torch.zeros(3, 1)
    assert torch.equal(e(x, x, x, x), torch.full((3, 1), 2.0))


def test_members_that_ignore_the_batch_are_refused():
    """Five members that each return one row agree with one another, so comparing members to
    each other accepts them; the wrapper then returns one row for a three-sample input."""
    e = HeadLiteEnsemble([Wrong(i, "one_row") for i in range(1, 6)]).eval()
    with pytest.raises(ValueError, match="row"):
        e(*inputs(3))


@pytest.mark.parametrize("mode,exc,match", [
    ("three_dim", ValueError, r"\(B, 1\)"),
    ("wide", ValueError, r"\(B, 1\)"),
    ("nan", ValueError, "non-finite"),
    ("inf", ValueError, "non-finite"),
    ("integer", TypeError, "floating"),
    ("bare", TypeError, "bare tensor"),
    ("dict", TypeError, "first element is the mean"),
])
def test_malformed_member_output_is_refused(mode, exc, match):
    e = HeadLiteEnsemble([Wrong(i, mode) for i in range(1, 6)]).eval()
    with pytest.raises(exc, match=match):
        e(*inputs(3))


def test_one_bad_member_among_four_good_ones_is_refused():
    members = [Constant(1), Constant(2), Wrong(3, "one_row"), Constant(4), Constant(5)]
    e = HeadLiteEnsemble(members).eval()
    with pytest.raises(ValueError, match="member 2"):
        e(*inputs(3))


def test_non_tensor_input_is_refused():
    e = HeadLiteEnsemble([Constant(i) for i in range(1, 6)]).eval()
    with pytest.raises(TypeError, match="acc"):
        e([[0.0]], None, None, None)


@pytest.mark.parametrize("batch", [1, 2, 5])
def test_real_models_average_at_several_batch_sizes(batch):
    """Control: five actual HeadLite networks, and the mean matches a direct average."""
    torch.manual_seed(11)
    members = [HeadLite().eval() for _ in range(5)]
    e = HeadLiteEnsemble(members).eval()
    xs = inputs(batch)
    with torch.inference_mode():
        got = e(*xs)
        direct = torch.stack([m(*xs)[0] for m in members], dim=0).mean(dim=0)
    assert got.shape == (batch, 1)
    assert torch.isfinite(got).all()
    assert torch.equal(got, direct)


def test_float64_members_are_accepted():
    """The dtype is required to be floating, not to equal the input dtype."""
    class Double(torch.nn.Module):
        def __init__(self, v):
            super().__init__(); self.v = float(v)
        def forward(self, acc, gyr, prs, meta):
            m = torch.full((acc.shape[0], 1), self.v, dtype=torch.float64)
            return m, torch.ones_like(m)
    e = HeadLiteEnsemble([Double(i) for i in range(1, 6)]).eval()
    out = e(*inputs(3))
    assert out.dtype == torch.float64
    assert torch.equal(out, torch.full((3, 1), 3.0, dtype=torch.float64))


class Shaped(torch.nn.Module):
    """A member that returns a well-formed mean wrapped however the caller asks."""
    def __init__(self, value, wrap):
        super().__init__()
        self.register_buffer("value", torch.tensor(float(value)))
        self.wrap = wrap
    def forward(self, acc, gyr, prs, meta):
        return self.wrap(self.value.expand(acc.shape[0], 1))


@pytest.mark.parametrize("wrap", [
    lambda m: (m,),                                   # one-element tuple
    lambda m: [m],                                    # one-element list
    lambda m: (m, torch.ones_like(m), "not read"),    # extra elements, one not even a tensor
], ids=["one_element_tuple", "one_element_list", "extra_elements"])
def test_only_the_first_element_of_the_return_value_is_read(wrap):
    """The documented contract: a non-empty tuple or list, whose first element is the mean.

    Later elements are neither read nor validated, so a member is free to return just the mean.
    """
    e = HeadLiteEnsemble([Shaped(i, wrap) for i in range(1, 6)]).eval()
    out = e(*inputs(3))
    assert out.shape == (3, 1)
    assert torch.equal(out, torch.full((3, 1), 3.0))


def test_one_element_tuple_gives_the_same_result_as_a_mean_variance_pair():
    """Dropping the variance changes nothing about the prediction, because it is never read."""
    pairs = HeadLiteEnsemble([Constant(i) for i in range(1, 6)]).eval()
    singles = HeadLiteEnsemble([Shaped(i, lambda m: (m,)) for i in range(1, 6)]).eval()
    xs = inputs(4)
    assert torch.equal(singles(*xs), pairs(*xs))


def test_a_one_element_tuple_is_still_checked_against_the_batch():
    """Accepting the shorter form does not skip the checks on the mean itself."""
    e = HeadLiteEnsemble([Shaped(i, lambda m: (m[:1],)) for i in range(1, 6)]).eval()
    with pytest.raises(ValueError, match="row"):
        e(*inputs(3))


@pytest.mark.parametrize("empty", [tuple, list], ids=["tuple", "list"])
def test_empty_return_value_is_refused(empty):
    """There is no first element to read, so this cannot be treated as a mean."""
    e = HeadLiteEnsemble([Shaped(i, lambda m, e=empty: e()) for i in range(1, 6)]).eval()
    with pytest.raises(TypeError, match="empty"):
        e(*inputs(3))
