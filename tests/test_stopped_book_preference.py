import torch
from scripts.train_pointing_causal_writer import whole_reply_preference


def test_control_preserves_gold_update_and_removes_rival_derivative():
    scores = torch.tensor(
        [-7.0, -2.0, -4.0, -3.0, -5.0], dtype=torch.double, requires_grad=True
    )
    candidate = whole_reply_preference(scores, [0, 2], True)
    control = whole_reply_preference(scores, [0, 2], False)
    assert candidate.item() == control.item()
    grad = torch.autograd.grad(candidate, scores)[0]
    other = torch.autograd.grad(control, scores)[0]
    torch.testing.assert_close(grad[:3], other[:3])
    assert grad[1] == 0 and torch.all(grad[[0, 2]] < 0) and torch.all(grad[3:] > 0)
    assert not other[3:].any()
    torch.testing.assert_close(grad[[0, 2]], -grad[3:])
