import numpy as np
import torch
from safetensors.torch import save_file
from ilm.visual_lm.pictured_revision import PicturedReviser


def test_unknown_draft_ink_and_actual_deletion(tmp_path):
    bank = torch.tensor([[0] * 128, [73] * 128, [149] * 128], dtype=torch.uint8)
    options = torch.tensor([[1, -1], [2, 0], [3, 0]], dtype=torch.int32)
    model = PicturedReviser(
        bank,
        torch.arange(48),
        options,
        torch.tensor([[0] * 4, [0] * 4, [1, 0, 0, 0], [2, 0, 0, 0]]),
        torch.tensor([0, 1, 1, 1]),
        torch.zeros(3, 2),
    )
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        model.change.bias.fill_(2)
    path = tmp_path / "reviser.safetensors"
    save_file(model.state_dict(), str(path))
    new = PicturedReviser.restore(path)
    unseen = np.full((1, 128), 77, np.uint8)
    observed = bank[1:].numpy()
    np.testing.assert_array_equal(
        new.revise(unseen, np.concatenate([observed, unseen])), unseen
    )
    assert new.revise(unseen, observed).shape == (0, 128)
    assert new.revise(unseen, observed[:0]).shape == (0, 128)
    new.margin.fill_(3)
    np.testing.assert_array_equal(new.revise(unseen, observed), observed)
