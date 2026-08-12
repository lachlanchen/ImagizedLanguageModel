from __future__ import annotations

import torch

from scripts.train_factorized_visual_context_v26 import VisualTargetQueue


def test_visual_target_queue_is_fifo_and_detached() -> None:
    queue = VisualTargetQueue(4, 3, device=torch.device("cpu"))
    first = torch.arange(9, dtype=torch.float32).reshape(3, 3).requires_grad_()
    queue.update(first)
    assert queue.count == 3
    candidates = queue.candidates(torch.full((1, 3), 99.0))
    assert candidates.shape == (4, 3)
    assert candidates.requires_grad is False

    queue.update(torch.tensor([[9.0, 10.0, 11.0], [12.0, 13.0, 14.0]]))
    assert queue.count == 4
    assert queue.pointer == 1
    state = queue.state_dict()
    restored = VisualTargetQueue(4, 3, device=torch.device("cpu"))
    restored.load_state_dict(state)
    assert restored.count == queue.count
    assert restored.pointer == queue.pointer
    assert torch.equal(restored.storage, queue.storage)


def test_visual_target_queue_large_update_keeps_latest() -> None:
    queue = VisualTargetQueue(3, 2, device=torch.device("cpu"))
    values = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    queue.update(values)
    assert queue.count == 3
    assert queue.pointer == 0
    assert torch.equal(queue.storage, values[-3:])
