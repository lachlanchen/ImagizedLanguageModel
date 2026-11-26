"""Replay both decoding rules on the original held image with released weights."""

from hashlib import sha256
from pathlib import Path

import torch

from ilm.visual_lm.portable_writer import read_image
from scripts.write_ordered_translation import infer, restore


def test_order_decoder_reproduces_held_gpu_ink_from_png():
    torch.set_num_threads(2)
    raw = Path(__file__).resolve().parents[1] / "publication/ilm-nature/figures/raw_ilm"
    cells = read_image(raw / "natural-en-input.png", 22)
    model = restore("paired-assignment", "cpu")
    greedy, stop = infer(model, cells, 0, "cpu")
    assert stop and greedy == read_image(raw / "natural-en-output.png", 88).tobytes()
    ordered, stop = infer(model, cells, 0, "cpu", "order")
    assert stop and len(ordered) == 82 * 128
    assert sha256(ordered).hexdigest() == (
        "8549c476636eef4cf41728c541b824e481d4835e5425fac84f985ef60fa2c397"
    )
