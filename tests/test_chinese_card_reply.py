import json

import numpy as np
import pytest
import torch

from scripts.prepare_chinese_card_reply import FIELDS, MODEL, PUB, Rasterizer
from scripts.learn_chinese_focus import mask_mentions
from scripts.write_chinese_card_reply import ChineseReply, load_page, pack, transcript


@pytest.fixture(scope="module")
def reader():
    torch.set_num_threads(2)
    return ChineseReply("cpu", focus=True)


def test_two_human_questions_change_only_the_prompt_and_reproduce_original_ink(reader):
    examples = MODEL / "examples"
    records = json.loads((PUB / "examples.json").read_text())
    first, second = (load_page(examples / f"question-{s}-card.png") for s in ("a", "b"))
    np.testing.assert_array_equal(first[1:], second[1:])
    assert not np.array_equal(first[0], second[0])
    for key, page in (("question-a", first), ("question-b", second)):
        field, output = reader.reply(page)
        assert field == records[key]["key"]
        assert output.tobytes() == (examples / f"{key}-answer.bin").read_bytes()
        assert transcript(output) == records[key]["generated"]


def test_interchanging_the_roles_preserves_focus_context_but_changes_the_emitted_value(
    reader,
):
    page = load_page(MODEL / "examples/question-b-card.png")
    raster = Rasterizer()
    normalized = []
    for target, other, value in (
        ("营业时间", "地址", "11:00-21:30"),
        ("地址", "营业时间", "西四南大街60号"),
    ):
        page[0] = pack(f"我想查{target}，不是{other}。", raster)
        view, positions, _ = mask_mentions(page[0], [True] * len(FIELDS))
        normalized.append((view, positions))
        selected, ink = reader.reply(page)
        assert selected == target
        assert value in transcript(ink)
    np.testing.assert_array_equal(normalized[0][0], normalized[1][0])
    assert normalized[0][1] == normalized[1][1]


def test_duplicate_record_field_is_rejected(reader):
    page = load_page(MODEL / "examples/question-a-card.png")
    with pytest.raises(ValueError, match="repeated field"):
        reader.reply(np.concatenate((page, page[-1:])))
