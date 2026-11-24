import numpy as np
from PIL import Image
import pytest
import torch

from ilm.visual_lm.reference_order import cue
from scripts.prepare_story_mentions import MODEL, documents
from scripts.story_mention_options import collapse, select
from scripts.write_masked_mentions import cast_cells, cast_options
from scripts.write_prose_fields import load_png
from scripts.write_referred_fact import ReferredFact


def test_query_overlap_and_repeated_spellings():
    indices, slots, spans, query = collapse("ABC A ABC A", (6, 9), [(0, 3), (4, 5)])
    assert [spans[i] for i, tag in enumerate(slots) if tag >= 0] == [
        (0, 3),
        (4, 5),
        (10, 11),
    ]
    assert [indices[query[0]], indices[query[1]]] == [6, 9]
    # Two individually weaker occurrences can outvote the strongest occurrence.
    assert select(
        np.log([0.3, 0.4, 0.3, 0.02]), [0, 1, 0], [(0, 1), (2, 3), (4, 5)], 3
    ) == (0, 1)


def test_unmarked_and_padded_name_pixels_are_hidden():
    values = torch.arange(2 * 6 * 3, dtype=torch.float32).reshape(2, 6, 3)
    slots = torch.tensor([[-1, 0, -1, 1, -1, -1], [-1, 0, -1, 1, -1, -1]])
    q = torch.tensor([[2, 3], [2, 3]])
    patch = cue(values, q, slots, torch.tensor([4, 6]))
    torch.testing.assert_close(patch[:, 0], values[:, 2])
    assert not patch[:, 8].any()
    assert not patch[0, 9:].any()
    assert not cue(values, q, slots, torch.tensor([4, 6]), True)[:, :8].any()


def test_cast_accepts_unencoded_source_marks():
    source = np.random.default_rng(92).integers(0, 256, (7, 128), dtype=np.uint8)
    panel = cast_cells([source[2:5], source[0:1]])
    assert cast_options(panel, source, 5) == [(2, 5), (0, 1)]
    with pytest.raises(ValueError):
        cast_options(Image.new("L", (16, 32)), source, 5)
    with pytest.raises(ValueError, match="document identifier"):
        list(documents("# text = An unidentified work."))


@pytest.fixture(scope="module")
def writer():
    torch.set_num_threads(2)
    return ReferredFact()


@pytest.mark.parametrize("index,bounds", [(1, (12, 22)), (2, (22, 29))])
def test_frozen_fee_images(writer, index, bounds):
    folder = MODEL / "examples" / f"fee-{index}"
    with Image.open(folder / "cast.png") as panel:
        result, region = writer.answer(load_png(folder / "input.png"), panel)
    assert region == bounds
    assert result["pixels"].tobytes() == (folder / "referred-answer.bin").read_bytes()


def test_ambiguous_bindings_leave_full_paragraph(writer):
    page = load_png(MODEL / "examples" / "fee-1" / "input.png")
    candidate = b"identical"
    kept, region = writer.restrict(page, [(0, 5, candidate), (6, 10, candidate)])
    assert kept is page and region is None
