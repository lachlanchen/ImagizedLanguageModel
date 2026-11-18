import numpy as np

from ilm.visual_lm.visual_phrase_composer import VisualPhraseComposer
from scripts.infer_visual_translation import page
from scripts.train_aligned_visual_translation import blocks


def test_alignment_preserves_inserted_and_deleted_cells():
    for source, target in [
        ([3, 4], [1, 2, 3, 4, 5, 6]),
        ([1, 2, 3], [1, 3]),
        ([3, 4], []),
    ]:
        assert sum(blocks(source, target), []) == target


def test_unknown_ink_survives_alongside_learned_phrase(tmp_path):
    cells = np.stack([np.full(128, i, np.uint8) for i in (12, 73, 94)])
    pairs = [((0, 1), (2, 1))] * 4
    model = VisualPhraseComposer.fit(cells, pairs, blocks)
    path = tmp_path / "weights.gz"
    model.save(path)
    restored = VisualPhraseComposer.restore(path)
    new = np.full(128, 55, np.uint8)
    actual = restored.generate(np.stack([*cells[:2], new]))
    np.testing.assert_array_equal(actual, [cells[2], cells[1], new])
    assert page(actual).size == (96, 32)


def test_paired_image_training_and_transparent_output(tmp_path):
    import json
    from scripts.fit_visual_translator import train

    source = np.stack([np.full(128, v, np.uint8) for v in (15, 30)])
    target = source.copy()
    target[0] = 60
    page(source).save(tmp_path / "source.png")
    page(target).save(tmp_path / "target.png")
    record = {"source_image": "source.png", "target_image": "target.png"}
    rows = tmp_path / "rows.jsonl"
    rows.write_text((json.dumps(record) + "\n") * 4)
    weights = tmp_path / "weights.gz"
    train(rows, "target", weights, 0.25)
    np.testing.assert_array_equal(
        VisualPhraseComposer.restore(weights).generate(source), target
    )


def test_block_writer_emits_empty_and_copies_unpadded_positions():
    import torch
    from ilm.visual_lm.block_translation import BlockTranslator

    torch.manual_seed(930)
    model = BlockTranslator(torch.arange(48)).eval()
    pictures = torch.rand(2, 3, 1024) > 0.8
    lengths = torch.tensor([3, 1])
    with torch.no_grad():
        model.length.weight.zero_()
        model.length.bias.zero_()
        model.length.bias[0] = 1.0
        assert all(len(row) == 0 for row in model.generate(pictures, lengths))
        model.length.bias[1] = 2.0
        model.copy.weight.zero_()
        model.copy.bias.fill_(1.0)
        for reverse in (True, False):
            outputs = model.generate(pictures, lengths, reverse_context=reverse)
            for i, row in enumerate(outputs):
                torch.testing.assert_close(row, pictures[i, : lengths[i]])
