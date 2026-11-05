"""Check physical image decoding and prompt/partition isolation."""

import hashlib
import json

import numpy as np
import pytest
import torch

from ilm.visual_lm.complete_writing_data import CompleteWritingImageData


def make_cache(tmp_path, *, development=False, corrupt=None):
    # A codec fixture, not language data or a model-quality experiment.
    white = np.ones((32, 320), dtype=np.uint8)
    white[5:25, 10:25] = 0
    white[8:24, 145:150] = 0
    if corrupt == "termination":
        white[-1, -1] = 0
    bits = torch.from_numpy(np.packbits(white.reshape(-1)))
    if corrupt == "bits":
        bits = bits[:-1]
    folder = tmp_path/"cmrc"
    folder.mkdir()
    name = folder/"fixture.pt"
    torch.save(dict(bits=bits, width=320, prompt_blocks=2), name)
    report = dict(training_only=not development, evaluation_only=development,
        split="development" if development else "train", all_retained_rows_lossless=True,
        files_sha256={"cmrc/fixture.pt": hashlib.sha256(name.read_bytes()).hexdigest()},
        summaries={"cmrc": {"counts": {"accepted": 1}}}, maximum_prompt_blocks=240,
        maximum_answer_blocks=128, termination_blocks=2)
    (tmp_path/"report.json").write_text(json.dumps(report))
    return white


@pytest.mark.parametrize("development", [False, True])
def test_prompt_owns_only_its_pixels_and_does_not_mutate_cached_target(tmp_path, development):
    white = make_cache(tmp_path, development=development)
    data = CompleteWritingImageData(tmp_path, purpose="development" if development else "train")
    pixels, boundary = data[0]
    assert boundary == 2 and pixels.shape == (1, 1, 32, 320)
    np.testing.assert_array_equal(pixels.numpy()[0, 0], white.astype(np.float32)*2-1)
    prompt, target = data.prompt(0), data.target(0)
    assert prompt.shape[-1] == 128 and target.shape[-1] == 192
    assert prompt.untyped_storage().nbytes() == prompt.numel()*prompt.element_size()
    assert target.untyped_storage().nbytes() == target.numel()*target.element_size()
    assert prompt.untyped_storage().data_ptr() != target.untyped_storage().data_ptr()
    prompt.fill_(0)
    pixels.fill_(0)
    np.testing.assert_array_equal(data.prompt(0).numpy()[0, 0], white[:, :128].astype(np.float32)*2-1)
    np.testing.assert_array_equal(data.target(0).numpy()[0, 0], white[:, 128:].astype(np.float32)*2-1)
    wrong_purpose = "train" if development else "development"
    with pytest.raises(ValueError):
        CompleteWritingImageData(tmp_path, purpose=wrong_purpose)


@pytest.mark.parametrize("corrupt", ["bits", "termination"])
def test_reject_malformed_physical_measurements(tmp_path, corrupt):
    make_cache(tmp_path, corrupt=corrupt)
    data = CompleteWritingImageData(tmp_path, purpose="train")
    with pytest.raises(ValueError):
        data[0]


def test_report_and_image_tampering_are_rejected(tmp_path):
    make_cache(tmp_path)
    with pytest.raises(ValueError, match="report differs"):
        CompleteWritingImageData(tmp_path, purpose="train", expected_report_sha256="0"*64)
    data = CompleteWritingImageData(tmp_path, purpose="train")
    with (tmp_path/"cmrc/fixture.pt").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="image changed"):
        data[0]
