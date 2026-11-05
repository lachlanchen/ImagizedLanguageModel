"""Actual cached legacy VL layout checks; no synthetic model substitutes."""

import json
from pathlib import Path

import pytest
import torch

from ilm.visual_lm.gguf_language_trunk import load_gguf_reader_source
from ilm.visual_lm.gguf_visual_language import QwenVisualLanguageArchive
from scripts.inspect_cached_visual_language import legacy_operator_errors


@pytest.fixture(scope="module")
def archive():
    root = Path(__file__).resolve().parents[2]/"LocalLLM/.local"
    manifest = root/"models/ollama/manifests/registry.ollama.ai/library/qwen3-vl/8b-instruct-q4_K_M"
    source = root/"tools/llama.cpp/gguf-py"
    if not manifest.is_file() or not source.is_dir():
        pytest.skip("real cached Qwen3-VL archive/reader is not installed")
    row = next(r for r in json.loads(manifest.read_text())["layers"] if r["mediaType"] == "application/vnd.ollama.image.model")
    return QwenVisualLanguageArchive(root/"models/ollama/blobs"/row["digest"].replace(":", "-"), load_gguf_reader_source(source))


def test_complete_native_meta_schema_without_language_vocabulary(archive):
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig, Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel, Qwen3VLVisionModel
    config = archive.configurations(legacy_mrope_sections=[24, 20, 20])
    with torch.device("meta"):
        vision = Qwen3VLVisionModel(Qwen3VLVisionConfig(**config["vision"]))
        text = Qwen3VLTextModel(Qwen3VLTextConfig(**config["text"]))
    expected = {"visual."+n: tuple(p.shape) for n, p in vision.named_parameters()}
    expected.update({"language_model."+n: tuple(p.shape) for n, p in text.named_parameters() if n != "embed_tokens.weight"})
    assert archive.shapes() == expected
    assert text.embed_tokens.weight.numel() == 0
    assert len(expected) == 748


def test_legacy_real_weight_time_channel_and_qkv_contractions(archive):
    torch.set_num_threads(4)
    errors = legacy_operator_errors(archive)
    assert max(errors.values()) < 1e-9


def test_vocabulary_and_unattributed_missing_position_configuration_rejected(archive):
    for name in ("token_embd.weight", "output.weight", "language_model.embed_tokens.weight"):
        with pytest.raises(ValueError):
            archive.read(name)
    with pytest.raises(TypeError):
        archive.configurations()
    with pytest.raises(ValueError):
        archive.configurations(legacy_mrope_sections=[1, 2, 3])
