"""Check real retained weights against llama.cpp's independent naming map."""

import json
from pathlib import Path

import pytest

from ilm.visual_lm.gguf_language_trunk import QwenTrunkArchive, load_gguf_reader_source


@pytest.mark.parametrize("tag", ["qwen3/1.7b", "qwen3/4b-q4_K_M"])
def test_every_cached_trunk_parameter_matches_official_gguf_name(tag):
    root = Path(__file__).resolve().parents[2]/"LocalLLM/.local"
    source = root/"tools/llama.cpp/gguf-py"
    manifest_path = root/"models/ollama/manifests/registry.ollama.ai/library"/tag
    if not source.is_dir() or not manifest_path.is_file():
        pytest.skip("real shared GGUF archives/reader are not installed")
    gguf = load_gguf_reader_source(source)
    manifest = json.loads(manifest_path.read_text())
    layer = next(row for row in manifest["layers"] if row["mediaType"] == "application/vnd.ollama.image.model")
    archive = QwenTrunkArchive(root/"models/ollama/blobs"/layer["digest"].replace(":", "-"), gguf)
    arch = next(key for key, value in gguf.MODEL_ARCH_NAMES.items() if value == "qwen3")
    mapping = gguf.get_tensor_name_map(arch, archive.configuration()["num_hidden_layers"])
    for name, tensor in archive.parameters.items():
        # The official ForCausalLM model prefixes its trunk with `model.`.
        # This source targets the base transformer, so that prefix is absent.
        stem, suffix = ("model."+name).rsplit(".", 1)
        assert mapping.get_name(stem)+"."+suffix == tensor.name
    assert not any("embed_tokens" in name or "lm_head" in name for name in archive.parameters)
