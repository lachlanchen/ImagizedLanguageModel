"""Read retained weights from the legacy combined Ollama Qwen3-VL archive.

This offline source exposes no vocabulary matrices and instantiates no model.
It reverses storage layouts, not quantization loss or language adaptation.
"""

from pathlib import Path
import re

import numpy as np

from ilm.visual_lm.gguf_language_trunk import LAYER_NAMES, VOCABULARY_NAMES


class QwenVisualLanguageArchive:
    def __init__(self, path, gguf_module):
        self.path, self.gguf = Path(path), gguf_module
        self.reader = gguf_module.GGUFReader(self.path, mode="r")
        if self.field("general.architecture") != "qwen3vl":
            raise ValueError("legacy dense Qwen3-VL archive required")
        if "qwen3vl.rope.dimension_sections" in self.reader.fields:
            raise ValueError("current split-projector GGUF is outside this mapping")
        self.parameters, self.omitted = {}, {}
        split = {}
        for tensor in self.reader.tensors:
            source = tensor.name
            if source in VOCABULARY_NAMES:
                self.omitted[source] = tensor
                continue
            if source == "output_norm.weight":
                target = "language_model.norm.weight"
            elif source.startswith("blk."):
                match = re.fullmatch(r"blk\.(\d+)\.(.+)", source)
                if match is None or match[2] not in LAYER_NAMES:
                    raise ValueError(f"unknown text parameter: {source}")
                target = f"language_model.layers.{int(match[1])}.{LAYER_NAMES[match[2]]}"
            elif source.startswith("v.blk."):
                match = re.fullmatch(r"v\.blk\.(\d+)\.(.+)\.(weight|bias)", source)
                if match is None:
                    raise ValueError(f"unknown visual block parameter: {source}")
                layer, stem, suffix = int(match[1]), match[2], match[3]
                if stem in ("attn_q", "attn_k", "attn_v"):
                    target = f"visual.blocks.{layer}.attn.qkv.{suffix}"
                    parts = split.setdefault(target, {})
                    if stem[-1] in parts:
                        raise ValueError("duplicate visual QKV part")
                    parts[stem[-1]] = tensor
                    continue
                names = {"attn_out": "attn.proj", "norm1": "norm1", "norm2": "norm2",
                         "mlp.linear_fc1": "mlp.linear_fc1", "mlp.linear_fc2": "mlp.linear_fc2"}
                if stem not in names:
                    raise ValueError(f"unknown visual parameter: {source}")
                target = f"visual.blocks.{layer}.{names[stem]}.{suffix}"
            elif source.startswith("v.deepstack_merger."):
                match = re.fullmatch(r"v\.deepstack_merger\.(\d+)\.(norm|linear_fc[12])\.(weight|bias)", source)
                if match is None:
                    raise ValueError(f"unknown deep-stack parameter: {source}")
                target = f"visual.deepstack_merger_list.{int(match[1])}.{match[2]}.{match[3]}"
            elif source.startswith("v.merger."):
                match = re.fullmatch(r"v\.merger\.(norm|linear_fc[12])\.(weight|bias)", source)
                if match is None:
                    raise ValueError(f"unknown visual merger: {source}")
                target = f"visual.merger.{match[1]}.{match[2]}"
            elif source in ("v.patch_embed.weight", "v.patch_embed.bias"):
                target = "visual.patch_embed.proj."+source.rsplit(".", 1)[1]
            elif source == "v.pos_embed.weight":
                target = "visual.pos_embed.weight"
            else:
                raise ValueError(f"unmapped retained tensor: {source}")
            self._insert(target, (tensor,))
        for target, parts in split.items():
            if set(parts) != {"q", "k", "v"}:
                raise ValueError(f"incomplete visual QKV: {target}")
            self._insert(target, tuple(parts[k] for k in ("q", "k", "v")))
        if "token_embd.weight" not in self.omitted:
            raise ValueError("source vocabulary embedding is undeclared")
        # Validate the flattened patch layout before any model allocation.
        raw = self.parameters["visual.patch_embed.proj.weight"][0]
        expected = (self.vision_field("embedding_length")*self.vision_field("num_channels"),
                    self.vision_field("temporal_patch_size"),
                    self.vision_field("patch_size"), self.vision_field("patch_size"))
        if tuple(raw.shape[::-1]) != expected:
            raise ValueError("legacy visual patch layout differs")

    def _insert(self, name, tensors):
        if name in self.parameters:
            raise ValueError("duplicate retained parameter")
        self.parameters[name] = tensors

    def field(self, name):
        if name not in self.reader.fields:
            raise ValueError(f"missing required GGUF field: {name}")
        return self.reader.fields[name].contents()

    def vision_field(self, name):
        return self.field("qwen3vl.vision."+name)

    def shapes(self):
        result = {}
        for name, tensors in self.parameters.items():
            shape = tuple(int(n) for n in tensors[0].shape[::-1])
            if len(tensors) == 3:
                if any(tuple(t.shape[::-1]) != shape for t in tensors):
                    raise ValueError("visual QKV parts differ in shape")
                shape = (3*shape[0], *shape[1:])
            if name == "visual.patch_embed.proj.weight":
                shape = (int(self.vision_field("embedding_length")),
                         int(self.vision_field("num_channels")), *shape[1:])
            result[name] = shape
        return result

    def configurations(self, *, legacy_mrope_sections):
        # This legacy archive omits M-RoPE sections. The caller must supply and
        # attribute the external configuration rather than assuming it is stored.
        sections = list(legacy_mrope_sections)
        head = int(self.field("qwen3vl.attention.key_length"))
        if len(sections) != 3 or any(type(n) is not int or n <= 0 for n in sections) or sum(sections) != head//2:
            raise ValueError("explicit compatible M-RoPE sections required")
        def text(name):
            return self.field("qwen3vl."+name)
        shapes = self.shapes()
        language = dict(vocab_size=0, hidden_size=int(text("embedding_length")),
            intermediate_size=int(text("feed_forward_length")), num_hidden_layers=int(text("block_count")),
            num_attention_heads=int(text("attention.head_count")), num_key_value_heads=int(text("attention.head_count_kv")),
            head_dim=head, rms_norm_eps=float(text("attention.layer_norm_rms_epsilon")),
            max_position_embeddings=int(text("context_length")), rope_theta=float(text("rope.freq_base")),
            rope_scaling={"rope_type": "default", "mrope_section": sections, "mrope_interleaved": True},
            attention_bias=False, bos_token_id=None, eos_token_id=None, pad_token_id=None, tie_word_embeddings=False)
        vision = dict(depth=int(self.vision_field("block_count")), hidden_size=int(self.vision_field("embedding_length")),
            intermediate_size=shapes["visual.blocks.0.mlp.linear_fc1.weight"][0], num_heads=int(self.vision_field("attention.head_count")),
            in_channels=int(self.vision_field("num_channels")), patch_size=int(self.vision_field("patch_size")),
            spatial_merge_size=int(self.vision_field("spatial_merge_size")), temporal_patch_size=int(self.vision_field("temporal_patch_size")),
            out_hidden_size=shapes["visual.merger.linear_fc2.weight"][0], num_position_embeddings=shapes["visual.pos_embed.weight"][0],
            deepstack_visual_indexes=list(self.vision_field("deepstack_visual_indexes")), hidden_act="gelu_pytorch_tanh")
        return {"text": language, "vision": vision}

    def read(self, name):
        if name not in self.parameters:
            raise ValueError("only retained visual/contextual weights can be read")
        values = [self.gguf.dequantize(t.data, t.tensor_type) for t in self.parameters[name]]
        result = np.concatenate(values, axis=0) if len(values) == 3 else values[0]
        expected = self.shapes()[name]
        if name == "visual.patch_embed.proj.weight":
            result = result.reshape(expected)
        if result.shape != expected or not np.isfinite(result).all():
            raise ValueError("dequantized visual-language weight is malformed")
        return np.array(result, dtype=np.float32, copy=True)
