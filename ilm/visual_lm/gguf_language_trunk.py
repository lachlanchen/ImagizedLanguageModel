"""Read-only, one-tensor-at-a-time access to cached Qwen3 transformer weights."""

import importlib
from pathlib import Path
import re
import sys

import numpy as np


LAYER_NAMES = {
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}
VOCABULARY_NAMES = {"token_embd.weight", "output.weight"}


def load_gguf_reader_source(source):
    source = Path(source).resolve()
    if not (source/"gguf/gguf_reader.py").is_file():
        raise FileNotFoundError("existing gguf-py reader source required")
    sys.path.insert(0, str(source))
    module = importlib.import_module("gguf")
    if not Path(module.__file__).resolve().is_relative_to(source):
        raise ValueError("another GGUF package was already imported")
    return module


class QwenTrunkArchive:
    """An offline weight source, not an image-language model or a tokenizer.

    The reader maps the existing file read-only. `read` dequantizes one
    explicitly retained transformer parameter; vocabulary tables are never
    exposed through that accessor. No model is instantiated by this class.
    """

    def __init__(self, path, gguf_module):
        self.path = Path(path)
        self.gguf = gguf_module
        self.reader = gguf_module.GGUFReader(self.path, mode="r")
        self.architecture = self.reader.fields["general.architecture"].contents()
        if self.architecture != "qwen3":
            raise ValueError("this trunk mapping supports plain Qwen3 only")
        self.parameters, self.omitted = {}, {}
        for tensor in self.reader.tensors:
            if tensor.name in VOCABULARY_NAMES:
                self.omitted[tensor.name] = tensor
                continue
            if tensor.name == "output_norm.weight":
                name = "norm.weight"
            else:
                match = re.fullmatch(r"blk\.(\d+)\.(.+)", tensor.name)
                if match is None or match[2] not in LAYER_NAMES:
                    raise ValueError(f"unmapped transformer tensor: {tensor.name}")
                name = f"layers.{int(match[1])}.{LAYER_NAMES[match[2]]}"
            if name in self.parameters:
                raise ValueError("duplicate transformer parameter")
            self.parameters[name] = tensor
        if "token_embd.weight" not in self.omitted:
            raise ValueError("source has no declared vocabulary embedding")

    def configuration(self):
        def field(name):
            return self.reader.fields["qwen3."+name].contents()
        layers = int(field("block_count"))
        head_key = ("qwen3.attention.key_length" if "qwen3.attention.key_length" in self.reader.fields
                    else "qwen3.rope.dimension_count")
        head_dim = int(self.reader.fields[head_key].contents())
        return dict(vocab_size=0, hidden_size=int(field("embedding_length")),
            intermediate_size=int(field("feed_forward_length")), num_hidden_layers=layers,
            num_attention_heads=int(field("attention.head_count")),
            num_key_value_heads=int(field("attention.head_count_kv")), head_dim=head_dim,
            rms_norm_eps=float(field("attention.layer_norm_rms_epsilon")),
            max_position_embeddings=int(field("context_length")),
            rope_theta=float(field("rope.freq_base")), attention_bias=False,
            bos_token_id=None, eos_token_id=None, pad_token_id=None, tie_word_embeddings=False)

    def shapes(self):
        # GGUF dimension order is the reverse of the dense NumPy/Torch order.
        return {name: tuple(int(n) for n in tensor.shape[::-1])
                for name, tensor in self.parameters.items()}

    def read(self, name):
        if name not in self.parameters:
            raise ValueError("only retained transformer parameters can be read")
        tensor = self.parameters[name]
        value = self.gguf.dequantize(tensor.data, tensor.tensor_type)
        expected = self.shapes()[name]
        if value.shape != expected or not np.isfinite(value).all():
            raise ValueError("dequantized transformer tensor is malformed")
        return np.array(value, dtype=np.float32, copy=True)
