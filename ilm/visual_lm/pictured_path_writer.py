"""Read source embedding addresses from pictured spans and write a translation image."""
from pathlib import Path
import gzip
import json

import numpy as np
import torch
from huggingface_hub import snapshot_download
from torch import nn
from transformers import MarianMTModel

from ilm.visual_lm.optical_span_rollout import COUNTS, PALETTE, OpticalSpanWriter, sha
from scripts.fit_pictured_source_spans import OpticalSourcePath


class PicturedPathReader(nn.Module):
    """Drop-in source state field for the published optical writer."""

    def __init__(self, lang, device):
        super().__init__()
        receipt = json.loads((COUNTS / "source-paths.json").read_text())["summary"][lang]
        weights = COUNTS / (lang + "-source-spans.json.gz")
        if sha(weights) != receipt["fitted"]:
            raise ValueError("Source-path weights differ from the measured development fit")
        base = json.loads((COUNTS / (lang + "-fit.json")).read_text())
        if base["revision"] != receipt["revision"]:
            raise ValueError("Source and target expect different pretrained revisions")
        folder = Path(snapshot_download(base["model"], revision=base["revision"], local_files_only=True))
        model = MarianMTModel.from_pretrained(folder, local_files_only=True, attn_implementation="eager")
        self.encoder = model.model.encoder.eval().to(device)
        self.eos = model.config.eos_token_id
        self.pad = model.config.pad_token_id
        unknown = json.loads((folder / "vocab.json").read_text())["<unk>"]
        self.path = OpticalSourcePath(json.loads(gzip.decompress(weights.read_bytes())), unknown)
        # Each pooled pixel contains the integer ink count (0..4) of a 2x2 block.
        bank = np.load(PALETTE / "palette.npy")
        bits = np.unpackbits(bank, axis=1).reshape(-1, 16, 2, 16, 2)
        counts = bits.sum(axis=(2, 4), dtype=np.uint8)
        self.lookup = {}
        for glyph, tile in enumerate(counts[2:], 2):
            k = tile.tobytes()
            self.lookup[k] = -1 if k in self.lookup else glyph
        self.ambiguous = sum(v < 0 for v in self.lookup.values())

    def addresses(self, features, source):
        counts = (4 * features).to(torch.uint8).cpu().numpy()
        glyphs = [self.lookup.get(row.tobytes(), -1) for row in counts]
        paths = []
        for handles in source.tolist():
            cells = [glyphs[i] for i in handles if i]
            addresses, _ = self.path.infer(cells)
            paths.append(addresses)
        return paths

    def encode(self, paths, device):
        batches = [torch.tensor([*path, self.eos], device=device) for path in paths]
        ids = nn.utils.rnn.pad_sequence(batches, batch_first=True, padding_value=self.pad)
        valid = ids.ne(self.pad).long()
        field = self.encoder(input_ids=ids, attention_mask=valid).last_hidden_state
        return field, valid, torch.zeros_like(valid, dtype=field.dtype)

    @torch.inference_mode()
    def states(self, features, source):
        return self.encode(self.addresses(features, source), source.device)


class PicturedPathWriter(OpticalSpanWriter):
    @torch.inference_mode()
    def __init__(self, lang, device):
        super().__init__(lang, device)
        self.reader = PicturedPathReader(lang, device)
        self.cost.update(optical_reader=0, source_encoder=sum(p.numel() for p in self.reader.parameters()),
                         ambiguous_pooled_patterns=self.reader.ambiguous)
