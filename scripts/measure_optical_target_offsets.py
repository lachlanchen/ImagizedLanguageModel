"""Measure target boundaries in the original pictured string using authentic SentencePiece offsets."""
from collections import Counter
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import MarianTokenizer

from scripts.reserve_field_transfer import PUB, ROOT, dump, library, sha
from scripts.score_bilingual_assignments import TEACHERS


def boundaries(tokenizer, text, labels):
    proto = tokenizer.spm_target.encode_as_immutable_proto(text)
    pieces = list(proto.pieces)
    tokens = [p.piece for p in pieces]
    if tokenizer.convert_tokens_to_ids(tokens) + [tokenizer.eos_token_id] != labels:
        raise ValueError("Target codec disagrees with the translator's actual supervision IDs")
    previous_end = 0
    decoded = ""
    eligible = {}
    seen = set()
    excluded = Counter()
    for k, piece in enumerate(pieces):
        if piece.begin != previous_end or piece.surface != text[piece.begin:piece.end]:
            raise ValueError("Target codec does not cover the original optical characters contiguously")
        previous_end = piece.end
        following = tokenizer.convert_tokens_to_string(tokens[:k+1])
        addition = following[len(decoded):] if following.startswith(decoded) else ""
        if piece.end <= piece.begin or piece.begin in seen:
            excluded["empty_or_duplicate"] += 1
        elif labels[k] in tokenizer.all_special_ids or not addition or addition[0] != text[piece.begin]:
            excluded["changed_next_character"] += 1
        else:
            eligible[k] = piece.begin
            seen.add(piece.begin)
        decoded = following
    if previous_end != len(text):
        raise ValueError("Target offsets omit the final optical cells")
    eligible[len(pieces)] = len(text)
    return eligible, excluded


def main():
    by_split = {}
    for t, lang in enumerate(("en", "zh")):
        name, rev = TEACHERS[t]
        folder = snapshot_download("Helsinki-NLP/opus-mt-" + name, revision=rev, local_files_only=True)
        tok = MarianTokenizer.from_pretrained(folder, local_files_only=True)
        for split in ("dev", "train"):
            rows = library(split)
            stats, excluded = Counter(), Counter()
            for j in range(0, len(rows), 256):
                texts = [r["text"][t] for r in rows[j:j+256]]
                labels = tok(text_target=texts).input_ids
                for text, ids in zip(texts, labels, strict=True):
                    allowed, reasons = boundaries(tok, text, ids)
                    stats.update(events=len(ids), aligned=len(allowed), glyph=len(allowed)-1, stop=1)
                    excluded.update(reasons)
            if sum(excluded.values()) + stats["aligned"] != stats["events"]:
                raise ValueError("Incomplete accounting")
            by_split[f"{lang}_{split}"] = dict(counts=stats, excluded=excluded, coverage=stats["aligned"]/stats["events"])
    dump(PUB/"optical_target_offsets.json", dict(sets=by_split, program=sha(Path(__file__)),
         protocol=sha(ROOT/"references/optical_target_offsets_protocol.md")))
    print(by_split, flush=True)


if __name__ == "__main__":
    main()
