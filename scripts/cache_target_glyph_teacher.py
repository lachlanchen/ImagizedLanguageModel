"""Cache conservative optical targets at observed boundaries of the training translations."""
from pathlib import Path
from time import monotonic
import os

from huggingface_hub import snapshot_download
import numpy as np
import torch
from transformers import MarianTokenizer

from scripts.measure_optical_target_offsets import boundaries
from scripts.reserve_target_glyph_book_panel import BASE, Ink, PUB, ROOT, RUN, dump, library, read, sha
from scripts.train_readout_objective import TEACHERS, configure_cuda_cli, decoder_hidden, gpu_owner, project, teacher_model
from scripts.sample_visual_library import RECEIPT

TOP = 32


def persist(path, arrays, metadata):
    temp = path.with_suffix(".partial.npz")
    np.savez(temp, **arrays)
    os.replace(temp, path)
    dump(path.with_suffix(".json"), dict(**metadata, sha256=sha(path)))


def valid(path, binding):
    receipt = path.with_suffix(".json")
    if not receipt.exists():
        return False
    record = read(receipt)
    if binding != record["binding"] or not path.is_file() or sha(path) != record["sha256"]:
        raise ValueError(f"Changed cache at {path}")
    return True


@torch.inference_mode()
def teacher_decisions(t, rows, path, binding):
    if valid(path, binding):
        return
    begin = monotonic()
    teacher, tok = teacher_model(t)
    id_batches, mass_batches, lengths = [], [], []
    for j in range(0, len(rows), 32):
        group = rows[j:j+32]
        source = tok([r["text"][1-t] for r in group], padding=True, return_tensors="pt").to("cuda")
        target = tok(text_target=[r["text"][t] for r in group], padding=True, return_tensors="pt").input_ids.cuda()
        mask = target.ne(tok.pad_token_id)
        enc = teacher.model.encoder(**source).last_hidden_state
        hidden = decoder_hidden(teacher, enc, source.attention_mask, target)[mask]
        lengths.extend(mask.sum(1).tolist())
        for chunk in hidden.split(128):
            probs, ids = project(teacher, chunk).float().softmax(-1).topk(TOP, dim=-1)
            mass_batches.append(probs.cpu().numpy())
            id_batches.append(ids.int().cpu().numpy())
    persist(path, dict(ids=np.concatenate(id_batches), mass=np.concatenate(mass_batches), lengths=np.array(lengths)),
            dict(binding=binding, seconds=monotonic()-begin))
    print("teacher", t, sum(lengths), flush=True)


def optical_targets(t, rows, folder, raw, path, binding):
    if valid(path, binding):
        return
    begin = monotonic()
    tok = MarianTokenizer.from_pretrained(folder, local_files_only=True)
    sp = tok.spm_target
    ink = Ink(np.load(BASE/"palette.npy"))
    opts = read(BASE/"inventories.json")[t]
    local = {glyph:i for i,glyph in enumerate(opts)}
    special = set(tok.all_special_ids)
    ids, mass, lengths = (raw[k] for k in ("ids","mass","lengths"))
    tokens = tok.convert_ids_to_tokens(list(range(len(tok))))
    labels = tok(text_target=[r["text"][t] for r in rows]).input_ids
    total = int(lengths.sum())
    addr = np.full((len(rows), 1+max(len(r["cells"][t]) for r in rows)), -1, dtype=np.int32)
    choices = np.zeros((total, TOP), np.int16)
    weights = np.zeros((total, TOP), np.float32)
    n = 0
    offset = 0
    char_map = {}

    def glyph_code(char):
        if char not in char_map:
            glyph = ink.encode(char)[0]
            char_map[char] = None if char in ink.bad else local.get(glyph)
        return char_map[char]

    for i, (row, label) in enumerate(zip(rows, labels, strict=True)):
        text = row["text"][t]
        if len(label) != lengths[i] or ink.encode(text) != row["cells"][t]:
            raise ValueError("Teacher length or pictured gold target mismatch")
        allowed, _ = boundaries(tok, text, label)
        wordpieces = [tokens[a] for a in label]
        for k, pos in allowed.items():
            prefix = wordpieces[:k]
            before = tok.convert_tokens_to_string(prefix)
            # The native target codec matches Marian for ordinary pieces. Let Marian
            # handle histories containing a special symbol using its own rules.
            fast = all(a not in special for a in label[:k])
            pooled = {}
            for piece_id, m in zip(ids[offset+k], mass[offset+k], strict=True):
                if piece_id == tok.eos_token_id:
                    code = local[1]
                elif piece_id in special:
                    continue
                else:
                    sequence = prefix + [tokens[piece_id]]
                    extended = (sp.decode_pieces(sequence).replace("▁"," ").strip() if fast
                                else tok.convert_tokens_to_string(sequence))
                    if i % 512 == 0 and extended != tok.convert_tokens_to_string(sequence):
                        raise ValueError("Optimized optical boundary disagrees with teacher codec")
                    if len(extended) <= len(before) or not extended.startswith(before):
                        continue
                    code = glyph_code(extended[len(before)])
                    if code is None:
                        continue
                pooled[code] = pooled.get(code, 0.) + float(m)
            if addr[i,pos] != -1:
                raise ValueError("Two teacher decisions mapped to the same optical prefix")
            addr[i,pos] = n
            for col, (code, probability) in enumerate(pooled.items()):
                choices[n,col] = code
                weights[n,col] = probability
            n += 1
        offset += len(label)
    expected = read(ROOT/"references/evidence/field_to_glyph_v1/optical_target_offsets.json")
    if n != expected["sets"][("en_train","zh_train")[t]]["counts"]["aligned"] or offset != total:
        raise ValueError("Incomplete boundary transport")
    w = weights[:n]
    if not np.isfinite(w).all() or (w.sum(1)>1.00001).any():
        raise ValueError("Invalid transported mass")
    persist(path, dict(addr=addr, choices=choices[:n], weights=w),
            dict(binding=binding, rows=len(rows), decisions=total, aligned=n,
                 mean_resolved=float(w.sum(dtype=np.float64)/n), seconds=monotonic()-begin))
    print("optical", t, n, flush=True)


def main():
    configure_cuda_cli()
    torch.set_num_threads(2)
    rows = library("train")
    programs = [Path(__file__), ROOT/"scripts/measure_optical_target_offsets.py",
                ROOT/"scripts/train_readout_objective.py", ROOT/"references/target_glyph_teaching_protocol.md",
                BASE/"inventories.json", BASE/"palette.npy", RECEIPT]
    common = {str(p.relative_to(ROOT)):sha(p) for p in programs}
    for t, lang in enumerate(("en", "zh")):
        name, rev = TEACHERS[t]
        binding = dict(**common, revision=rev, target=lang)
        raw_path = RUN/f"{lang}-teacher.npz"
        with gpu_owner():
            teacher_decisions(t, rows, raw_path, binding)
        torch.cuda.empty_cache()
        folder = snapshot_download("Helsinki-NLP/opus-mt-"+name, revision=rev, local_files_only=True)
        path = RUN/f"{lang}-optical.npz"
        with np.load(raw_path) as raw:
            optical_targets(t, rows, folder, raw, path, dict(**binding, teacher_cache=sha(raw_path)))
    dump(PUB/"targets.json", {lang:read(RUN/f"{lang}-optical.json") for lang in ("en","zh")})


if __name__=="__main__":
    main()
