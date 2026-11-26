"""Measure conservative teacher supervision at real target piece / optical boundaries."""
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from scripts.reserve_field_transfer import BASE, Ink, PUB, ROOT, dump, sha
from scripts.train_field_to_glyph import data_for, library, material, restore, scores
from scripts.train_readout_objective import configure_cuda_cli, decoder_hidden, gpu_owner, project, teacher_model


def extend(tokenizer, head, pieces, piece):
    """Use the pinned target decoder with the full preceding piece history."""
    text = tokenizer.convert_tokens_to_string([*pieces, piece])
    return text[len(head):] if text.startswith(head) else ""


@torch.inference_mode()
def measure(direction, rows, writer, optical, ink):
    teacher, tok = teacher_model(direction)
    features, inventories, mappings = optical
    bank = writer.writer.optical(features)
    inv = inventories[direction]
    keys = writer.writer.prototypes(bank, inv)
    available = set(inv.tolist())
    special = set(tok.all_special_ids)
    tokens = tok.convert_ids_to_tokens(list(range(teacher.config.vocab_size)))
    inputs, targets, lens = data_for(rows, direction, "cuda")
    results = {"glyph": Counter(), "stop": Counter()}
    rejected = Counter()
    total = 0
    tokenizer_options = dict(padding=True, return_tensors="pt")
    for start in range(0, len(rows), 12):
        group = rows[start:start + 12]
        langs = [r["text"][direction] for r in group]
        source = tok([r["text"][1-direction] for r in group], **tokenizer_options).to("cuda")
        labels = tok(text_target=langs, **tokenizer_options).input_ids.cuda()
        state = teacher.model.encoder(**source).last_hidden_state
        h = decoder_hidden(teacher, state, source.attention_mask, labels)
        w, v = lens[start:start + len(group)].max(0)
        src = inputs[start:start + len(group), :w]
        dst = targets[start:start + len(group), :v]
        field = writer.readers[direction](features, src)[0][:, 1:w+1]
        student = scores(writer, bank, src, dst, field, direction, keys).softmax(-1)
        for i, (row, text) in enumerate(zip(group, langs, strict=True)):
            if ink.encode(text) != row["cells"][direction]:
                raise ValueError("Human target and measured optical grid differ")
            ids = labels[i, labels[i].ne(tok.pad_token_id)].tolist()
            probs, top = project(teacher, h[i, :len(ids)]).softmax(-1).topk(32)
            prefixes = []
            for end in range(len(ids)):
                pieces = [tokens[a] for a in ids[:end]]
                prefixes.append(tok.convert_tokens_to_string(pieces))
            seen = set()
            for k, target_piece in enumerate(ids):
                total += 1
                head = prefixes[k]
                pos = len(head)
                is_stop = target_piece == tok.eos_token_id
                if not text.startswith(head) or pos in seen or (k and head == prefixes[k-1]):
                    rejected["unaligned_or_repeated"] += 1
                    continue
                if is_stop:
                    if pos != len(text):
                        rejected["inexact_stop"] += 1
                        continue
                    goal = 1
                else:
                    if (target_piece in special or pos == len(text) or
                            not prefixes[k+1].startswith(text[:pos+1])):
                        rejected["nonextending_or_different"] += 1
                        continue
                    goal = row["cells"][direction][pos]
                seen.add(pos)
                buckets = Counter()
                pieces = [tokens[a] for a in ids[:k]]
                for piece, probability in zip(top[k].tolist(), probs[k].tolist(), strict=True):
                    if piece == tok.eos_token_id:
                        buckets[1] += probability
                    elif piece not in special:
                        visible = extend(tok, head, pieces, tokens[piece])
                        if visible:
                            char = visible[0]
                            glyph = ink.encode(char)[0]
                            if char not in ink.bad and glyph in available:
                                buckets[glyph] += probability
                resolved = sum(buckets.values())
                if not 0 <= resolved <= 1.000001:
                    raise ValueError("Probability mass error")
                rest = max(0., 1.-resolved)
                rival = max((p for g,p in buckets.items() if g != goal), default=0.)
                student_p = student[i, pos]
                local_goal = mappings[direction][goal].item()
                counts = results["stop" if is_stop else "glyph"]
                counts.update(events=1, resolved=resolved, lower=buckets[goal],
                              upper=min(1., buckets[goal]+rest),
                              certified_correct=int(buckets[goal] > rival+rest),
                              certified_wrong=int(rival > buckets[goal]+rest),
                              student_p=student_p[local_goal].item(),
                              student_correct=int(student_p.argmax().item() == local_goal))
    kept = sum(c["events"] for c in results.values())
    if kept + sum(rejected.values()) != total:
        raise ValueError("Unaccounted target decisions")
    resolved = sum(c["resolved"] for c in results.values())
    return dict(total_boundaries=total, kept=kept, rejected=rejected, by_event=results,
                mean_resolved=resolved/kept, coverage=kept/total,
                adequate=bool(2*kept>=total and 2*resolved>=kept))


def main():
    configure_cuda_cli()
    torch.set_num_threads(2)
    with gpu_owner():
        rows = library("dev")
        net, _ = restore("prediction", "cuda")
        optical = material("cuda")
        ink = Ink(np.load(BASE/"palette.npy"))
        results = {lang:measure(t, rows, net, optical, ink) for t,lang in enumerate(("en","zh"))}
    dump(PUB/"teacher_boundaries.json", dict(directions=results, pairs=len(rows), top_pieces=32,
         program=sha(Path(__file__)), protocol=sha(ROOT/"references/teacher_glyph_boundary_protocol.md")))
    print(results, flush=True)


if __name__ == "__main__":
    main()
