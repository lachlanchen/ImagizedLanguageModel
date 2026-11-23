"""Compare the likelihoods of original stopped outputs and annotated alternatives."""

from collections import Counter
import json

import torch
from torch.nn import functional as F
from ilm.visual_lm.pointing_causal_writer import PointingCausalWriter, layout, padded
from scripts.evaluate_pointing_causal_writer import load_streams
from scripts.pointing_glyph_corpus import Row
from scripts.reserve_first_glyph_books import reserved
from scripts.train_bilingual_novel_writer import WORK as ORDINARY
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_first_glyph_book import PUBLIC, WORK
from scripts.train_pointing_causal_writer import inputs
from scripts.train_span_choice import gpu_owner


def score_batch(net, read, alphabet, pairs, ys):
    prompts, offsets = zip(
        *(layout(p, q, len(read) - 1) for p, q in pairs), strict=True
    )
    ids, mask = padded(
        [p + y[:-1] for p, y in zip(prompts, ys, strict=True)], read.device
    )
    src, valid = padded([p for p, q in pairs], read.device)
    target, keep = padded(ys, read.device)
    prev, _ = padded([[-1] + y[:-1] for y in ys], read.device)
    h = net.writer.body(
        inputs_embeds=read[ids], attention_mask=mask, use_cache=False
    ).last_hidden_state
    keys = net.source_keys(h, offsets, src.size(1), valid)
    starts = torch.tensor([len(p) - 1 for p in prompts], device=read.device)
    at = (starts[:, None] + torch.arange(target.size(1), device=read.device)).clamp_max(
        h.size(1) - 1
    )
    states = h[torch.arange(len(pairs), device=read.device)[:, None], at]
    address, gate = net.pointer.scores(states, keys, src, valid, prev, False)
    branch = gate.sigmoid()
    prob = (states @ alphabet.T).float().softmax(-1) * branch
    prob = F.pad(prob, (0, max(0, int(src.max()) + 1 - len(alphabet))))
    prob.scatter_add_(
        2,
        src[:, None].expand(-1, target.size(1), -1),
        address.softmax(-1) * (1 - branch),
    )
    selected = prob.gather(2, target[:, :, None]).squeeze(2)
    if not (selected[keep] > 0).all():
        raise ValueError("Unsupported or numerically underflowed observed action")
    logp = selected.double().log().masked_fill(~keep, 0).sum(1)
    wrong = (prob.argmax(-1) != target) & keep
    return logp.tolist(), wrong.cpu().tolist()


@torch.inference_mode()
def run():
    result = json.loads((PUBLIC / "result.json").read_text())
    arm = "first" if result["selected"] else "ordinary"
    folder = (WORK / "first" if arm == "first" else ORDINARY / "bilingual") / "page"
    weights = json.loads((folder.parent / "page_fit.json").read_text())["weights"]
    assert all(sha(folder / k) == v for k, v in weights.items())
    net = PointingCausalWriter.restore(folder, "cuda")
    samples = []
    marks = Counter()
    for group, cases in reserved().items():
        if group == "fresh_erased":
            continue
        path = WORK / "evaluation" / arm / (group + ".npz")
        assert sha(path) == result["scores"][arm][group]["sha256"]
        ys, ends = load_streams(path)
        rows = []
        indices = []
        for i, c in enumerate(cases):
            candidates = list(dict.fromkeys([ys[i], *c.source, *cases[i ^ 1].source]))
            indices.append({r: len(rows) + j for j, r in enumerate(candidates)})
            rows.extend(
                Row("analysis", c.question_id, c.p, c.q, r, b"") for r in candidates
            )
        read, alphabet, pairs, labels = inputs(net.writer, rows)
        probs = []
        differences = []
        for j in range(0, len(rows), 8):
            p, d = score_batch(net, read, alphabet, pairs[j : j + 8], labels[j : j + 8])
            probs.extend(p)
            differences.extend(d)
        for i, c in enumerate(cases):
            ix = indices[i]
            idx = ix[ys[i]]
            actions = len(labels[idx]) - (not ends[i])
            assert not any(differences[idx][:actions]), (group, i)
            marks["verified_greedy_actions"] += actions
            best = max(probs[ix[r]] for r in c.source)
            rival = max(probs[ix[r]] for r in cases[i ^ 1].source)
            actual = probs[idx] if ends[i] else None
            exact = c.grade(ys[i], ends[i])[2]
            if exact:
                marks["exact"] += 1
            elif not ends[i]:
                marks["capped_misses"] += 1
            else:
                marks["stopped_misses"] += 1
                delta = best - actual
                side = (
                    "reference"
                    if delta > 1e-5
                    else "emitted"
                    if delta < -1e-5
                    else "tie"
                )
                marks["favors_" + side] += 1
                if all(ys[i][:128] != r[:128] for r in c.source):
                    marks["wrong_first_favors_" + side] += 1
            marks["target_over_paired_rival"] += best > rival + 1e-5
            samples.append(
                dict(
                    group=group,
                    id=c.question_id,
                    exact=exact,
                    emitted=actual,
                    best_reference=best,
                    best_rival=rival,
                )
            )
    record(
        PUBLIC / "sequence_probability.json",
        dict(arm=arm, counts=dict(marks), samples=samples),
    )
    print(dict(marks), flush=True)


if __name__ == "__main__":
    with gpu_owner():
        run()
