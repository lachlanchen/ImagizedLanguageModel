"""Compare the two human answer starts with unrestricted glyph probabilities."""

import json
from collections import Counter

import torch
from torch.nn import functional as F

from ilm.visual_lm.pointing_causal_writer import PointingCausalWriter, layout, padded
from scripts.evaluate_bilingual_novel_writer import corpus
from scripts.evaluate_pointing_causal_writer import load_streams
from scripts.pointing_glyph_corpus import Row
from scripts.train_bilingual_novel_writer import WORK, PUBLIC
from scripts.train_causal_glyph_writer import in_cells, record, sha
from scripts.train_pointing_causal_writer import inputs
from scripts.train_span_choice import gpu_owner


def branch(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i, x, y
    raise ValueError("The two stopped targets have no conflicting action")


@torch.inference_mode()
def run():
    groups = corpus()
    weights = json.loads((PUBLIC / "bilingual_fit.json").read_text())["weights"]
    folder = WORK / "bilingual/page"
    assert all(sha(folder / k) == v for k, v in weights.items())
    model = PointingCausalWriter.restore(folder, "cuda")
    out = {}
    for lang in ("zh", "en"):
        key = "book_" + lang
        cases = groups[key]
        rows = [Row("book", c.question_id, c.p, c.q, c.source[0], b"") for c in cases]
        read, alphabet, pairs, answers = inputs(model.writer, rows)
        saved = WORK / "evaluation/bilingual" / (key + ".npz")
        assert (
            sha(saved)
            == json.loads((PUBLIC / "result.json").read_text())["scores"]["bilingual"][
                key
            ]["sha256"]
        )
        emitted, stops = load_streams(saved)
        data = []
        counts = Counter()
        for begin in range(0, len(rows), 12):
            indexes = list(range(begin, min(begin + 12, len(rows))))
            prompts, starts, expected, other, previous, prefixes = (
                [],
                [],
                [],
                [],
                [],
                [],
            )
            for i in indexes:
                n, a, b = branch(answers[i], answers[i ^ 1])
                p, q = pairs[i]
                prompt, start = layout(p, q, len(read) - 1)
                prefix = answers[i][:n]
                prompts.append(prompt + prefix)
                starts.append(start)
                expected.append(a)
                other.append(b)
                previous.append(prefix[-1] if prefix else -1)
                prefixes.append(n)
            idx, mask = padded(prompts, "cuda", left=True)
            starts = [
                s + idx.size(1) - len(seq)
                for s, seq in zip(starts, prompts, strict=True)
            ]
            pos = (mask.long().cumsum(1) - 1).clamp_min(0)
            hidden = model.writer.body(
                inputs_embeds=read[idx],
                position_ids=pos,
                attention_mask=mask,
                use_cache=False,
            ).last_hidden_state
            src, keep = padded([pairs[i][0] for i in indexes], "cuda")
            keys = model.source_keys(hidden, starts, src.size(1), keep)
            s, g = model.pointer.scores(
                hidden[:, -1:],
                keys,
                src,
                keep,
                torch.tensor(previous, device="cuda")[:, None],
                False,
            )
            frac = g[:, 0].sigmoid()
            probs = (hidden[:, -1] @ alphabet.T).float().softmax(-1) * frac
            probs = F.pad(probs, (0, max(0, int(src.max()) + 1 - len(alphabet))))
            probs.scatter_add_(1, src, s[:, 0].softmax(-1) * (1 - frac))
            for j, i in enumerate(indexes):
                a, b, n = expected[j], other[j], prefixes[j]
                winner = int(probs[j].argmax())
                pa, pb = map(float, probs[j, [a, b]])
                counts["cases"] += 1
                counts["target_over_rival"] += int(pa > pb)
                counts["target_top"] += int(winner == a)
                counts["rival_top"] += int(winner == b)
                counts["third_top"] += int(winner not in (a, b))
                counts["target_over_rival_but_third_top"] += int(
                    pa > pb and winner not in (a, b)
                )
                counts["later_branch"] += int(n > 0)
                # Decode the selected action back to the same physical ink used at inference.
                if winner == 0:
                    chosen = None
                elif winner < len(alphabet):
                    chosen = model.writer.packed[winner - 1].cpu().numpy().tobytes()
                else:
                    chosen = in_cells(cases[i].p)[pairs[i][0].index(winner)]
                ref = in_cells(rows[i].y)
                y = in_cells(emitted[i])
                if len(y) >= n and y[:n] == ref[:n] and (len(y) > n or stops[i]):
                    counts["auditable"] += 1
                    counts["argmax_disagreement"] += int(
                        chosen != (y[n] if len(y) > n else None)
                    )
                data.append(
                    (
                        cases[i].question_id,
                        n,
                        pa,
                        pb,
                        float(probs[j, winner]),
                        winner == a,
                        winner == b,
                    )
                )
        path = WORK / ("conditional_" + lang + ".json")
        record(path, data)
        out[lang] = dict(**counts, scores_sha256=sha(path))
    record(PUBLIC / "conditional.json", out)
    print(out, flush=True)


if __name__ == "__main__":
    with gpu_owner():
        run()
