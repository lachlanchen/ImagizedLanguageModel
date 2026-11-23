"""Count distinct physical source spans needed to cover known MLQA reading errors."""

from collections import Counter
import json
import torch
from torch.nn import functional as F
from safetensors.torch import load_file
from ilm.visual_lm.ordered_page_features import contextualize
from ilm.visual_lm.question_evidence_head import QuestionEvidenceHead
from scripts.evaluate_question_evidence_head import mlqa
from scripts.evaluate_pointing_causal_writer import load_streams
from scripts.reserve_mlqa_pictures import PUBLIC, WORK
from scripts.train_question_evidence_head import bodies
from scripts.train_span_choice import gpu_owner
from scripts.train_causal_glyph_writer import record, sha


@torch.inference_mode()
def run():
    read, questions = bodies()
    h = QuestionEvidenceHead().cuda().eval()
    weights = WORK / "matched.safetensors"
    assert (
        sha(weights) == json.loads((PUBLIC / "matched_fit.json").read_text())["sha256"]
    )
    h.load_state_dict(load_file(str(weights)))
    sets, _ = mlqa()
    out = {}
    for lang, rows in sets.items():
        previous = load_streams(WORK / f"mlqa_{lang}_matched.npz")[0]
        written = {
            a: load_streams(WORK / f"mlqa_{lang}_writer_{a}.npz")
            for a in ("full", "matched")
        }
        counts = Counter()
        for j in range(0, len(rows), 24):
            chunk = rows[j : j + 24]
            vec = questions.encode([c.q for c in chunk])
            states = contextualize(read, [(c.p, c.q) for c in chunk])
            for k, (c, (_, state)) in enumerate(zip(chunk, states, strict=True)):
                left, right = h(state[None, :, :770].cuda(), vec[k : k + 1])[0].unbind(
                    1
                )
                logits = left[:, None] + F.pad(right, (0, 47), value=-torch.inf).unfold(
                    0, 48, 1
                )
                seen = set()
                rank = None
                # Stable physical-position order also matches the original argmax for ties.
                for index in (
                    logits.flatten().argsort(descending=True, stable=True).tolist()
                ):
                    a, b = divmod(index, 48)
                    if a + b >= len(state):
                        continue
                    y = c.p[a * 128 : (a + b + 1) * 128]
                    if y in seen:
                        continue
                    seen.add(y)
                    if len(seen) == 1:
                        counts["same_top1"] += int(previous[j + k] == y)
                    if c.grade(y, True)[2]:
                        rank = len(seen)
                        break
                    if len(seen) == 128:
                        break
                old = (
                    any(
                        c.grade(stream[j + k], ends[j + k])[2]
                        for stream, ends in written.values()
                    )
                    or c.grade(previous[j + k], True)[2]
                )
                counts["n"] += 1
                counts["old_all_wrong"] += int(not old)
                for n in (1, 8, 32, 128):
                    found = rank is not None and rank <= n
                    counts[f"top{n}"] += int(found)
                    counts[f"new_among_misses_top{n}"] += int(found and not old)
        out[lang] = counts
    record(PUBLIC / "interval_ranks.json", out)


if __name__ == "__main__":
    with gpu_owner():
        run()
