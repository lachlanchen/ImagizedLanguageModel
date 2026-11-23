"""Condition a free-running glyph writer on two pictured questions and ranked excerpts."""

from hashlib import sha256

import torch
from torch.nn.functional import pad

from ilm.visual_lm.ordered_page_features import contextualize


@torch.inference_mode()
def ranked_intervals(reader, questions, head, pairs, count=8):
    """Return distinct physical spellings as half-open positions in each original page."""
    if not pairs:
        return []
    if count < 1 or any(not s or len(s) % 128 for pair in pairs for s in pair):
        raise ValueError(
            "Expected packed 32 by 32 glyphs and a positive candidate count"
        )
    vectors = questions.encode([q for _, q in pairs])
    windows = []
    merged = [{} for _ in pairs]
    for i, (page, question) in enumerate(pairs):
        cap = 509 - len(question) // 128
        if cap < 48:
            raise ValueError("Question exceeds the excerpt reader's context")
        for start in range(0, len(page) // 128, cap - 47):
            part = page[128 * start : 128 * (start + cap)]
            windows.append((i, start, part, question))
            if (start + cap) * 128 >= len(page):
                break
    for j in range(0, len(windows), 24):
        batch = windows[j : j + 24]
        states = contextualize(reader, [(p, q) for _, _, p, q in batch])
        for (i, start, part, _), (_, state) in zip(batch, states, strict=True):
            features = state[None, :, :770].to(vectors.device)
            left, right = head(features, vectors[i : i + 1])[0].unbind(1)
            scores = left[:, None] + pad(right, (0, 47), value=-torch.inf).unfold(
                0, 48, 1
            )
            flat = scores.flatten().cpu()
            seen = set()
            for ix in flat.argsort(descending=True, stable=True).tolist():
                a, delta = divmod(ix, 48)
                b = a + delta + 1
                if b > len(state):
                    continue
                spelling = part[a * 128 : b * 128]
                if spelling in seen:
                    continue
                seen.add(spelling)
                item = (-float(flat[ix]), start + a, start + b)
                if spelling not in merged[i] or item < merged[i][spelling]:
                    merged[i][spelling] = item
                if len(seen) == count:
                    break
    return [[(a, b) for _, a, b in sorted(d.values())[:count]] for d in merged]


def bank_intervals(page, question, intervals, *, shifted):
    n = len(page) // 128
    budget = 60
    kept = []
    seed = sha256(page + question).digest()
    for rank, (a, b) in enumerate(intervals):
        width = b - a
        if not 0 <= a < b <= n:
            raise ValueError("Excerpt outside page")
        if width > budget:
            continue
        if shifted and n > width:
            h = sha256(seed + rank.to_bytes(4, "big")).digest()
            a = (a + 1 + int.from_bytes(h[:8], "big") % (n - width)) % (n - width + 1)
        kept.append((a, a + width))
        budget -= width
    return kept


def writer_inputs(page, qs, ranks, separator, *, shifted=False, plain=False):
    if len(qs) != 2 or len(ranks) != 2 or len(separator) != 128:
        raise ValueError("Expected two pictured questions and one separator cell")
    prompt = separator.join(qs)
    if plain:
        return page, prompt
    parts = [page]
    for question, intervals in zip(qs, ranks, strict=True):
        bank = bank_intervals(page, question, intervals, shifted=shifted)
        parts.append(separator.join(page[a * 128 : b * 128] for a, b in bank))
    return (separator * 2).join(parts), prompt
