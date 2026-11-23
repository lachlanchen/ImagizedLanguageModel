"""Intervene on glyph identity alone while preserving physical source addresses."""

from collections import defaultdict
import json
import torch
from torch import nn

from ilm.visual_lm.prepared_glyph_responder import PreparedGlyphResponder
from scripts.evaluate_page_evidence_writer import tensor
from scripts.evaluate_pointing_causal_writer import store_streams
from scripts.fit_font_invariant_reader import Face
from scripts.train_typeface_writer import WORK, PUBLIC, PREVIOUS, FACES, HELD
from scripts.train_causal_glyph_writer import record, sha
from scripts.typeface_writer_cases import sets, extra_sources
from scripts.train_span_choice import gpu_owner


class GivenMap(nn.Module):
    def __init__(self, mapping):
        super().__init__()
        self.mapping = mapping

    def identify(self, packed):
        ids = torch.tensor(
            [self.mapping.get(bytes(row), 0) for row in packed.cpu().numpy()],
            device=packed.device,
        )
        return ids, ids != 0


@torch.inference_mode()
def run():
    groups, _ = sets()
    raw = (
        list(map(json.loads, (PREVIOUS / "pairs_dev.jsonl").read_text().splitlines()))
        + extra_sources()[0]
    )
    model = PreparedGlyphResponder(WORK / "mapped/page", "cuda")
    canonical, face = Face(FACES[0]), Face(HELD)
    used = {v.question_id for group in ("hana", "trial_hana") for v in groups[group]}
    occ = defaultdict(int)
    lookup = defaultdict(set)
    for v in raw:
        if v["id"] not in used:
            continue
        for c in v["p"] + v["q"]:
            x = face.cell(c)
            occ[x] += 1
            lookup[x].add(model.fixed.get(canonical.cell(c)))
    mapping = {
        p: next(iter(ids))
        for p, ids in lookup.items()
        if len(ids) == 1 and None not in ids
    }
    model.net.writer.typeface_map = GivenMap(mapping)
    report = {}
    for group in ("hana", "trial_hana"):
        cases = groups[group]
        pairs = [(tensor(c.p), tensor(c.q)) for c in cases]
        outputs = []
        for i in range(0, len(cases), 16):
            outputs.extend(
                (p.numpy().tobytes(), s)
                for p, s in model.answer_many(pairs[i : i + 16])
            )
        ys, ends = zip(*outputs, strict=True)
        ok = [c.grade(p, s)[2] for c, (p, s) in zip(cases, outputs, strict=True)]
        path = WORK / (group + "_given.npz")
        store_streams(path, ys, ends)
        report[group] = dict(
            n=len(ok),
            exact=sum(ok),
            both=sum(a and b for a, b in zip(ok[::2], ok[1::2], strict=True)),
            streams=sha(path),
        )
    record(
        PUBLIC / "supplied_identity.json",
        dict(
            scores=report,
            types=len(lookup),
            mapped_types=len(mapping),
            input_occurrences=sum(occ.values()),
            unmapped_occurrences=sum(n for k, n in occ.items() if k not in mapping),
            exact_fixed_conflicts=sum(
                n
                for k, n in occ.items()
                if k in model.fixed and mapping.get(k) != model.fixed[k]
            ),
        ),
    )


if __name__ == "__main__":
    with gpu_owner():
        run()
