"""Replay every pictured request in the reserved Pu Songling experiment on CPU."""

import argparse
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from scripts.answer_pictured_action import answer, restore_lookup
from scripts.answer_visual_gloss import load_focus
from scripts.evaluate_visual_action_meaning import ARMS
from scripts.pictured_action_questions import HELD_FRAMES
from scripts.prepare_visual_action_meaning import ROOT, STUDY, WORK, dump, read, table
from scripts.train_typeface_writer import Face, FACES
from scripts.write_visual_gloss import restore, load_adapter, respond


@torch.inference_mode()
def evaluate(models, output):
    screen = read(STUDY / "pu_screen.json")
    rows = screen["rows"]
    assert [r["line"] for r in rows] == [r["line"] for r in table("pu.tsv")]
    eligible = set(screen["clean_pair_indices"])
    definitions = {r["action"]: r["description"] for r in table("webster.tsv")}
    writer = restore()
    source = load_focus(ROOT / "models/visual_clause_roles", "cpu")
    adapter = load_adapter(ROOT / "models/pronoun_writer", "cpu")
    arms = [
        restore_lookup(p, "cpu")
        for p in (
            ROOT / "models/pictured_action_lookup",
            models / "aligned",
            models / "exchanged",
        )
    ]
    face = Face(FACES[0])

    @lru_cache(None)
    def image(s):
        return np.frombuffer(face(s), np.uint8).reshape(-1, 128).copy()

    alphabet = {face.cell(c): c for c in "abcdefghijklmnopqrstuvwxyz-"}

    def transcribe(pixels):
        return "".join(alphabet.get(p.tobytes(), "�") for p in pixels)

    output.mkdir(parents=True, exist_ok=False)
    draws, offsets, records = [], [0], []
    ceilings = []
    for row in rows:
        obj = row["focus"][1]
        pixels, stopped = respond(
            writer, image(row["line"]), obj, context_adapter=adapter
        )
        ceilings.append(
            dict(
                word=transcribe(pixels),
                stopped=stopped,
                accepted=stopped and transcribe(pixels) in row["accepted"].split("|"),
            )
        )
    for i in range(0, len(rows), 2):
        pair = rows[i : i + 2]
        for mode in ("literal", "webster"):
            for arm, net in zip(ARMS, arms):
                for rev in (False, True):
                    lines = [image(r["line"]) for r in (pair[::-1] if rev else pair)]
                    for side, r in enumerate(pair):
                        expr = (
                            r["action"]
                            if mode == "literal"
                            else definitions[r["action"]]
                        )
                        for frame in HELD_FRAMES:
                            question = frame.format(expr)
                            ink, rep = answer(
                                writer, source, adapter, net, lines, image(question)
                            )
                            draws.append(ink)
                            offsets.append(offsets[-1] + len(ink))
                            records.append(
                                dict(
                                    pair=i // 2,
                                    mode=mode,
                                    arm=arm,
                                    reverse=rev,
                                    side=side,
                                    question=question,
                                    row=rep["row"],
                                    cell=rep["source_cell"],
                                    stopped=rep["stopped"],
                                )
                            )
    ink_file = output / "emitted.npz"
    np.savez_compressed(
        ink_file, pixels=np.concatenate(draws), offsets=np.array(offsets)
    )
    dump(output / "requests.json", records)
    # Recount from the persisted pixels and positions, including both row orders.
    with np.load(ink_file) as archive:
        pixels, offsets = archive["pixels"], archive["offsets"]
    groups = {}
    for mode in ("literal", "webster"):
        for arm in ARMS:
            by_pair = {}
            for n, r in enumerate(read(output / "requests.json")):
                if (r["mode"], r["arm"]) != (mode, arm):
                    continue
                side = r["side"]
                target = rows[2 * r["pair"] + side]
                drawn = transcribe(pixels[offsets[n] : offsets[n + 1]])
                row_ok = r["row"] == 1 + (side ^ int(r["reverse"]))
                written = (
                    row_ok
                    and r["stopped"]
                    and r["cell"] == target["focus"][1]
                    and drawn in target["accepted"].split("|")
                )
                by_pair.setdefault(r["pair"], []).append((row_ok, written))
            assert all(len(v) == 8 for v in by_pair.values())
            groups[mode + "_" + arm] = dict(
                pairs=len(eligible),
                line=sum(all(x for x, y in by_pair[k]) for k in eligible),
                written=sum(all(y for x, y in by_pair[k]) for k in eligible),
                written_indices=[
                    k for k in sorted(eligible) if all(y for x, y in by_pair[k])
                ],
            )
    result = dict(
        groups=groups,
        supplied_positions=ceilings,
        supplied_pair_ceiling=sum(
            all(ceilings[2 * k + j]["accepted"] for j in (0, 1)) for k in eligible
        ),
        calls=len(records),
        clean_pair_indices=sorted(eligible),
    )
    dump(output / "result.json", result)
    print(groups, "supplied ceiling", result["supplied_pair_ceiling"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", type=Path, default=WORK)
    p.add_argument("--output", type=Path, default=WORK / "pu_evaluation")
    args = p.parse_args()
    torch.set_num_threads(2)
    evaluate(args.models, args.output)
