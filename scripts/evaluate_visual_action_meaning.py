"""Measure action-description transfer on identical pages and previously untested poems."""

from collections import defaultdict
from functools import lru_cache
import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from ilm.visual_lm.gloss_focus import cell_features
from ilm.visual_lm.causal_glyph_writer import write_strip
from scripts.answer_pictured_action import (
    restore_lookup,
    analyse,
    contains,
    answer,
    read_lines,
)
from scripts.answer_visual_gloss import load_focus
from scripts.audit_pictured_action_lookup import alternate
from scripts.pictured_action_questions import HELD_FRAMES
from scripts.prepare_pronoun_writer import window
from scripts.prepare_visual_action_meaning import (
    ROOT,
    WORK,
    PARENT,
    clean,
    table,
    read,
    dump,
    sha,
)
from scripts.train_typeface_writer import Face, FACES
from scripts.train_asked_gloss import packed_batch
from scripts.write_visual_gloss import restore, load_adapter

ARMS = ("parent", "aligned", "exchanged")
OUTPUT = Path(os.environ.get("ILM_MEANING_RESULTS", WORK / "evaluation"))


def records():
    rows = read(PARENT / "held.json")
    pairs = read(PARENT / "pairs.json")["overall"]
    definitions = {r["action"]: r["description"] for r in table("webster.tsv")}
    requests = defaultdict(list)

    def add(g, ij, terms):
        requests[g].append((ij, [f.format(s) for s in terms for f in HELD_FRAMES]))

    for pair in pairs:
        actions, objects = zip(
            *[(rows[i]["gloss"][0][0], rows[i]["gloss"][1][0]) for i in pair]
        )
        add("literal", pair, actions)
        terms = alternate(list(actions), list(objects))
        if terms:
            add("editorial", pair, terms)
        desc = [definitions.get(a) for a in actions]
        if (
            all(desc)
            and desc[0] != desc[1]
            and not any(set(clean(s).split()) & set(objects + actions) for s in desc)
        ):
            add("webster", pair, desc)
            if all(len(v.split()) > 1 for v in desc):
                add("webster_multiword", pair, desc)
            add("webster_literal", pair, actions)
    for p in table("tang.tsv"):
        text = p["line"]
        rows.append(
            dict(
                page=text,
                book="tang-" + p["poem"],
                split="tang",
                focus=[text.index(p["verb"]), text.index(p["object"])],
                gloss=[[p["action"]], p["accepted"].split("|")],
                id=p["pair"],
            )
        )
    start = len(rows) - len(table("tang.tsv"))
    for i in range(start, len(rows), 2):
        ij = [i, i + 1]
        actions = [rows[j]["gloss"][0][0] for j in ij]
        assert rows[i]["id"] == rows[i + 1]["id"]
        add("tang_literal", ij, actions)
        add("tang_webster", ij, [definitions[a] for a in actions])
    return rows, requests


@torch.inference_mode()
def run():
    torch.set_num_threads(3)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    rows, requests = records()
    writer = restore()
    reader = load_focus(ROOT / "models/visual_clause_roles", "cpu")
    adapter = load_adapter(ROOT / "models/pronoun_writer", "cpu")
    release = Path(os.environ.get("ILM_MEANING_MODELS", WORK))
    nets = [
        restore_lookup(f, "cpu")
        for f in (
            ROOT / "models/pictured_action_lookup",
            release / "aligned",
            release / "exchanged",
        )
    ]
    face = Face(FACES[0])
    symbols = {face.cell(c): c for c in "abcdefghijklmnopqrstuvwxyz-"}

    @lru_cache(None)
    def pic(s):
        return np.frombuffer(face(s), np.uint8).reshape(-1, 128).copy()

    def text(x):
        return "".join(symbols.get(b.tobytes(), "�") for b in x)

    predicted = []
    given = []
    arcs = []
    for a in range(0, len(rows), 64):
        batch = rows[a : a + 64]
        pix, l = packed_batch([r["page"] for r in batch], face, "cpu")
        state, arc = analyse(writer, reader, pix, l)
        ix = torch.arange(len(batch))
        predicted.append(state[ix, arc[:, 0]])
        given.append(state[ix, [r["focus"][0] for r in batch]])
        arcs += arc.tolist()
    states = torch.stack([torch.cat(predicted), torch.cat(given)])
    windows = sorted(
        {window(r["page"], n) for r, p in zip(rows, arcs) for n in (*p, *r["focus"])}
    )
    words = {}
    for a in range(0, len(windows), 96):
        batch = windows[a : a + 96]
        pix, _ = packed_batch(batch, face, "cpu")
        codes, end = writer.write(cell_features(pix), context_adapter=adapter)
        ink = np.packbits(writer.draw(codes).numpy(), axis=-1)
        for w, bits, c, stopped in zip(batch, ink, codes, end):
            glyphs = bits[c.numpy() != writer.stop]
            words[w] = (
                glyphs,
                bool(stopped),
                text(glyphs) if stopped else "<unfinished>",
            )
    lines = []
    for r, p in zip(rows, arcs):
        emitted = [words[window(r["page"], j)][2] for j in [*p, *r["focus"]]]
        lines.append(
            dict(
                arc=p,
                words=emitted,
                source=r["page"],
                focus=r["focus"],
                registered=r["gloss"],
            )
        )
    queries = sorted({q for cases in requests.values() for _, qs in cases for q in qs})
    qid = {q: i for i, q in enumerate(queries)}
    # Score just the requested comparisons; every unequal score has opposite
    # physical row indices under page reversal (also checked in raw-image replay).
    need = defaultdict(set)
    for cases in requests.values():
        for ij, qs in cases:
            for q in qs:
                need[q].update(ij)
    scores = [{} for _ in nets]
    for arm, net in enumerate(nets):
        keys = net.predicates(states.reshape(-1, 160)).reshape(2, len(rows), -1)
        for a in range(0, len(queries), 32):
            batch = queries[a : a + 32]
            pix, lens = packed_batch(batch, face, "cpu")
            features, mask = net.questions(cell_features(pix), lens)
            for k, q in enumerate(batch):
                indices = sorted(need[q])
                value = (keys[:, indices] @ features[k, mask[k]].T).amax(-1).numpy().T
                scores[arm][q] = dict(zip(indices, value))
    groups = {}
    arrays = {}
    cases_json = {}
    examples = {}
    report_hits = {}
    for g, cases in requests.items():
        margins = np.zeros((len(cases), 4, len(ARMS), 3), np.float32)
        hits = np.zeros((len(cases), 4, 2), bool)
        gold = []
        object_ok = []
        for n, (ij, questions) in enumerate(cases):
            gold.append(all(lines[j]["words"][3] in rows[j]["gloss"][1] for j in ij))
            object_ok.append(
                all(
                    arcs[j][1] == rows[j]["focus"][1]
                    and lines[j]["words"][1] in rows[j]["gloss"][1]
                    for j in ij
                )
            )
            for k, q in enumerate(questions):
                side = k // 2
                i, j = ij[side], ij[1 - side]
                for t, ix in enumerate((i, j)):
                    w = words[window(rows[ix]["page"], arcs[ix][0])]
                    hits[n, k, t] = contains(pic(q), w[0], w[1])
                opt = int(hits[n, k, 0]) - int(hits[n, k, 1])
                for arm, record in enumerate(scores):
                    pred, true = record[q][i] - record[q][j]
                    margins[n, k, arm] = (opt if opt else pred, pred, true)
        strict = (margins > 0).all(1)
        complete = strict[:, :, 0] & np.asarray(object_ok)[:, None]
        groups[g] = dict(
            pairs=len(cases),
            line=strict[:, :, 0].sum(0).tolist(),
            written=complete.sum(0).tolist(),
            line_neural=strict[:, :, 1].sum(0).tolist(),
            given_verb_line_neural=strict[:, :, 2].sum(0).tolist(),
            given_line_written=sum(object_ok),
            given_line_and_object_written=sum(gold),
            unique_physical=int((hits[:, :, 0] != hits[:, :, 1]).sum()),
            requests=len(cases) * 4,
            both_correct_baseline_new=int((complete[:, 0] & complete[:, 1]).sum()),
            both_correct_control_new=int((complete[:, 2] & complete[:, 1]).sum()),
        )
        if g.startswith("tang") or g in ("editorial", "webster"):
            for label, mask in [
                ("improvement", complete[:, 1] & ~complete[:, 0]),
                ("success", complete[:, 1]),
                ("failure", ~complete[:, 1]),
            ]:
                win = np.flatnonzero(mask)
                if len(win):
                    examples[g + "_" + label] = (g, int(win[0]))
        arrays[g] = margins
        report_hits[g] = hits
        cases_json[g] = [[ij, [qid[q] for q in qs]] for ij, qs in cases]
    assert (
        groups["literal"]["written"][0] == 703
        and groups["editorial"]["written"][0] == 7
    )
    params = sum(
        p.numel() for n in (nets[1], writer, reader, adapter) for p in n.parameters()
    )
    dump(OUTPUT / "cases.json", dict(queries=queries, cases=cases_json))
    dump(OUTPUT / "lines.json", lines)
    np.savez_compressed(OUTPUT / "margins.npz", **arrays)
    np.savez_compressed(OUTPUT / "physical.npz", **report_hits)
    dump(
        OUTPUT / "result.json",
        dict(
            arms=ARMS,
            groups=groups,
            examples=examples,
            parameters=params,
            margin_axes=["combined", "neural", "supplied_verb_neural"],
            margins_sha256=sha(OUTPUT / "margins.npz"),
        ),
    )
    # Direct physical replay of all poetry requests and selected Kyoto examples.
    directory = OUTPUT / "samples"
    directory.mkdir()
    examples.update(
        {
            "tang_" + str(k): ("tang_webster", k)
            for k in range(len(requests["tang_webster"]))
        }
    )
    replay = []
    cache = {}
    for name, (g, k) in examples.items():
        ij, qs = requests[g][k]
        for reverse in (False, True):
            ids = ij[::-1] if reverse else ij
            paths = [pic(rows[i]["page"]) for i in ids]
            for qn, q in enumerate(qs):
                key = (tuple(ids), q)
                if key not in cache:
                    cache[key] = answer(writer, reader, adapter, nets[1], paths, pic(q))
                ink, rep = cache[key]
                side = qn // 2
                delta = arrays[g][k, qn, 1, 0]
                chosen = (
                    ij[side] if delta > 0 else ij[1 - side] if delta < 0 else ids[0]
                )
                expected = words[window(rows[chosen]["page"], arcs[chosen][1])]
                assert (
                    ids[rep["row"] - 1] == chosen
                    and expected[1] == rep["stopped"]
                    and np.array_equal(ink, expected[0])
                ), (name, ids, q, rep, delta)
                replay.append(
                    [
                        name,
                        int(reverse),
                        qn,
                        rep["row"],
                        rep["source_cell"],
                        rep["stopped"],
                        text(ink),
                    ]
                )
                if (
                    not reverse
                    and qn in (0, 2)
                    and name.endswith(("success", "improvement", "failure"))
                ):
                    write_strip(pic(q), directory / f"{name}-q{side}.png")
                    write_strip(ink, directory / f"{name}-a{side}.png")
            if not reverse and name.endswith(("success", "improvement", "failure")):
                im = Image.new("L", (max(map(len, paths)) * 32, 64), 255)
                for j, x in enumerate(paths):
                    b = (
                        np.unpackbits(x, axis=1)
                        .reshape(-1, 32, 32)
                        .transpose(1, 0, 2)
                        .reshape(32, -1)
                    )
                    im.paste(Image.fromarray(255 * (1 - b)), (0, 32 * j))
                path = directory / f"{name}-page.png"
                im.save(path)
                assert all(
                    np.array_equal(x, y) for x, y in zip(paths, read_lines(path))
                )
    dump(
        OUTPUT / "replay.json", dict(actual_release_calls=len(cache), responses=replay)
    )
    for g, s in groups.items():
        print(
            g,
            s["pairs"],
            s["line"],
            s["written"],
            s["given_line_written"],
            s["given_line_and_object_written"],
            flush=True,
        )


if __name__ == "__main__":
    run()
