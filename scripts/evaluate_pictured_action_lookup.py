"""Score two actions, both unseen question frames and both line orders per page."""

from collections import Counter, defaultdict
import numpy as np
import torch
from PIL import Image
from ilm.visual_lm.gloss_focus import cell_features
from ilm.visual_lm.causal_glyph_writer import write_strip
from scripts.answer_pictured_action import (
    analyse,
    contains,
    restore_lookup,
    answer,
    read_lines,
)
from scripts.answer_visual_gloss import load_focus
from scripts.prepare_pictured_action_lookup import (
    ALIGNED,
    CONTROL,
    WORK,
    PUBLIC,
    ROOT,
    read,
    dump,
    sha,
)
from scripts.prepare_pronoun_writer import window
from scripts.pictured_action_questions import HELD_FRAMES
from scripts.train_asked_gloss import packed_batch
from scripts.train_typeface_writer import Face, FACES
from scripts.write_visual_gloss import restore, load_adapter

ARMS = ("neural", "exchanged", "count", "physical", "combined", "supplied_verb_neural")


@torch.inference_mode()
def evaluate():
    torch.set_num_threads(4)
    split = read(PUBLIC / "split.json")
    for path, key in [
        (WORK / "train.json", "train"),
        (PUBLIC / "held.json", "held"),
        (PUBLIC / "pairs.json", "pairs"),
    ]:
        assert sha(path) == split[key + "_sha256"]
    train = read(WORK / "train.json")
    rows = read(PUBLIC / "held.json")
    pairs = read(PUBLIC / "pairs.json")
    face = Face(FACES[0])
    writer = restore()
    reader = load_focus(ROOT / "models/visual_clause_roles", "cpu")
    adapter = load_adapter(ROOT / "models/pronoun_writer", "cpu")
    nets = [restore_lookup(ALIGNED, "cpu"), restore_lookup(CONTROL, "cpu")]

    def pack(s):
        return np.frombuffer(face(s), np.uint8).reshape(-1, 128)

    symbols = {face.cell(c): c for c in "abcdefghijklmnopqrstuvwxyz-"}

    def label(ink):
        return "".join(symbols.get(p.tobytes(), "�") for p in ink)

    counts = defaultdict(Counter)
    for r in train:
        counts[face.cell(r["page"][r["focus"][0]])][r["gloss"][0][0]] += 1
    rawstates = []
    truestates = []
    arcs = []
    for a in range(0, len(rows), 64):
        batch = rows[a : a + 64]
        pixels, lens = packed_batch([r["page"] for r in batch], face, "cpu")
        states, arc = analyse(writer, reader, pixels, lens)
        ix = torch.arange(len(batch))
        rawstates.append(states[ix, arc[:, 0]])
        truestates.append(states[ix, [r["focus"][0] for r in batch]])
        arcs.extend(arc.tolist())
    contexts = torch.cat(rawstates)
    goldstates = torch.cat(truestates)
    windows = sorted(
        {window(r["page"], i) for r, p in zip(rows, arcs) for i in [*p, r["focus"][1]]}
    )
    words = {}
    for a in range(0, len(windows), 96):
        batch = windows[a : a + 96]
        raw, _ = packed_batch(batch, face, "cpu")
        codes, end = writer.write(cell_features(raw), context_adapter=adapter)
        ink = np.packbits(writer.draw(codes).numpy(), axis=-1)
        for w, p, c, e in zip(batch, ink, codes, end):
            words[w] = (p[c.numpy() != writer.stop], bool(e))
    line = []
    for r, p in zip(rows, arcs):
        form = [words[window(r["page"], j)] for j in (*p, r["focus"][1])]
        text = [label(ink) if e else "<unfinished>" for ink, e in form]
        line.append(dict(arc=p, verb=text[0], object=text[1], gold_object=text[2]))
    allverbs = sorted({r["gloss"][0][0] for r in rows})
    queries = [f.format(v) for v in allverbs for f in HELD_FRAMES]
    qindex = {s: i for i, s in enumerate(queries)}
    projected = []
    for net in nets:
        pred = net.predicates(contexts)
        true = net.predicates(goldstates)
        arrays = [[], []]
        for a in range(0, len(queries), 32):
            raw, ql = packed_batch(queries[a : a + 32], face, "cpu")
            q, mask = net.questions(cell_features(raw), ql)
            for j, arr in enumerate((pred, true)):
                parts = [
                    torch.einsum("id,jkd->ijk", z, q)
                    .masked_fill(~mask, -torch.inf)
                    .amax(-1)
                    for z in arr.split(256)
                ]
                arrays[j].append(torch.cat(parts).numpy())
        projected.append([np.concatenate(p, axis=1) for p in arrays])
    sets = {}
    examples = {}
    full = {}
    for group, indices in pairs.items():
        margin = np.zeros((len(indices), 2, 2, len(ARMS)), np.float32)
        semantic = []
        supplied = []
        available = []
        for m, pair in enumerate(indices):
            available.append(
                all(line[i]["gold_object"] in rows[i]["gloss"][1] for i in pair)
            )
            supplied.append(
                all(
                    line[i]["arc"][1] == rows[i]["focus"][1]
                    and line[i]["object"] in rows[i]["gloss"][1]
                    for i in pair
                )
            )
            semantic.append(all(line[i]["object"] in rows[i]["gloss"][1] for i in pair))
            for side, i in enumerate(pair):
                other = pair[1 - side]
                act = rows[i]["gloss"][0][0]
                for k, f in enumerate(HELD_FRAMES):
                    q = f.format(act)
                    qi = qindex[q]
                    question = pack(q)
                    lex = []
                    physical = []
                    for ix in (i, other):
                        row = rows[ix]
                        v = line[ix]["arc"][0]
                        freq = counts[face.cell(row["page"][v])]
                        lex.append(freq[act] / max(1, freq.total()))
                        physical.append(
                            int(contains(question, *words[window(row["page"], v)]))
                        )
                    n, e = [float(p[0][i, qi] - p[0][other, qi]) for p in projected]
                    ph = physical[0] - physical[1]
                    margin[m, side, k] = (
                        n,
                        e,
                        lex[0] - lex[1],
                        ph,
                        ph if ph else n,
                        projected[0][1][i, qi] - projected[0][1][other, qi],
                    )
        strict = (margin > 0).all((1, 2))
        written = strict & np.array(supplied)[:, None]
        stat = dict(
            pairs=len(indices),
            strict_line=strict.sum(0).tolist(),
            strict_written=written.sum(0).tolist(),
            both_meanings_given_line=(strict & np.array(semantic)[:, None])
            .sum(0)
            .tolist(),
            equal_score_questions=(margin == 0).sum((0, 1, 2)).tolist(),
            questions=4 * len(indices),
            supplied_line_and_position=sum(available),
            supplied_line=sum(supplied),
        )
        sets[group] = stat
        full[group] = margin
        for name, mask in [
            ("success", written[:, 4]),
            ("wrong_line", ~strict[:, 4]),
            ("wrong_word", strict[:, 4] & ~np.array(supplied)),
        ]:
            candidates = np.flatnonzero(mask)
            if len(candidates):
                examples[group + "_" + name] = indices[int(candidates[0])]
    np.savez_compressed(PUBLIC / "margins.npz", **full)
    dump(PUBLIC / "line_outputs.json", line)
    sample = PUBLIC / "samples"
    sample.mkdir(exist_ok=True)
    # Run the release itself on the exported page in both row orders.
    replay = []
    for name, pair in examples.items():
        for order, index in enumerate((pair, pair[::-1])):
            page = sample / f"{name}-page-{order}.png"
            width = max(len(rows[i]["page"]) for i in index)
            image = Image.new("L", (width * 32, 64), 255)
            for k, i in enumerate(index):
                bits = (
                    np.unpackbits(pack(rows[i]["page"]), axis=1)
                    .reshape(-1, 32, 32)
                    .transpose(1, 0, 2)
                    .reshape(32, -1)
                )
                image.paste(Image.fromarray(255 * (1 - bits)), (0, 32 * k))
            image.save(page)
            lines = read_lines(page)
            for side, i in enumerate(pair):
                for k, f in enumerate(HELD_FRAMES):
                    query = f.format(rows[i]["gloss"][0][0])
                    question = pack(query)
                    ink, rep = answer(writer, reader, adapter, nets[0], lines, question)
                    g = "same_head" if name.startswith("same_head") else "overall"
                    m = full[g][pairs[g].index(pair), side, k, 4]
                    chosen = i if m > 0 else pair[1 - side] if m < 0 else index[0]
                    assert (
                        rep["row"] == index.index(chosen) + 1
                        and rep["source_cell"] == arcs[chosen][1]
                    )
                    expected = words[window(rows[chosen]["page"], arcs[chosen][1])]
                    assert (
                        np.array_equal(ink, expected[0])
                        and rep["stopped"] == expected[1]
                    )
                    if order == 0 and k == 0:
                        write_strip(question, sample / f"{name}-request-{side}.png")
                        write_strip(ink, sample / f"{name}-answer-{side}.png")
                    replay.append(
                        dict(
                            sample=name,
                            order=order,
                            query=query,
                            expected=rows[i]["gloss"][1],
                            actual=label(ink),
                            **rep,
                        )
                    )
    dump(PUBLIC / "replay.json", replay)
    dump(
        PUBLIC / "result.json",
        dict(
            arms=ARMS,
            groups=sets,
            examples=examples,
            replayed=len(replay),
            replay_sha256=sha(PUBLIC / "replay.json"),
            margins_sha256=sha(PUBLIC / "margins.npz"),
            lines_sha256=sha(PUBLIC / "line_outputs.json"),
            parameters=sum(
                p.numel()
                for net in (nets[0], reader, adapter, writer)
                for p in net.parameters()
            ),
        ),
    )
    print(sets, flush=True)


if __name__ == "__main__":
    evaluate()
