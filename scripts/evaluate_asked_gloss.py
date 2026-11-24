"""Measure focus, actual stopped answer images, and training-count controls."""

from collections import Counter, defaultdict
import numpy as np
from safetensors.torch import load_file
import torch

from ilm.visual_lm.gloss_focus import GlossFocus, cell_features, encoding_windows
from ilm.visual_lm.visual_word_sense import RADIUS
from ilm.visual_lm.causal_glyph_writer import write_strip
from scripts.prepare_asked_gloss import (
    WORK,
    PUBLIC,
    ROOT,
    PRIOR,
    QUESTIONS,
    read,
    dump,
    sha,
)
from scripts.train_asked_gloss import packed_batch, question_batch, rows_and_split, ARMS
from scripts.train_typeface_writer import Face, FACES
from scripts.write_visual_gloss import restore
from scripts.audit_visual_word_senses import backoff_key


def physical_window(page, i, blank):
    pad = [blank] * 8 + list(page) + [blank] * 8
    return pad[i : i + 17]


def count_predictions(train, held, face):
    cell = {face.cell(" "): 0}

    def encode(c):
        bits = face.cell(c)
        return cell.setdefault(bits, len(cell))

    focus = defaultdict(lambda: np.zeros(3, dtype=np.int32))
    meanings = defaultdict(Counter)
    for row in train:
        page = list(map(encode, row["page"]))
        for j in range(len(page)):
            role = row["focus"].index(j) if j in row["focus"] else 2
            for k, key in enumerate(backoff_key(physical_window(page, j, 0))):
                focus[k, key][role] += 1
    # The stronger lexical baseline gets the full gloss writer's training corpus.
    prior = read(ROOT / "references/evidence/visual_word_senses_20261124/split.json")
    if sha(PRIOR / "train.json") != prior["data"]["train"]["meta"]:
        raise ValueError("Training gloss records changed")
    for _, _, _, left, char, right, g in read(PRIOR / "train.json"):
        v = list(map(encode, left.rjust(8) + char + right.ljust(8)))
        for k, key in enumerate(backoff_key(v)):
            meanings[k, key][g] += 1
    predictions = []
    for r in held:
        page = list(map(encode, r["page"]))
        strength = []
        texts = []
        for i in range(len(page)):
            keys = list(enumerate(backoff_key(physical_window(page, i, 0))))
            count = next(
                (focus[k] for k in keys if k in focus and focus[k].sum() >= 3),
                np.zeros(3),
            )
            strength.append((count[:2] + 1) / (count.sum() + 3))
            votes = next((meanings[k] for k in keys if k in meanings), {})
            texts.append(min(votes, key=lambda w: (-votes[w], w)) if votes else "")
        predictions.append((np.argmax(strength, axis=0).tolist(), texts))
    return predictions


def summary(rows, guesses, outputs):
    location = np.array(
        [[i == j for i, j in zip(pred, r["focus"])] for pred, r in zip(guesses, rows)]
    )
    good = np.array(
        [[out[k] in r["gloss"][k] for k in range(2)] for out, r in zip(outputs, rows)]
    )
    return dict(
        pages=len(rows),
        focus=location.sum(0).tolist(),
        meaning=good.sum(0).tolist(),
        both_focus=int(location.all(1).sum()),
        both_meaning=int(good.all(1).sum()),
        both_grounded=int((location & good).all(1).sum()),
    )


@torch.inference_mode()
def run():
    device = "cpu"
    train_and_held, split = rows_and_split()
    train = [r for r in train_and_held if r["split"] == "train"]
    rows = [r for r in train_and_held if r["split"] != "train"]
    face = Face(FACES[0])
    writer = restore().requires_grad_(False)
    nets = {}
    for arm in ARMS:
        file = WORK / arm / "focus.safetensors"
        if sha(file) != read(WORK / arm / "fit.json")["sha256"]:
            raise ValueError("Changed fitted focus")
        net = GlossFocus()
        net.load_state_dict(load_file(str(file)))
        nets[arm] = net.eval()
    qraw, ql = question_batch(face, device)
    q = cell_features(qraw)
    trained_questions = {arm: net.read_questions(q, ql) for arm, net in nets.items()}
    count = count_predictions(train, rows, face)
    alphabet = {face.cell(c): c for c in "abcdefghijklmnopqrstuvwxyz-"}
    streams = []
    stopped = []
    records = []
    for a in range(0, len(rows), 32):
        batch = rows[a : a + 32]
        packed, lengths = packed_batch([r["page"] for r in batch], face, device)
        f = cell_features(packed)
        ctx = encoding_windows(writer, f)
        selected = {}
        for arm, net in nets.items():
            # A complete question image, with no gold role indicator supplied to the network.
            query = trained_questions[arm][None].expand(len(batch), -1, -1)
            pred = (
                net(ctx, lengths, query)
                .argmax(-1)
                .numpy()
                .reshape(len(batch), len(QUESTIONS), 2)
            )
            for k in range(len(QUESTIONS)):
                selected[f"{arm}/{k}"] = pred[:, k].tolist()
        for j, r in enumerate(batch):
            L = len(r["page"])
            pad = torch.nn.functional.pad(f[j, :L], (0, 0, RADIUS, RADIUS))
            win = pad.unfold(0, 17, 1).transpose(-1, -2)
            codes, ended = writer.write(win)
            ink = np.packbits(writer.draw(codes).numpy(), axis=-1)
            raws = [
                piece[state != writer.stop] for piece, state in zip(ink, codes.numpy())
            ]
            words = [
                "".join(alphabet.get(c.tobytes(), "�") for c in piece)
                if done
                else "<unfinished>"
                for piece, done in zip(raws, ended.tolist())
            ]
            indices = []
            for piece, done in zip(raws, ended.tolist()):
                indices.append(len(streams))
                streams.append(piece)
                stopped.append(done)
            base = count[a + j]
            pred = {n: v[j] for n, v in selected.items()}
            pred["same_question"] = [pred["answer/4"][0]] * 2
            pred["supplied"] = r["focus"]
            pred["quoted"] = [r["page"].index(r["page"][i]) for i in r["focus"]]
            pred["counts"] = base[0]
            pred["counts_given_focus"] = r["focus"]
            out = {
                n: [(base[1] if n.startswith("counts") else words)[idx] for idx in pair]
                for n, pair in pred.items()
            }
            records.append(
                dict(**r, positions=pred, replies=out, indices=indices, all_words=words)
            )
    archive = PUBLIC / "emitted.npz"
    np.savez_compressed(
        archive,
        ink=np.concatenate(streams),
        offsets=np.r_[0, np.cumsum(list(map(len, streams)))],
        stopped=np.asarray(stopped),
    )
    dump(PUBLIC / "predictions.json", records)
    results = {}
    for book in sorted({r["book"] for r in rows} | {"combined"}):
        selected = [r for r in records if book == "combined" or r["book"] == book]
        results[book] = {
            n: summary(
                selected,
                [r["positions"][n] for r in selected],
                [r["replies"][n] for r in selected],
            )
            for n in selected[0]["positions"]
        }
    # Identical local ink in different pages annotated once as verb and once as object.
    hard = {}
    for radius in (1, 2):
        groups = defaultdict(lambda: [[], []])
        for j, r in enumerate(records):
            for k, i in enumerate(r["focus"]):
                s = (
                    r["page"]
                    .rjust(len(r["page"]) + radius)
                    .ljust(len(r["page"]) + 2 * radius)
                )
                groups[face(s[i : i + 2 * radius + 1])][k].append((j, k))
        pairs = []
        for v in groups.values():
            for (a, k), (b, h) in zip(*v):
                if a == b:
                    continue
                pairs.append(((a, k), (b, h)))
        hard[radius] = dict(pairs=len(pairs), indices=pairs)
        for n in ["answer/4", "locations/4", "exchanged/4", "counts", "supplied"]:

            def hit(case):
                j, k = case
                r = records[j]
                return (
                    r["positions"][n][k] == r["focus"][k]
                    and r["replies"][n][k] in r["gloss"][k]
                )

            hard[radius][n] = sum(hit(a) and hit(b) for a, b in pairs)
    # Matched, source-stratified resampling of pages rather than separate questions.
    rng = np.random.default_rng(241127)
    intervals = {}
    for arm in ["answer/4", "locations/4"]:

        def both(r, n):
            return all(
                r["positions"][n][k] == r["focus"][k]
                and r["replies"][n][k] in r["gloss"][k]
                for k in range(2)
            )

        delta = np.array(
            [int(both(r, arm)) - int(both(r, "exchanged/4")) for r in records]
        )
        draws = np.zeros(3000)
        for b in sorted({r["book"] for r in records}):
            v = delta[[r["book"] == b for r in records]]
            draws += rng.choice(v, (len(draws), len(v))).sum(1)
        intervals[arm] = np.percentile(100 * draws / len(records), [2.5, 97.5]).tolist()
    dump(
        PUBLIC / "result.json",
        dict(
            books=results,
            hard=hard,
            paired_pp_intervals=intervals,
            ink=sha(archive),
            predictions=sha(PUBLIC / "predictions.json"),
            split=split["rows_sha256"],
            unrecognized_or_unstopped=sum(
                "�" in w or w == "<unfinished>" for r in records for w in r["all_words"]
            ),
        ),
    )
    for book, res in results.items():
        print(
            book,
            {
                n: res[n]["both_grounded"]
                for n in (
                    "answer/4",
                    "exchanged/4",
                    "locations/4",
                    "counts",
                    "supplied",
                )
            },
        )
    for success, name in [(True, "correct"), (False, "miss")]:
        for r in records:
            if r["book"] != "Daodejing":
                continue
            n = "answer/4"
            ok = all(
                r["positions"][n][k] == r["focus"][k]
                and r["replies"][n][k] in r["gloss"][k]
                for k in range(2)
            )
            if ok != success:
                continue
            example = dict(
                id=r["id"],
                page=r["page"],
                questions=QUESTIONS[4],
                focus=r["positions"][n],
                truth=r["focus"],
                replies=r["replies"][n],
                gloss=r["gloss"],
            )
            dump(PUBLIC / (name + ".json"), example)
            write_strip(
                np.frombuffer(face(r["page"]), np.uint8).reshape(-1, 128),
                PUBLIC / (name + "_page.png"),
            )
            for k, i in enumerate(r["positions"][n]):
                write_strip(streams[r["indices"][i]], PUBLIC / f"{name}_answer{k}.png")
                write_strip(
                    np.frombuffer(face(QUESTIONS[4][k]), np.uint8).reshape(-1, 128),
                    PUBLIC / f"{name}_question{k}.png",
                )
            break


if __name__ == "__main__":
    torch.set_num_threads(3)
    run()
