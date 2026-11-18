"""Score complete, unprompted modern glosses in physical bitmap space."""

import fcntl
from collections import Counter, defaultdict

import numpy as np
import torch
from rapidfuzz.distance import Levenshtein
from safetensors.torch import load_file

from ilm.visual_lm.block_translation import BlockTranslator
from scripts.align_visual_translation import (
    ROOT,
    WORK,
    load,
    require,
    sha,
    tiles,
    write_new,
)
from scripts.train_aligned_visual_translation import blocks, read_rows


def source_batch(batch, bank):
    lengths = torch.tensor([len(row["x"]) for row in batch], device=bank.device)
    indices = torch.zeros(
        (len(batch), int(lengths.max())), dtype=torch.long, device=bank.device
    )
    for i, row in enumerate(batch):
        indices[i, : len(row["x"])] = torch.as_tensor(row["x"], device=bank.device)
    return bank[indices], lengths


def fit_local(rows, ids):
    tables = defaultdict(Counter)
    for row in rows:
        src, target = ([int(ids[i]) for i in row[key]] for key in ("x", "z"))
        padded = [-1, -1] + src + [-1, -1]
        for i, block in enumerate(blocks(src, target)):
            for radius in (0, 1, 2):
                key = tuple(padded[i + 2 - radius : i + 3 + radius])
                tables[key][tuple(block)] += 1
    modes = {}
    for key, counts in tables.items():
        value, n = counts.most_common(1)[0]
        if n >= 2 and n > counts.total() / 2:
            modes[key] = value
    return modes


def local_result(src, modes):
    padded = [-1, -1] + src + [-1, -1]
    output = []
    for i, cell in enumerate(src):
        for radius in (2, 1, 0):
            key = tuple(padded[i + 2 - radius : i + 3 + radius])
            if key in modes:
                output.extend(modes[key])
                break
        else:
            output.append(cell)
    return output


def bootstrap(rows, stats, a, b):
    grouped = defaultdict(lambda: np.zeros(2))
    for row, s in zip(rows, stats):
        grouped[row["group"]] += [s[a]["z"] - s[b]["z"], len(row["z"])]
    sums = np.stack(list(grouped.values()))
    rng = np.random.default_rng(325674)
    samples = sums[rng.integers(len(sums), size=(2000, len(sums)))].sum(axis=1)
    return np.quantile(samples[:, 0] / samples[:, 1], [0.025, 0.975]).tolist()


def evaluate():
    rows = read_rows("heldout")[:2048]
    train = read_rows("train")[:32768]
    require(
        not {r["group"] for r in train}.intersection(r["group"] for r in rows),
        "Overlapping chapters",
    )
    bank = tiles()
    known, ids = np.unique(bank, axis=0, return_inverse=True)
    index = {cell.tobytes(): i for i, cell in enumerate(known)}
    by_id = {i: cell for i, cell in enumerate(known)}
    modes = fit_local(train, ids)
    device_bank = torch.from_numpy(np.unpackbits(bank, axis=1)).to("cuda", torch.bool)
    observed = {
        arm: [] for arm in ("copy", "local", "y", "z", "y_reversed", "z_reversed")
    }
    for row in rows:
        src = ids[row["x"]].tolist()
        observed["copy"].append(src)
        observed["local"].append(local_result(src, modes))
    out = WORK / "evaluation"
    out.mkdir()
    fit = load(WORK / "fit/fit.json")
    for arm in ("y", "z"):
        filename = WORK / "fit" / (arm + ".safetensors")
        require(sha(filename) == fit["arms"][arm]["sha256"], "Checkpoint changed")
        state = load_file(str(filename))
        model = BlockTranslator(state["landmarks"])
        model.load_state_dict(state)
        model.cuda().eval()
        for start in range(0, len(rows), 48):
            images, lengths = source_batch(rows[start : start + 48], device_bank)
            for reverse in (False, True):
                key = arm + ("_reversed" if reverse else "")
                for cells in model.generate(images, lengths, reverse_context=reverse):
                    packed = np.packbits(cells.cpu().numpy(), axis=1)
                    result = []
                    for cell in packed:
                        bitmap = cell.tobytes()
                        number = index.setdefault(bitmap, len(index))
                        by_id.setdefault(number, cell)
                        result.append(number)
                    observed[key].append(result)
    all_stats = [{} for _ in rows]
    summary = {}
    np.save(out / "bitmap_index.npy", np.stack(list(by_id.values())))
    for key, streams in observed.items():
        require(len(streams) == len(rows), "Missing translations")
        np.savez_compressed(
            out / (key + ".npz"),
            sequence=np.array([i for s in streams for i in s], dtype=np.int32),
            offsets=np.cumsum([0] + list(map(len, streams))),
        )
        for row, seq, stats in zip(rows, streams, all_stats):
            stats[key] = {
                ref: Levenshtein.distance(seq, ids[row[ref]].tolist())
                for ref in ("y", "z")
            }
        summary[key] = dict(
            length=sum(map(len, streams)),
            empty=sum(not s for s in streams),
            novel_bitmap_instances=sum(i >= len(known) for s in streams for i in s),
        )
        for ref in ("y", "z"):
            errs = np.array([s[key][ref] for s in all_stats])
            summary[key][ref] = dict(
                edits=int(errs.sum()),
                exact=int((errs == 0).sum()),
                cer=float(errs.sum() / sum(len(r[ref]) for r in rows)),
            )
        delta = np.array([s[key]["z"] - s["copy"]["z"] for s in all_stats])
        summary[key]["vs_copy"] = [
            int((delta < 0).sum()),
            int((delta == 0).sum()),
            int((delta > 0).sum()),
        ]
        summary[key]["delta_cer_ci95_vs_copy"] = bootstrap(rows, all_stats, key, "copy")
    chars = defaultdict(Counter)
    for row in train + rows:
        for key, name in [("x", "source"), ("z", "new")]:
            for char, i in zip(row[name], ids[row[key]]):
                chars[int(i)][char] += 1

    def transcript(seq):
        return "".join(
            chars[i].most_common(1)[0][0] if i in chars else "▧" for i in seq
        )

    seen = set()
    blind, mapping = [], []
    rng = np.random.default_rng(249)
    for j, row in enumerate(rows):
        if row["group"] in seen:
            continue
        seen.add(row["group"])
        opts = {
            k: v[j] for k, v in observed.items() if k in ("local", "y", "z", "copy")
        }
        opts["gloss"] = ids[row["z"]].tolist()
        unique = list(dict.fromkeys(tuple(s) for s in opts.values()))
        rng.shuffle(unique)
        blind.append(
            dict(source=row["source"], choices=[transcript(seq) for seq in unique])
        )
        mapping.append(
            dict(row=j, options={k: unique.index(tuple(s)) for k, s in opts.items()})
        )
        if len(blind) == 24:
            break
    write_new(out / "review.json", blind)
    write_new(out / "mapping.json", mapping)
    write_new(out / "per_row.json", all_stats)
    write_new(
        out / "summary.json",
        dict(
            sentences=len(rows),
            chapters=len({r["group"] for r in rows}),
            changed=sum(r["y"] != r["z"] for r in rows),
            corrected_length=sum(len(r["z"]) for r in rows),
            old_length=sum(len(r["y"]) for r in rows),
            modes=len(modes),
            arms=summary,
            corrected_minus_original_ci=bootstrap(rows, all_stats, "z", "y"),
            corrected_minus_reversed_ci=bootstrap(rows, all_stats, "z", "z_reversed"),
            review_sha256=sha(out / "review.json"),
            files={p.name: sha(p) for p in out.iterdir()},
        ),
    )


if __name__ == "__main__":
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("high")
    with (ROOT / "artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        evaluate()
