"""Test frozen list models on fresh pages, same-page question pairs, and English statements."""

from collections import defaultdict
from hashlib import sha256
from time import monotonic
import numpy as np
import torch

from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.multiple_ink_runs import intervals
from ilm.visual_lm.portable_writer import read_text, DEFAULT_FONT
from scripts.benchmark_multiple_ink_runs import compare
from scripts.evaluate_competing_facts import CorpusFace
from scripts.evaluate_multiple_ink_runs import corpus, evaluate_cases, grade
from scripts.evaluate_visual_discourse import packed, unpack
from scripts.measure_writer_energy import consumed, snapshot
from scripts.prepare_complete_fact_sets import MODEL, WORK, PUB, ROOT
from scripts.prepare_multiple_ink_runs import read, dump, sha, multi, normalize
from scripts.write_multiple_ink_runs import load_reader, forward, sentences, joined
from scripts.write_visual_discourse import Writer
from scripts.write_visual_statements import editor
from scripts.train_paired_evidence import gpu_owner

ARMS = ("parent", "control", "contrast")


def setup():
    spec = read(PUB / "fresh.json")
    if (
        sha(WORK / "confirm.json") != spec["sha256"]
        or sha(WORK / "confirm-tiles.npy") != spec["tiles"]
    ):
        raise ValueError("Fresh list corpus changed")
    rows = read(WORK / "confirm.json")
    _, tiles = corpus()
    return rows, np.concatenate((tiles, np.load(WORK / "confirm-tiles.npy")))


def run_dir(arm):
    return MODEL / arm if arm != "parent" else None


def paired_set(base):
    p = ROOT / "artifacts/competing_facts_v1/held.json"
    if (
        sha(p)
        != read(ROOT / "references/evidence/competing_facts_v1/corpus.json")["files"][
            p.name
        ]
    ):
        raise ValueError("Changed paired pages")
    face = CorpusFace()
    bits = list(map(bytes, base))
    lookup = {v: i for i, v in enumerate(bits) if i >= 2}

    def encode(txt):
        seq = []
        for c in txt:
            try:
                b = face.tile(c)
            except ValueError:
                b = face.tile("�")
            if b not in lookup:
                lookup[b] = len(bits)
                bits.append(b)
            seq.append(lookup[b])
        return seq

    rows = []
    for page in read(p):
        text = page["source"][:1200]
        pi = encode(text)
        for no, r in enumerate(page["rows"]):
            q = r["query"][:160]
            rows.append(
                dict(
                    kind=page["kind"],
                    doc=page["doc"],
                    i=no,
                    q=q,
                    p=text,
                    refs=[[a] for a in r["refs"]],
                    pi=pi,
                    qi=encode(q),
                )
            )
    return rows, np.frombuffer(b"".join(bits), np.uint8).reshape(-1, 128)


@torch.inference_mode()
def measure(arm, rows, tiles, paired, paired_tiles):
    net = load_reader("cuda", True, run_dir(arm))
    writer = Writer(device="cuda")
    head = editor("cuda")
    sep = read_text(" and ", DEFAULT_FONT)
    bar = read_text(" | ", DEFAULT_FONT)
    sent = []
    paired_records = []
    details = unpack(PUB / (arm + "-answers.json.gz"))
    actual = {
        v["id"]: v
        for v in details
        if v["mode"] == "count" and v["condition"] == "original"
    }
    alpha = {bytes(b): c for b, c in zip(writer.palette, writer.transcript)}
    en = [r for r in rows if r["lang"] == "en"]
    for r in en:
        for j, c in zip(r["qi"] + r["pi"], r["q"] + r["p"], strict=True):
            alpha.setdefault(bytes(tiles[j]), c)
    for i in range(0, len(en), 8):
        g = en[i : i + 8]
        lists = []
        eligible = []
        for r in g:
            page = tiles[r["pi"]]
            parts = [page[a:b] for a, b in actual[r["id"]]["spans"]]
            assert [sha256(w.tobytes()).hexdigest() for w in parts] == actual[r["id"]][
                "ink"
            ]
            if (
                len(r["qi"])
                + 2
                + sum(map(len, parts))
                + len(sep) * max(0, len(parts) - 1)
                > 600
            ):
                sent.append(
                    dict(
                        id=r["id"],
                        reason="input exceeds sentence-writer geometry",
                        stop=False,
                        all_present=False,
                    )
                )
                continue
            eligible.append(r)
            lists.append(parts)
        if not eligible:
            continue
        ans = sentences(writer, head, [tiles[r["qi"]] for r in eligible], lists, sep)
        for (pixels, stop), r in zip(ans, eligible, strict=True):
            text = "".join(alpha[bytes(v)] for v in pixels)
            sent.append(
                dict(
                    id=r["id"],
                    text=text,
                    stop=stop,
                    cells=len(pixels),
                    sha256=sha256(pixels.tobytes()).hexdigest(),
                    all_present=stop
                    and all(normalize(a) in normalize(text) for a in r["refs"][0]),
                )
            )
    bank = net.ink(
        torch.from_numpy(observations({"tiles": paired_tiles}, "ink")).cuda()
    )
    for i in range(0, len(paired), 12):
        g = paired[i : i + 12]
        roles, counts = forward(
            net, bank, [r["qi"] for r in g], [r["pi"] for r in g], True
        )
        for role, count, r in zip(roles, counts, g, strict=True):
            bounds = intervals(role[: len(r["pi"])], count)
            parts = [r["p"][a:b] for a, b in bounds]
            paired_records.append(
                dict(
                    doc=r["doc"],
                    kind=r["kind"],
                    i=r["i"],
                    parts=parts,
                    refs=r["refs"],
                    exact=grade(parts, r["refs"])["exact"],
                )
            )
    by = defaultdict(list)
    for r in paired_records:
        by[r["doc"]].append(r)
    summar = {}
    for kind in sorted({r["kind"] for r in paired_records}):
        pairs = [p for p in by.values() if p[0]["kind"] == kind]
        assert all(len(p) == 2 for p in pairs)
        summar[kind] = dict(
            pages=len(pairs),
            both=sum(all(r["exact"] for r in p) for p in pairs),
            both_swapped=sum(
                all(
                    grade(r["parts"], p[1 - j]["refs"])["exact"]
                    for j, r in enumerate(p)
                )
                for p in pairs
            ),
        )
    # Same first 16 plural pages per language as the local quantized comparison.
    small = [
        r
        for lang in ("en", "zh")
        for r in [r for r in rows if r["lang"] == lang and multi(r)][:16]
    ]
    bank = net.ink(torch.from_numpy(observations({"tiles": tiles}, "ink")).cuda())
    torch.cuda.synchronize()
    scored = []
    changed = 0
    t = monotonic()
    energy = snapshot()
    for i in range(0, 32, 8):
        g = small[i : i + 8]
        roles, counts = forward(
            net, bank, [r["qi"] for r in g], [r["pi"] for r in g], True
        )
        for role, count, r in zip(roles, counts, g, strict=True):
            bounds = intervals(role[: len(r["pi"])], count)
            ink = [tiles[r["pi"][a:b]] for a, b in bounds]
            changed += [sha256(v.tobytes()).hexdigest() for v in ink] != actual[
                r["id"]
            ]["ink"]
            scored.append(grade([r["p"][a:b] for a, b in bounds], r["refs"])["exact"])
            joined(ink, bar)
    torch.cuda.synchronize()
    elapsed = monotonic() - t
    cost = consumed(energy, snapshot())["joules"]
    packed(PUB / (arm + "-sentences.json.gz"), sent)
    packed(PUB / (arm + "-pairs.json.gz"), paired_records)
    dump(
        PUB / (arm + "-transfer.json"),
        dict(
            paired=summar,
            sentences=dict(
                n=len(sent),
                stopped=sum(r["stop"] for r in sent),
                contains_all=sum(r["all_present"] for r in sent),
            ),
            matched32=dict(
                seconds=elapsed,
                joules=cost,
                english=sum(scored[:16]),
                chinese=sum(scored[16:]),
                different_from_full_batch=changed,
            ),
        ),
    )


@torch.inference_mode()
def history():
    from scripts.evaluate_recurrent_history import historical_pages
    from scripts.write_visual_discourse import encode_pages
    from scripts.evaluate_visual_discourse import draw

    old, pages, targets = historical_pages()
    shapes, seq = encode_pages(np.zeros((2, 128), np.uint8), pages)
    physical = torch.from_numpy(observations({"tiles": shapes}, "ink")).cuda()
    separator = read_text(" | ", DEFAULT_FONT)
    for arm in ARMS:
        model = load_reader("cuda", True, run_dir(arm))
        bank = model.ink(physical)
        detail = []
        for i in range(0, len(old), 16):
            q = seq[2 * i : 2 * (i + 16) : 2]
            p = seq[2 * i + 1 : 2 * (i + 16) : 2]
            roles, counts = forward(model, bank, q, p, True)
            for j, lengths in enumerate(zip(roles, counts, p, strict=True)):
                role, count, ids = lengths
                r = old[i + j]
                spans = intervals(role[: len(ids)], count)
                ans = joined([shapes[ids[a:b]] for a, b in spans], separator)
                record = dict(
                    id=r["label"],
                    lang=r["language"],
                    headword=r["headword"],
                    example=r["example"],
                    stage=r["stage"],
                    cells=len(ans),
                    correct=np.array_equal(ans, targets[i + j]),
                    output=sha256(ans.tobytes()).hexdigest(),
                )
                detail.append(record)
                if r["example"] and arm == "contrast":
                    dest = MODEL / "examples" / "yan" / r["language"] / r["stage"]
                    dest.mkdir(parents=True, exist_ok=True)
                    draw(dest, "answer", ans)
                    dump(dest / "receipt.json", record)
        dump(PUB / (arm + "-history.json"), detail)


def main():
    rows, tiles = setup()
    paired, larger = paired_set(tiles)
    for arm in ARMS:
        evaluate_cases(rows, tiles, PUB, arm + "-", True, run_dir(arm))
        with gpu_owner():
            measure(arm, rows, tiles, paired, larger)
    with gpu_owner():
        history()
    sample = [
        r
        for l in ("en", "zh")
        for r in [r for r in rows if r["lang"] == l and multi(r)][:16]
    ]
    for text in (False, True):
        compare(text, sample, tiles, PUB, PUB / "contrast-answers.json.gz")


if __name__ == "__main__":
    from scripts.verse_gpu_runtime import configure_cuda_cli

    configure_cuda_cli()
    main()
