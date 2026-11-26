"""Evaluate independent, stopped PNG translations and recovery on the same fixed drafts."""

from collections import Counter, defaultdict
from hashlib import sha256
import numpy as np
from rapidfuzz.distance import Levenshtein
import torch
from ilm.visual_lm.book_prefix import observations
from ilm.visual_lm.portable_writer import DEFAULT_FONT, read_text, read_image, save_grid
from ilm.visual_lm.glyph_typeface_reader import load_typeface_reader
from scripts.prepare_visual_rollins import endpoints, training_rows, BOOK
from scripts.reserve_visual_prefix import (
    BASE,
    ROOT,
    PREVIOUS,
    PUB,
    WORK,
    read,
    dump,
    sha,
)
from scripts.train_rehearsal_typeface import MODEL as READER
from scripts.train_visual_prefix import MODEL
from scripts.train_paired_ink import material
from scripts.train_ordered_translation import pack
from scripts.evaluate_proposition_translation import semantic
from scripts.write_ordered_translation import restore, stopped
from scripts.train_span_choice import gpu_owner
from scripts.verse_gpu_runtime import configure_cuda_cli

MODES = {
    "parent": "book-rehearsal",
    "clean": "prefix-reference",
    "recover": "prefix-recovery",
}
NATURAL = ("book", "talk", "flores")


def cohort():
    fresh = read(WORK / "test.json")
    rows = []
    for j, r in enumerate(fresh):
        foil = next(
            z
            for z in fresh[j + 1 :] + fresh[:j]
            if z["domain"] == r["domain"] and z["doc"] != r["doc"]
        )
        for t in (0, 1):
            rows.append(
                dict(
                    group=r["domain"],
                    id=j,
                    doc=r["doc"],
                    t=t,
                    source=r["text"][1 - t],
                    ref=r["text"][t],
                    foil=foil["text"][t],
                    input=r["cells"][1 - t],
                    target=r["cells"][t],
                )
            )
    previous = read(PREVIOUS / "generated.json")
    roles = [r for r in previous["rows"] if r["group"].startswith("role-")]
    assert len(roles) == 944 and len(rows) == 144
    return rows + roles + [dict(r, group="serif-" + r["group"]) for r in roles]


def fitted_examples():
    """Match one seen natural pair to each held pair in the same training domain by both lengths."""
    cells = training_rows()
    lens = np.count_nonzero(cells, axis=2)
    domains = np.load(BOOK / "domains.npy")
    excluded = np.zeros(len(cells), dtype=bool)
    excluded[np.load(PREVIOUS / "replacement.npz")["indices"]] = True
    lookup = read(BASE / "readback.json")
    held = read(WORK / "test.json")
    rng = np.random.default_rng(27913)
    rr = []
    for t_domain, domain in enumerate(("book", "talk")):
        order = rng.permutation(np.flatnonzero((domains == t_domain) & ~excluded))
        chosen = []
        for r in (z for z in held if z["domain"] == domain):
            diff = np.abs(lens[order] - list(map(len, r["cells"]))).sum(axis=1)
            diff[excluded[order]] = 10000
            i = int(order[diff.argmin()])
            excluded[i] = True
            chosen.append(i)
        for j, i in enumerate(chosen):
            pair = [cells[i, t, : lens[i, t]].tolist() for t in (0, 1)]
            for t in (0, 1):
                dec = lambda seq: "".join(lookup[str(v)] for v in seq).replace(
                    "\u3000", " "
                )
                foil = chosen[(j + 1) % len(chosen)]
                rr.append(
                    dict(
                        group="fitted-" + domain,
                        doc="training-" + str(i),
                        id=i,
                        t=t,
                        source=dec(pair[1 - t]),
                        ref=dec(pair[t]),
                        foil=dec(cells[foil, t, : lens[foil, t]]),
                        input=pair[1 - t],
                        target=pair[t],
                    )
                )
    assert len(rr) == 96
    return rr


@torch.inference_mode()
def generate(rows):
    palette = np.load(BASE / "palette.npy")
    reader = load_typeface_reader(READER, palette[2:], "cpu")
    lookup = read(BASE / "readback.json")
    glyphs = list(map(bytes, palette))
    ids = {b: i for i, b in enumerate(glyphs) if i >= 2}
    alphabet = sorted(
        {c for r in rows if r["group"].startswith("serif") for c in r["source"]}
    )
    serif = dict(
        zip(
            alphabet,
            map(
                bytes,
                read_text(
                    "".join(alphabet),
                    DEFAULT_FONT.with_name("NotoSerifCJK-Regular.ttc"),
                ),
            ),
            strict=True,
        )
    )
    # Compute the adapter once per observed bitmap. Inference itself has no source labels.
    images = np.frombuffer(b"".join(serif.values()), np.uint8).reshape(-1, 128)
    adapted = dict(
        zip(
            serif.values(),
            map(bytes, reader.transport(images, palette[2:])),
            strict=True,
        )
    )
    sources = []
    for i, r in enumerate(rows):
        folder = WORK / "outputs" / f"{i:04d}"
        folder.mkdir(exist_ok=True, parents=True)
        changed = r["group"].startswith("serif")
        raw = (
            b"".join(serif[c] for c in r["source"])
            if changed
            else palette[r["input"]].tobytes()
        )
        p = folder / "input.png"
        save_grid(raw, p)
        physical = list(map(bytes, read_image(p, len(r["input"]))))
        assert b"".join(physical) == raw
        cells = [adapted[b] for b in physical] if changed else physical
        for b in cells:
            if b not in ids:
                ids[b] = len(glyphs)
                glyphs.append(b)
        sources.append([ids[b] for b in cells])
    f = torch.tensor(
        observations(
            {"tiles": np.frombuffer(b"".join(glyphs), np.uint8).reshape(-1, 128)}, "ink"
        ),
        device="cuda",
    )
    inv = [torch.tensor(v, device="cuda") for v in read(BASE / "inventories.json")]
    out = [{} for _ in rows]
    for m, tag in MODES.items():
        net = restore(tag, "cuda")
        for t in (0, 1):
            chosen = [i for i, r in enumerate(rows) if r["t"] == t]
            for start in range(0, len(chosen), 48):
                ix = chosen[start : start + 48]
                seq = net.generate(
                    f, pack([sources[i] for i in ix], "cuda"), inv[t], t
                ).cpu()
                for i, y in zip(ix, seq, strict=True):
                    cells, stop = stopped(y)
                    raw = palette[cells].tobytes()
                    path = WORK / "outputs" / f"{i:04d}" / (m + ".png")
                    save_grid(raw, path)
                    out[i][m] = dict(
                        cells=cells,
                        stop=stop,
                        raw_exact=stop and cells == rows[i]["target"],
                        png=sha(path),
                        packed=sha256(raw).hexdigest(),
                        text="".join(lookup[str(a)] for a in cells),
                    )
    return out


@torch.inference_mode()
def continue_after(net, bank, inv, rows, t, prefixes):
    results = [None] * len(rows)
    sizes = sorted(set(map(len, prefixes)))
    for cut in sizes:
        selected = [j for j, p in enumerate(prefixes) if len(p) == cut]
        for chunk in range(0, len(selected), 48):
            ix = selected[chunk : chunk + 48]
            source = pack([rows[i]["input"] for i in ix], "cuda")
            ctx, valid = net.encode(bank, source)
            prefix = torch.tensor(
                [prefixes[i] for i in ix], dtype=torch.long, device="cuda"
            )
            x = torch.cat((net.begin[t].expand(len(ix), 1, -1), bank[prefix]), 1)
            proto = net.prototypes(bank, inv)
            h, cache = net.decode(x, ctx, valid)
            done = torch.zeros(len(ix), dtype=torch.bool, device="cuda")
            complete = []
            for k in range(129 - cut):
                got = inv[(h[:, -1] @ proto.T).argmax(-1)]
                complete.append(got.masked_fill(done, 0))
                done = done | (got == 1)
                if done.all() or k == 128 - cut:
                    break
                h, cache = net.decode(bank[got, None], ctx, valid, cache, cut + k + 1)
            for i, y in zip(ix, torch.stack(complete, 1).cpu(), strict=True):
                cells, stop = stopped(y, 128 - cut)
                results[i] = (cells, stop)
    return results


@torch.inference_mode()
def recovery(rows, out):
    f, inv, _ = material("cuda")
    logs = {}
    for t in (0, 1):
        # A writer that already stopped before the eighth glyph has no live prefix here.
        ix = [
            i
            for i, r in enumerate(rows[:144])
            if r["t"] == t and len(out[i]["parent"]["cells"]) >= 8
        ]
        rr = [rows[i] for i in ix]
        draft = [out[i]["parent"]["cells"][:8] for i in ix]
        target = np.zeros((len(ix), 16), np.int32)
        for k, r in enumerate(rr):
            v = r["target"][:16]
            target[k, : len(v)] = v
        js, cost = endpoints(np.asarray(draft), target)
        clean = [r["target"][:j] for r, j in zip(rr, js, strict=True)]
        for m, tag in MODES.items():
            net = restore(tag, "cuda")
            bank = net.optical(f)
            for v, prefixes in (("draft", draft), ("gold", clean)):
                emissions = continue_after(net, bank, inv[t], rr, t, prefixes)
                for i, p, j, c, (suffix, stop) in zip(
                    ix, prefixes, js, cost, emissions, strict=True
                ):
                    seq = p + suffix
                    dist = Levenshtein.distance(seq, rows[i]["target"])
                    floor = int(c) if v == "draft" else 0
                    assert dist >= floor
                    if (
                        m == "parent"
                        and v == "draft"
                        and (seq != out[i][m]["cells"] or stop != out[i][m]["stop"])
                    ):
                        raise ValueError(
                            "Forced-prefix continuation of its own reply differs"
                        )
                    logs.setdefault(str(i), {})[m + "-" + v] = dict(
                        prefix=p,
                        reference_endpoint=int(j),
                        floor=floor,
                        cells=seq,
                        stop=stop,
                        edit=dist,
                        excess=dist - floor,
                        suffix_distance=Levenshtein.distance(
                            suffix, rows[i]["target"][j:]
                        ),
                    )
    return logs


def summarize(rows, out, grades, probe):
    exact = defaultdict(Counter)
    difference = defaultdict(lambda: defaultdict(list))
    cases = []
    for i, (r, o) in enumerate(zip(rows, out, strict=True)):
        g = exact[r["group"]]
        g["n"] += 1
        for m in MODES:
            g[m] += int(o[m]["raw_exact"])
            g[m + "_stop"] += int(o[m]["stop"])
        if i < 144:
            ok = {
                m: bool(
                    o[m]["stop"] and all(grades["votes"][2 * i + k][m] for k in (0, 1))
                )
                for m in (*MODES, "qwen")
            }
            cases.append(ok)
            for m in ("clean", "parent"):
                difference[m][(r["group"], r["doc"])].append(
                    int(ok["recover"]) - int(ok[m])
                )
    families = {}
    for start, layout in [(144, "canonical"), (1088, "serif")]:
        families[layout] = {
            m: sum(
                all(v[m]["raw_exact"] for v in out[i : i + 8])
                for i in range(start, start + 944, 8)
            )
            for m in MODES
        }
    ci = {}
    for rival, docs in difference.items():
        rng = np.random.default_rng(203211)
        # Sample source documents independently within each of the three equally weighted domains.
        boot = []
        for g in NATURAL:
            ds = [s for (domain, _), s in docs.items() if domain == g]
            k = rng.integers(len(ds), size=(10000, len(ds)))
            n = np.array([len(s) for s in ds])
            val = np.array([sum(s) for s in ds])
            boot.append(val[k].sum(1) / n[k].sum(1))
        ci[rival] = np.quantile(np.mean(boot, axis=0), [0.025, 0.975]).tolist()
    stats = {}
    for m in MODES:
        stats[m] = {}
        for v in ("draft", "gold"):
            a = [record[m + "-" + v] for record in probe.values()]
            stats[m][v] = {
                k: sum(r[k] for r in a)
                for k in ("edit", "excess", "suffix_distance", "stop", "floor")
            }
            stats[m][v].update(
                n=len(a), exact=sum(r["stop"] and r["edit"] == 0 for r in a)
            )
    return dict(
        exact=exact,
        families=families,
        paired_difference_intervals=ci,
        strict=cases,
        prefixes=stats,
    )


def main():
    configure_cuda_cli()
    torch.set_num_threads(2)
    rows = cohort() + fitted_examples()
    binding = dict(
        split=sha(WORK / "test.json"),
        program=sha(ROOT / "scripts/evaluate_visual_prefix.py"),
        fits={m: sha(MODEL / m / "fit.json") for m in ("reference", "recovery")},
    )
    if binding["split"] != read(PUB / "reservation.json")["sha256"]:
        raise ValueError("Panel changed")
    with gpu_owner():
        p = WORK / "generated.json"
        if p.exists():
            gen = read(p)
            assert gen["binding"] == binding and gen["rows"] == rows
            out = gen["answers"]
        else:
            out = generate(rows)
            dump(p, dict(binding=binding, rows=rows, answers=out))
        pp = WORK / "conditional.json"
        if pp.exists():
            saved = read(pp)
            assert saved["binding"] == binding
            probe = saved["cases"]
        else:
            probe = recovery(rows, out)
            dump(pp, dict(binding=binding, cases=probe))
        torch.cuda.empty_cache()
        indices = [
            i
            for i, r in enumerate(rows)
            if r["group"] in NATURAL or r["group"].startswith("fitted-")
        ]
        path = WORK / "graded.json"
        if path.exists():
            stored = read(path)
            assert stored["binding"] == binding
            grades = stored["grades"]
            for i, s in zip(indices, stored["answers"], strict=True):
                out[i].update(s)
        else:
            assessed = [out[i] for i in indices]
            grades = semantic([rows[i] for i in indices], assessed, tuple(MODES))
            dump(path, dict(binding=binding, grades=grades, answers=assessed))
    summary = summarize(rows, out, grades, probe)
    dump(
        PUB / "results.json",
        dict(
            binding=binding,
            semantic=grades,
            **summary,
            generated=sha(p),
            graded=sha(path),
            conditional=sha(pp),
        ),
    )
    print(summary["families"], grades["groups"], flush=True)


if __name__ == "__main__":
    main()
