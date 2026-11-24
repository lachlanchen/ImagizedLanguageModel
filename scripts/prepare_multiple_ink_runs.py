"""Build source-disjoint bilingual visual list questions from original annotations."""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import fcntl
import re
from zipfile import ZipFile

import numpy as np
from scripts.evaluate_competing_facts import CorpusFace
from scripts.prepare_compose_dialogue import ARCHIVE as EN, ARCHIVE_SHA
from scripts.prepare_recurrent_evidence import ARCHIVE as ZH, fragments
from scripts.prepare_competing_facts import ROOT, dump, read, sha
from scripts.evaluate_visual_cursor import normalize

WORK = ROOT / "artifacts/multiple_ink_runs_v1"
PUB = ROOT / "references/evidence/multiple_ink_runs_v1"
MODEL = ROOT / "models/multiple_ink_runs"
CAP = 1200


def string(s):
    return " ".join(s.split())


def key(s):
    return sha256(s.encode()).hexdigest()


def chinese(split):
    part = "train" if split == "train" else "dev"
    with ZipFile(ZH) as z:
        for source in ("search", "zhidao"):
            with z.open(f"raw/{part}set/{source}.{part}.json") as f:
                for line in f:
                    row = json.loads(line)
                    if row["question_type"] != "ENTITY":
                        continue
                    refs = []
                    for alts in row.get("entity_answers", []):
                        answer = list(
                            dict.fromkeys(v for s in alts if (v := string(s)))
                        )
                        if answer and answer not in refs:
                            refs.append(answer)
                    if not refs:
                        continue
                    p = " ".join(
                        string(" ".join(d["paragraphs"]))[:300]
                        for d in row["documents"][:4]
                    )[:CAP]
                    if p.strip():
                        yield dict(
                            id=str(row["question_id"]),
                            lang="zh",
                            q=string(row["question"])[:160],
                            p=p,
                            refs=refs,
                        )


def english(split):
    with ZipFile(EN) as z:
        for r in json.loads(z.read("MultiSpanQA_data/" + split + ".json"))["data"]:
            words = r["context"]
            page = " ".join(words)
            offsets = []
            at = 0
            for w, label in zip(words, r["label"], strict=True):
                if label == "B":
                    offsets.append([at, at + len(w)])
                elif label == "I":
                    offsets[-1][1] = at + len(w)
                elif label != "O":
                    raise ValueError(label)
                at += len(w) + 1
            if len(offsets) != r["num_span"]:
                raise ValueError("Annotation count")
            answers = [page[i:j] for i, j in offsets]
            if answers:
                yield dict(
                    id=str(r["id"]),
                    lang="en",
                    q=" ".join(r["question"])[:160],
                    p=page[:CAP],
                    refs=[answers],
                    offsets=offsets,
                )


def locations(row):
    if row["lang"] == "en":
        a = row["offsets"]
        return a if all(j <= len(row["p"]) for _, j in a) else None
    positions = []
    for answer in sorted(row["refs"][0], key=len, reverse=True):
        for m in re.finditer(re.escape(answer), row["p"]):
            if all(m.end() <= i or m.start() >= j for i, j in positions):
                positions.append(m.span())
                break
        else:
            return None
    return sorted(positions)


def multi(r):
    return all(len({normalize(s) for s in a}) > 1 for a in r["refs"])


def prepared_files(work, receipt, filenames, build):
    """Rebuild a missing private cache and authenticate it against any released receipt."""
    work.mkdir(parents=True, exist_ok=True)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    with (work / (receipt.stem + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        released = read(receipt) if receipt.exists() else None
        if released is not None:
            complete = True
            for name, key in filenames.items():
                path = work / name
                if not path.is_file():
                    complete = False
                    continue
                digest = released
                for field in key:
                    digest = digest[field]
                if sha(path) != digest:
                    raise ValueError(f"Changed corpus file: {name}")
            if complete:
                return
        with TemporaryDirectory(dir=work) as temp:
            stage = Path(temp)
            measured = build(stage)
            if released is not None and released != measured:
                raise ValueError("Rebuilt corpus differs from its published receipt")
            for name in filenames:
                (stage / name).replace(work / name)
            if released is None:
                dump(stage / receipt.name, measured)
                (stage / receipt.name).replace(receipt)


def prepare(work=WORK, public=PUB):
    names = {f: ("files", f) for f in ("training.npz", "held.json")}
    prepared_files(work, public / "corpus.json", names, build_training)


def build_training(work):
    if sha(EN) != ARCHIVE_SHA:
        raise ValueError("English archive changed")
    val = list(english("valid")) + list(chinese("valid"))
    held = []
    seenp = set()
    seenq = set()
    for lang, plural, n in (("en", True, 160), ("zh", True, 160), ("zh", False, 32)):
        candidates = sorted(
            (
                r
                for r in val
                if r["lang"] == lang
                and (multi(r) if plural else all(len(a) == 1 for a in r["refs"]))
            ),
            key=lambda r: key("multiple-ink-runs-v1 " + r["id"] + r["p"]),
        )
        count = 0
        for r in candidates:
            p, q = normalize(r["p"]), normalize(r["q"])
            if q in seenq or p in seenp or not q:
                continue
            held.append(r)
            seenq.add(q)
            seenp.add(p)
            count += 1
            if count == n:
                break
        if count != n:
            raise ValueError((lang, plural, count))
    blocked = set().union(*(fragments(r["p"], 24) for r in held))
    palette = [bytes(128), bytes(128)]
    ids = {}
    face = CorpusFace()

    def encode(s):
        seq = []
        for c in s:
            t = face.tile(c)
            if t not in ids:
                ids[t] = len(palette)
                palette.append(t)
            seq.append(ids[t])
        return seq

    # Fallback for an unrenderable input codepoint uses a reserved pictorial
    # marker, never a reference-dependent page exclusion.
    missing = Counter()

    def render(s):
        try:
            return encode(s)
        except ValueError:
            seq = []
            for c in s:
                try:
                    seq.extend(encode(c))
                except ValueError:
                    missing[c] += 1
                    seq.extend(encode("�"))
            return seq

    qs = []
    ps = []
    ys = []
    qo = [0]
    po = [0]
    langs = []
    counts = Counter()
    ids_seen = set()
    import itertools

    for r in itertools.chain(english("train"), chinese("train")):
        lang = r["lang"]
        counts[lang + "_raw"] += 1
        a = locations(r)
        if not a or len(a) > 16:
            continue
        q, p = normalize(r["q"]), normalize(r["p"])
        if not q or not p or q in seenq or p in seenp or (q, p) in ids_seen:
            continue
        if fragments(r["p"], 1) & blocked:
            continue
        try:
            qv = encode(r["q"])
            pv = encode(r["p"])
        except ValueError:
            counts[lang + "_font_excluded"] += 1
            continue
        ids_seen.add((q, p))
        tags = bytearray(len(pv))
        for i, j in a:
            tags[i] = 1
            tags[i + 1 : j] = bytes([2]) * (j - i - 1)
        qs.extend(qv)
        ps.extend(pv)
        ys.extend(tags)
        qo.append(len(qs))
        po.append(len(ps))
        langs.append(int(lang == "zh"))
        counts[lang + "_used"] += 1
        if len(a) > 1:
            counts[lang + "_multi"] += 1
    for r in held:
        r["qi"] = render(r["q"])
        r["pi"] = render(r["p"])
        r["visible_parts"] = [all(ans in r["p"] for ans in a) for a in r["refs"]]
    np.savez_compressed(
        work / "training.npz",
        q=np.asarray(qs, np.int32),
        p=np.asarray(ps, np.int32),
        y=np.asarray(ys, np.uint8),
        qo=np.asarray(qo),
        po=np.asarray(po),
        lang=np.asarray(langs, np.uint8),
        tiles=np.stack([np.frombuffer(t, np.uint8) for t in palette]),
    )
    dump(work / "held.json", held)
    return dict(
        raw=dict(en=sha(EN), zh=sha(ZH)),
        counts=dict(counts),
        held=[r["id"] for r in held],
        fallback_test_characters=dict(missing),
        files={f: sha(work / f) for f in ("training.npz", "held.json")},
    )


if __name__ == "__main__":
    prepare()
