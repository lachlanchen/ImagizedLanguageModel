"""Bind local participant heads to annotated antecedents in pictured transitive clauses."""
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import ImageFont
import requests

from scripts.prepare_natural_pictured_roles import (
    FONT, RAW, ROOT, RUN as OLD, WINDOW, bitmap, clauses, dump, read, sha,
)

WORK = ROOT / "artifacts/visual_referents_v1"
PUB = ROOT / "references/evidence/visual_referents_v1"
MODEL = ROOT / "models/visual_referents"
EXTERNAL = "12cbbbbf9edef2036309cb3bb8e8db346cdad815"


def paired(block, corpus):
    rows = clauses(block, Counter(), corpus)
    if not rows:
        return []
    text = next(line[9:] for line in block.splitlines() if line.startswith("# text = "))
    tokens, offset = {}, 0
    for line in block.splitlines():
        fields = line.split("\t")
        if len(fields) == 10 and fields[0].isdigit():
            a = text.index(fields[1], offset)
            offset = a + len(fields[1])
            refs = [part.split(":")[0] for part in fields[8].split("|") if part.endswith(":ref")]
            tokens[fields[0]] = ((a, offset), refs)
    result = []
    for row in rows:
        matches = [i for i in range(len(text)-len(row["text"])+1) if text.startswith(row["text"], i)]
        if len(matches) != 1:
            continue
        left, = matches
        at = {(a-left, b-left): refs for (a,b),refs in tokens.values()}
        local = [row["candidates"][k] for k in (row["label"], 1-row["label"])]
        target = []
        for span in local:
            refs = at[span]
            if len(refs) > 1 or (refs and refs[0] not in tokens):
                break
            if refs:
                a,b = tokens[refs[0]][0]
                span = (a-left, b-left)
            target.append(span)
        if len(target) != 2:
            continue
        spans = sorted(target + [row["verb"]])
        if spans[0][0]<0 or spans[-1][1]>len(row["text"]) or any(a[1]>b[0] for a,b in zip(spans,spans[1:])):
            continue
        result.append(dict(**row, local=local, target=target, linked=target!=local))
    return result


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    PUB.mkdir(parents=True, exist_ok=True)
    external = RAW / "GENTLE"
    external.mkdir(exist_ok=True)
    for name in ("en_gentle-ud-test.conllu","README.md","LICENSE.txt"):
        path = external / name
        if not path.exists():
            url = f"https://raw.githubusercontent.com/UniversalDependencies/UD_English-GENTLE/{EXTERNAL}/{name}"
            res = requests.get(url, timeout=45)
            res.raise_for_status()
            path.write_bytes(res.content)
    parent_train = read(OLD/"train.json")
    seen_full, seen_crop = set(), set()
    fit_full = {r["full"] for r in parent_train}
    fit_crop = {r["text"] for r in parent_train}
    counts, hashes, raw, chars = {}, {}, {}, set()
    for split in ("external","test","dev","train"):
        rows = []
        corpora = ("GENTLE",) if split=="external" else ("EWT","GUM")
        for corpus in corpora:
            part = "test" if split=="external" else split
            path = RAW/corpus/f"en_{corpus.lower()}-ud-{part}.conllu"
            raw[f"{corpus}/{path.name}"] = sha(path)
            for block in path.read_text().split("\n\n"):
                group = paired(block, corpus)
                if not group or group[0]["full"] in seen_full:
                    continue
                if split!="train" and group[0]["full"] in fit_full:
                    continue
                good = [r for r in group if r["text"] not in seen_crop and
                        (split=="train" or r["text"] not in fit_crop)]
                rows.extend(good)
                seen_full.add(group[0]["full"])
                seen_crop.update(r["text"] for r in good)
        counts[split] = dict(Counter(f'{r["voice"]}/{"linked" if r["linked"] else "direct"}' for r in rows))
        chars.update(c for r in rows for c in r["text"])
        path = WORK/f"{split}.json"
        dump(path, rows)
        hashes[split] = sha(path)
    font = ImageFont.truetype(str(FONT),24)
    codes, by_ink, palette = {}, {}, [np.zeros((32,32),np.uint8)]
    for char in sorted(chars):
        image = bitmap(char,font)
        if image.tobytes() not in by_ink:
            by_ink[image.tobytes()] = len(palette)
            palette.append(image)
        codes[char] = by_ink[image.tobytes()]
    np.save(WORK/"palette.npy", np.stack(palette))
    dump(WORK/"codes.json",codes)
    dump(PUB/"corpus.json",dict(counts=counts,splits=hashes,raw=raw,gentle=EXTERNAL,
         parent_train=sha(OLD/"train.json"),palette=sha(WORK/"palette.npy"),codes=sha(WORK/"codes.json"),
         preparer=sha(Path(__file__)),protocol=sha(ROOT/"references/visual_referent_learning_protocol.md")))
    print(counts)


if __name__=="__main__":
    main()
