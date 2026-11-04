#!/usr/bin/env python3
"""Check host-renderer coverage before turning public instructions into pixels."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from fontTools.ttLib import TTFont

from ilm.visual_lm.canonical_glyph_language_data import CanonicalGlyphRenderConfig


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",default="data/general_instructions/dolly15k")
    p.add_argument("--font",default=CanonicalGlyphRenderConfig().font_path)
    a=p.parse_args();data=Path(a.data);font=Path(a.font)
    out=data/"font_coverage_report.json"
    if out.exists():raise FileExistsError("completed renderer coverage audit exists")
    tt=TTFont(str(font),fontNumber=0,lazy=True);cmap=tt.getBestCmap();tt.close()
    counts={};missing=Counter();files={}
    for split in ("train","development","sealed_test"):
        path=data/(split+".jsonl");files[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
        rows=[json.loads(s) for s in path.read_text().splitlines()]
        bad=0;characters=0;uncovered=0
        for row in rows:
            text=row["prompt_flat"]+row["answer_flat"]
            unseen=[c for c in text if not c.isspace() and ord(c) not in cmap]
            bad+=bool(unseen);uncovered+=len(unseen);characters+=len(text);missing.update(unseen)
        counts[split]=dict(examples=len(rows),examples_with_uncovered_characters=bad,
                           characters=characters,uncovered_occurrences=uncovered)
    report=dict(font_path=str(font),font_sha256=hashlib.sha256(font.read_bytes()).hexdigest(),
        font_face_index=0,source_sha256=files,counts=counts,
        missing_codepoints={f"U+{ord(c):04X}":n for c,n in sorted(missing.items(),key=lambda x:ord(x[0]))},
        model_calls=0,data_modified=False,
        scope="host canonical text renderer only; no claim about unencoded image input capability")
    out.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))


if __name__=="__main__":main()
