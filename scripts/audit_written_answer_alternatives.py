"""Compare preserved answer images with every renderable annotation for the same human question."""
from collections import Counter
from zipfile import ZipFile
import json
import numpy as np

from scripts.audit_same_passage_writer import EVIDENCE
from scripts.audit_traditional_passage_training import archived
from scripts.prepare_composed_chinese_answers import ARCHIVE
from scripts.prepare_pictured_entry_answers import Rasterizer, load, sha, write_json
from scripts.train_grounded_glyph_writer import DATA, Split
from scripts.train_same_passage_writer import WORK, ARMS


def main():
    ds = Split("development")
    sources = load(DATA/"development/sources.json")
    registry = load(WORK/"result.json")["archives"]
    writers = {}
    for arm in ("parent",*ARMS):
        name = f"{arm}/dureader.npz"
        writers[arm] = archived(WORK/name,registry[name])
    raster = Rasterizer()
    examples = []
    report = {g:Counter() for g in ("all",*set(ds.groups))}
    with ZipFile(ARCHIVE) as z:
        for member in sorted({r["member"] for r in sources}):
            loc = {r["line"]:i for i,r in enumerate(sources) if r["member"]==member}
            with z.open(member) as f:
                for line,raw in enumerate(f):
                    if line not in loc:continue
                    i = loc[line]
                    row = json.loads(raw)
                    meta = sources[i]
                    assert str(row["question_id"])==meta["question_id"]
                    canonical = ds.tiles[ds.y[i]].tobytes()
                    options=[]
                    option_text={}
                    misses=0
                    for a in row["answers"]:
                        try:
                            bitmap=b"".join(raster.tile(c) for c in a.strip())
                            options.append(bitmap)
                            option_text[bitmap]=a.strip()
                        except ValueError: misses+=1
                    assert canonical in options
                    available=set(options)
                    visible=set(map(bytes,ds.tiles[np.r_[ds.p[i],ds.q[i]]]))
                    alt_visible=any(all(s[j:j+128] in visible for j in range(0,len(s),128))
                                    for s in available if len(s))
                    stats=dict(questions=1,annotations=len(row["answers"]),unrenderable_annotations=misses,
                               multiple_annotations=int(len(row["answers"])>1),
                               distinct_renderable_alternative=int(len(available)>1),
                               repeat_of_canonical=int(options.count(canonical)>1),
                               has_an_answer_using_only_visible_ink=int(alt_visible),
                               retrieved_document_marked_selected=int(bool(row["documents"][meta["doc"]]["is_selected"])))
                    for arm,(pixels,ends) in writers.items():
                        hit=bool(ends[i] and pixels[i] in available)
                        stats[arm+"_exact_any"]=int(hit)
                        stats[arm+"_exact_canonical"]=int(ends[i] and pixels[i]==canonical)
                        stats[arm+"_exact_with_missing_ink"]=int(hit and any(
                            pixels[i][j:j+128] not in visible for j in range(0,len(pixels[i]),128)))
                        alnum=bool(hit and any(c.isalnum() and raster.tile(c) not in visible
                                              for c in option_text[pixels[i]]))
                        stats[arm+"_exact_with_missing_alphanumeric"]=int(alnum)
                        if hit and pixels[i]!=canonical:
                            text=option_text[pixels[i]]
                            examples.append(dict(index=i,arm=arm,group=ds.groups[i],question=row["question"],
                                response=text,original_in_source=all(raster.tile(c) in visible for c in text),
                                absent=[c for c in text if raster.tile(c) not in visible],
                                selected_document=row["documents"][meta["doc"]]["is_selected"]))
                    for g in ("all",ds.groups[i]):
                        report[g].update(stats)
    assert report["all"]["questions"]==571
    write_json(EVIDENCE/"natural_answer_alternatives.json",dict(counts=report,sha256=sha(__file__)))
    write_json(WORK/"alternative_matches_private.json",examples)
    print(report)


if __name__=="__main__":
    main()
