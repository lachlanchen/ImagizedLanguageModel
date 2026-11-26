"""Attribute missed book translations to missing drafts or wrong reverse-score selection."""
from collections import Counter
import json
from pathlib import Path
from zipfile import ZipFile

from scripts.collect_sentence_drafts import score_reverse
from scripts.evaluate_sentence_reconstruction import ARCHIVE, PUB, ROOT, configure_cuda_cli, gpu_owner, dump, read, sha
import torch


def main():
    evidence = read(PUB/"assessment.json")
    assert evidence["archive"]==sha(ARCHIVE)
    path = PUB/"routing.json"
    if path.exists():
        assert read(path)["assessment"]==sha(PUB/"assessment.json")
        return
    with ZipFile(ARCHIVE) as z:
        records = json.loads(z.read("records.json"))
    grades = evidence["primary"]["records"]
    cases = [None]*len(records)
    totals, by_direction = Counter(), {}
    configure_cuda_cli()
    with gpu_owner():
        torch.set_num_threads(2)
        for t,lang in enumerate(("en","zh")):
            idx = [i for i,r in enumerate(records) if r["direction"]==lang]
            samples,originals = [],[]
            for i in idx:
                row = records[i]
                samples += [row["answers"][arm]["text"] for arm in ("preference","likelihood")] + [row["reference"]]
                originals += [row["source"]]*3
            scores,_ = score_reverse(("zh","en")[t],samples,originals)
            stats = Counter()
            for j,i in enumerate(idx):
                a,b,ref = scores[3*j:3*j+3]
                selected = "preference" if a>=b else "likelihood"
                g = grades[i]
                possible = g["preference"] or g["likelihood"]
                part = "routed_correct" if g[selected] else ("routing_miss" if possible else "neither_correct")
                stats.update([part])
                stats["either_correct"]+=possible
                stats["reference_above_both"]+=ref>max(a,b)
                stats["n"]+=1
                cases[i]=dict(index=i,reverse_scores=[a,b,ref],selected=selected,part=part)
            assert stats["routed_correct"]+stats["routing_miss"]==stats["either_correct"]
            assert stats["either_correct"]+stats["neither_correct"]==stats["n"]
            by_direction[lang]=dict(stats)
            totals.update(stats)
    dump(path,dict(total=totals,directions=by_direction,cases=cases,assessment=sha(PUB/"assessment.json"),
                   proof=sha(ROOT/"formal/span_list/DraftReconstruction.lean"),program=sha(Path(__file__))))
    print(totals,flush=True)


if __name__=="__main__":
    main()
