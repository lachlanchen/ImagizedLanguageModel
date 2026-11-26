"""Measure recoverable meaning with and without the original sentence as editor input."""
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from zipfile import ZipFile
import numpy as np
from scripts.audit_full_image_roles import image_cells
from scripts.evaluate_covered_phrases import ARCHIVE
from scripts.evaluate_ordered_translation import QWEN,JUDGE,server,chat
from scripts.evaluate_visual_library import blind_votes
from scripts.prepare_covered_phrases import PUB,ROOT,WORK,base
from scripts.train_span_choice import gpu_owner
from scripts.verse_gpu_runtime import configure_cuda_cli


def inputs():
    palette=np.load(base.BASE/"palette.npy")
    names=base.read(base.BASE/"readback.json")
    lookup={p.tobytes():names[str(i)] for i,p in enumerate(palette) if i>=2 and str(i) in names}
    jobs=[]
    with ZipFile(ARCHIVE) as z:
        book=json.loads(z.read("records.json"))[:96]
        for i,r in enumerate(book):
            drafts={}
            for m in ("phrase","neural"):
                a=r["answers"][m]
                img=image_cells(z.read(f"{i:04d}/{m}.png"),a["cells"])
                text="".join(lookup[v.tobytes()] for v in img)
                assert text==a["text"]
                drafts[m]=text.replace("　"," ")
            lang=("English","Mandarin Chinese")[r["t"]]
            for k in ("phrase","neural","source_and_phrase"):
                if k=="source_and_phrase":
                    instruction=f"Translate the source into {lang}. You may revise or discard the rough draft. Preserve the complete meaning. Return only the translation."
                    packet=dict(source=r["source"],rough_draft=drafts["phrase"])
                else:
                    instruction=f"Edit this rough {lang} draft into fluent {lang}. Preserve its intended meaning and concrete details. A fragment may remain a fragment. Return only the edited passage."
                    packet=dict(rough_draft=drafts[k])
                jobs.append((i,k,instruction+"\n"+json.dumps(packet,ensure_ascii=False)))
    return book,jobs


def main():
    configure_cuda_cli()
    archive=base.sha(ARCHIVE)
    assert archive==base.read(PUB/"generation.json")["archive"]
    if (PUB/"editing.json").exists():
        assert base.read(PUB/"editing.json")["archive"]==archive
        return
    book,jobs=inputs()
    work=WORK/"edited-drafts.json"
    inherited=base.read(WORK/"comparison.json")
    with gpu_owner():
        if work.exists():
            edits=base.read(work)
            assert edits["archive"]==archive
        else:
            with server() as url,ThreadPoolExecutor(max_workers=4) as workers:
                results=list(workers.map(lambda job:chat(url,QWEN,job[2]),jobs))
            edits=dict(archive=archive,answers=[dict(i=i,condition=k,**response) for (i,k,_),response in zip(jobs,results,strict=True)])
            base.dump(work,edits)
        answers=[]
        for r,c in zip(book,inherited["outputs"],strict=True):
            answers.append(dict(phrase=r["answers"]["phrase"],neural=r["answers"]["neural"],**c))
        for response in edits["answers"]:
            answers[response["i"]]["edited_"+response["condition"]]={k:response[k] for k in ("text","stop")}
        votes=blind_votes(book,answers)
    grades=[]
    counts=defaultdict(Counter)
    for i,(r,a) in enumerate(zip(book,answers,strict=True)):
        g={k:bool(v["stop"] and votes[2*i][k] and votes[2*i+1][k]) for k,v in a.items()}
        grades.append(g)
        for label in ("all",r["doc"],("en","zh")[r["t"]]):
            counts[label].update({k:int(v) for k,v in g.items()})
            counts[label]["n"]+=1
    base.dump(PUB/"editing.json",dict(archive=archive,teacher=QWEN,judge=JUDGE,edits=edits["answers"],
                counts=counts,grades=grades,votes=votes,protocol=base.sha(ROOT/"references/covered_draft_editing_probe.md")))
    print(dict(counts["all"]),flush=True)


if __name__=="__main__":
    main()
