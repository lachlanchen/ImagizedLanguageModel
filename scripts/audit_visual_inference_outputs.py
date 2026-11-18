"""Reconstruct the official and validation comparisons directly from archived physical streams."""
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.audit_traditional_passage_training import archived
from scripts.prepare_visual_inference_pairs import DEST, RESPONSES
from scripts.train_visual_inference_writer import PRIVATE, PUBLIC, WORK, Corpus, Rasterizer, load, sha, write_json
from scripts.train_paired_passage import unpack_sequences


def tiles(raw):
    return [raw[i:i+128] for i in range(0,len(raw),128)]


def image(path,raw):
    if not raw:
        img=np.zeros((32,32),np.uint8)
    else:
        arr=np.unpackbits(np.frombuffer(raw,np.uint8)).reshape(-1,32,32)
        img=np.concatenate(arr,axis=1)
    Image.fromarray(255*(1-img)).save(path)


def audit():
    report=load(PUBLIC/"result.json")
    checks={}
    raster=Rasterizer()
    allowed=[b"".join(raster.tile(c) for c in text) for text in RESPONSES]
    role=load(PUBLIC/"role_switch.json")
    with np.load(PRIVATE) as splits:
        for name,subset in (("validation","train"),("official","development")):
            ds=DEST/subset;bank=np.load(ds/"pixels.npy")
            seq=[[bank[s].tobytes() for s in unpack_sequences(ds/(key+".npz"))] for key in ("premises","claims","responses")]
            p,h,y=seq
            rows=splits[name]
            triple=np.searchsorted(rows,splits[name+"_triples"])
            indices=np.load(WORK/(name+"_cloze_rows.npy"))
            def get(path):
                return archived(WORK/path,report["archives"][path])
            holes,_=get(name+"_cloze_requests.npz"); targets,_=get(name+"_cloze_targets.npz")
            assert len(indices)==len(targets)==len(holes)
            absent=[]; classes=[]
            for i,request,target in zip(indices,holes,targets,strict=True):
                a=tiles(request);box=a.index(Corpus.mask_tile())
                ref=tiles(h[i]);hidden=tiles(target)
                prefix_len=len(a)+len(hidden)-1-len(ref)
                assert a[prefix_len:box]+hidden+a[box+1:]==ref and 1<=len(hidden)<=3
                source=tiles(p[i])
                missing=set(hidden)-set(a)-set(source)
                absent.append(bool(missing))
                def visible_in(line):
                    return any(line[k:k+len(hidden)]==hidden for k in range(len(line)-len(hidden)+1))
                if missing==set(hidden):
                    category="every_target_glyph_absent"
                elif missing:
                    category="some_target_glyph_absent"
                elif visible_in(source):
                    category="intact_in_premise"
                elif visible_in(a):
                    category="intact_in_claim_or_instruction"
                else:
                    category="present_as_separate_glyphs"
                classes.append(category)
            absent=np.array(absent)
            rec={}
            for arm in ("parent","paired","mispaired"):
                arms=report["arms"][arm]
                details={}
                for task,ids,refs in (("decision",rows,[y[i] for i in rows]),("restoration",indices,targets)):
                    tag=name+"_"+task
                    for mode in (("intact","passage_erased","claim_erased") if task=="decision" else ("intact","passage_erased")):
                        label=tag+"_"+mode
                        out,end=get(arm+"/"+label+".npz")
                        ok=np.asarray([bool(s and a==b) for s,a,b in zip(end,out,refs,strict=True)])
                        counts=arms[label]
                        assert len(ok)==counts["records"] and ok.sum()==counts["exact"] and end.sum()==counts["stopped"]
                        if task=="decision":
                            assert ok[triple].all(axis=1).sum()==counts["three"]
                        else:
                            assert ok[absent].sum()==counts["absent"] and absent.sum()==counts["absent_total"]
                        if mode!="intact":continue
                        if task=="decision":
                            conf=np.zeros((3,4),int)
                            for pred,stop,ref in zip(out,end,refs,strict=True):
                                conf[allowed.index(ref),allowed.index(pred) if stop and pred in allowed else 3]+=1
                            details["confusion"]=conf.tolist()
                        else:
                            details["visibility"]={}
                            for kind in sorted(set(classes)):
                                selected=np.asarray(classes)==kind
                                answers=[refs[i] for i in np.flatnonzero(ok&selected)]
                                details["visibility"][kind]=dict(total=int(selected.sum()),exact=len(answers),
                                    distinct_exact=len(set(answers)),most_common_exact_count=max(Counter(answers).values(),default=0))
                            sizes=np.array([len(x)//128 for x in refs])
                            details["absent_by_length"]={str(n):[int(ok[absent&(sizes==n)].sum()),int((absent&(sizes==n)).sum())] for n in (1,2,3)}
                            details["all_target_glyphs_missing"]={str(n):[0,0] for n in (1,2,3)}
                            for j,(i,outcome) in enumerate(zip(ids,ok,strict=True)):
                                g=set(tiles(refs[j]))
                                if g.isdisjoint(set(tiles(holes[j]))|set(tiles(p[i]))):
                                    pair=details["all_target_glyphs_missing"][str(sizes[j])]
                                    pair[0]+=int(outcome);pair[1]+=1
                rec[arm]=details
            checks[name]=rec
    folder=PUBLIC
    # Public images use only the two fixed, newly authored CLI sentences.
    cli=load(PUBLIC/"cli.json")
    for i,r in enumerate(cli):
        typed=WORK/"cli"/(str(i)+"_typed")
        grid=WORK/"cli"/(str(i)+"_image")
        answer=(typed/"answer.bin").read_bytes()
        assert r["identical"] and (grid/"answer.bin").read_bytes()==answer and sha(typed/"answer.bin")==r["answer_sha256"]
        if i<2:
            image(folder/f"cli{i}.png",answer)
    for arm in ("parent","paired","mispaired"):
        for task in ("decision","restoration"):
            path=WORK/arm/(task+"_switches.npz")
            images,ended=archived(path,role[arm][task]["sha256"])
            assert len(images)==len(ended)==role[arm][task]["total"]
    write_json(folder/"audit.json",dict(partitions=checks,source_sha256=sha(Path(__file__)),
                                        result_sha256=sha(folder/"result.json"),role_switch_sha256=sha(folder/"role_switch.json")))


if __name__=="__main__":
    audit()
