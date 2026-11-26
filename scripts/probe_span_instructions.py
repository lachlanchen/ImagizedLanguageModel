"""Probe translation contrasts, explanations of origins and writing of pictured historical forms."""
from io import BytesIO
from zipfile import ZIP_DEFLATED,ZipFile
import json
import sqlite3
import numpy as np
import torch
from ilm.visual_lm.optical_span_rollout import OpticalSpanWriter
from scripts.reserve_optical_span_rollout import BASE, PUB, ROOT, dump, read, sha
from scripts.prepare_region_writer import Ink
from scripts.evaluate_optical_span_rollout import cells,configure_cuda_cli,gpu_owner,save_ink,write_condition

PAIRS=[
    ("The girl followed the doctor.","女孩跟着医生。"),("The doctor followed the girl.","医生跟着女孩。"),
    ("My sister opened the door.","我姐姐打开了门。"),("My sister did not open the door.","我姐姐没有打开门。"),
    ("The boy bought two red books.","男孩买了两本红色的书。"),("The boy bought three red books.","男孩买了三本红色的书。"),
    ("I ate before he came home.","他回家前我吃了饭。"),("I ate after he came home.","他回家后我吃了饭。")]
WORDS=[("window","Old Norse vindauga, wind and eye"),("alphabet","Greek alpha and beta"),
       ("school","Greek scholē, leisure")]
HANZI={"言":"speech or talking","木":"a tree or wood","山":"a mountain","日":"the sun"}
ARCHIVE=ROOT/"publication/ilm-nature/evidence/word_origin_probes.zip"
STAGES=("oracle","bronze","seal")


def png(tiles):
    buf=BytesIO()
    save_ink(tiles,buf,image_format="PNG")
    return buf.getvalue()


def prompts():
    def row(kind,question,lang,reference,**meta):
        return dict(kind=kind,question=question,to=lang,reference=reference,**meta)
    for j,(en,zh) in enumerate(PAIRS):
        yield row("translation",en,"zh",zh,pair=j//2),[]
        yield row("translation",zh,"en",en,pair=j//2),[]
    for word,ref in WORDS:
        for q,lang in [(f"What is the origin of the English word {word}?","zh"),(f"英语单词{word}的词源是什么？","en")]:
            yield row("English origin",q,lang,ref,word=word,source="https://www.merriam-webster.com/dictionary/"+word),[]
    db=ROOT/"data/chinese_origin_library_v1/origin.sqlite3"
    with sqlite3.connect(db.as_uri()+"?mode=ro",uri=True) as con:
        for char,meaning in HANZI.items():
            source,=con.execute("select fields from entry where headword=? order by entry_id limit 1",(char,)).fetchone()
            fields=json.loads(source)
            forms=[]
            labels=[]
            for stage in STAGES:
                label,packed=con.execute("select label,pixels32 from form join image using(svg_sha) where headword=? and stage=? and has_ink=1 order by label limit 1",(char,stage)).fetchone()
                forms.append(~np.unpackbits(np.frombuffer(packed,dtype=np.uint8)).reshape(32,32).astype(bool))
                labels.append(label)
            for lang in ("zh","en"):
                q=f"What did the old Chinese character {char} originally mean?" if lang=="zh" else f"早期的{char}字本来表示什么？"
                yield row("Chinese origin",q,lang,meaning,word=char,source_meaning=fields["original_meaning"]),[]
                q=f"For {char}, draw these forms in order: oracle, bronze, seal." if lang=="zh" else f"请把{char}的三个字形按甲骨文、金文、篆书的顺序画出来。"
                yield row("historical forms",q,lang,labels,word=char,shown=[labels[i] for i in (2,0,1)]),forms


def main():
    configure_cuda_cli()
    receipt=PUB/"word-origin-images.json"
    if receipt.exists():
        assert sha(ARCHIVE)==read(receipt)["archive"]
        return
    ds=list(prompts())
    ink=Ink(np.load(BASE/"palette.npy"))
    grids=[]
    for r,forms in ds:
        seq=ink.encode(r["question"])
        if ink.bad.intersection(r["question"]): raise ValueError("Missing prompt glyph")
        original=np.unpackbits(np.frombuffer(b"".join(ink.tiles[i] for i in seq),dtype=np.uint8)).reshape(-1,32,32).astype(bool)
        grids.append(np.concatenate([original,np.stack([forms[i] for i in (2,0,1)])]) if forms else original)
    rows=[r for r,forms in ds]
    palette=np.load(BASE/"palette.npy")
    readback=read(BASE/"readback.json")
    alphabet={bytes(v):readback[str(i)] for i,v in enumerate(palette) if i>=2 and str(i) in readback}
    directions=[("en","zh").index(r["to"]) for r in rows]
    partial=ARCHIVE.with_suffix(".tmp")
    with gpu_owner(), ZipFile(partial,"w",compression=ZIP_DEFLATED) as out:
        def record(i,name,data,n,stopped,edges):
            bit=np.packbits(cells(data,n).reshape(n,1024),axis=1)
            forms=ds[i][1]
            packed=[np.packbits(f).tobytes() for f in forms]
            got=[bytes(v) for v in bit]
            rows[i][name]=dict(text="".join(alphabet.get(v,"�") for v in got),stop=bool(stopped),cells=n,edges=edges,
                               ordered_forms=(bool(stopped) and got==packed) if forms else None)
            out.writestr(f"{i:03d}/{name}.png",data)
        for i,g in enumerate(grids):
            out.writestr(f"{i:03d}/input.png",png(g))
            if ds[i][1]: out.writestr(f"{i:03d}/requested.png",png(np.stack(ds[i][1])))
        for i,data,n,stopped in write_condition("guided","cuda",grids,directions,range(len(rows)),12):
            record(i,"single",data,n,stopped,None)
        for t,lang in enumerate(("en","zh")):
            net=OpticalSpanWriter(lang,"cuda")
            available=set(map(bytes,np.packbits(net.ink.cpu().numpy().reshape(len(net.ink),1024),axis=1)))
            ix=[i for i,d in enumerate(directions) if d==t]
            for j in range(0,len(ix),12):
                group=ix[j:j+12]
                for i,(bits,stop,edges) in zip(group,net.write([grids[k] for k in group]),strict=True):
                    record(i,"span",png(bits),len(bits),stop,edges)
                    if ds[i][1]: rows[i]["representable"]=[np.packbits(f).tobytes() in available for f in ds[i][1]]
            del net
        out.writestr("records.json",json.dumps(rows,ensure_ascii=False))
    partial.replace(ARCHIVE)
    dump(receipt,dict(archive=sha(ARCHIVE),prompts=len(rows),model_answers=2*len(rows),
                      protocol=sha(ROOT/"references/span_instruction_probe.md"),program=sha(ROOT/"scripts/probe_span_instructions.py")))
    print("Frozen",len(rows),"prompts",flush=True)


if __name__=="__main__":
    main()
