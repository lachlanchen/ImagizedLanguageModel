"""Change only the optical reader supplying the trained editor's two source regions."""
import torch
from scripts.evaluate_grounded_ink_editor import first_region,target
from scripts.prepare_grounded_ink_editor import PUBLIC,WORK,STYLES,ROOT,read,sha,record
from scripts.train_optical_role_adapter import BASE
from scripts.respond_grounded_ink import load
from ilm.visual_lm.optical_role_reader import embeddings,read_many,answer,restore
from ilm.visual_lm.book_participant_reader import InkBank
from ilm.visual_lm.portable_writer import read_image
from scripts.figure_role_change_writer import png
from scripts.train_typeface_writer import Face,FACES


@torch.inference_mode()
def run():
    device="cpu"
    face=Face(FACES[0]);net=restore(BASE,device);bank=InkBank()
    from ilm.visual_lm.grounded_ink_editor import GroundedInkEditor
    from safetensors.torch import load_file
    writer=GroundedInkEditor().eval()
    weight=WORK/"aligned/editor.safetensors";assert sha(weight)==read(PUBLIC/"aligned_fit.json")["weights"]
    writer.load_state_dict(load_file(str(weight)))
    rows=read(WORK/"held.json");assert sha(WORK/"held.json")==read(PUBLIC/"corpus.json")["held"]
    pairs=[(bank.encode(face(r["page"])),bank.encode(face(q))) for r in rows for q in r["questions"]]
    emb=embeddings(net,bank);anchors=[]
    for i in range(0,len(pairs),24):
        for img in read_many(net,emb,pairs[i:i+24]): anchors.append(b"".join(bank.cells[j] for j in img))
    out=[]
    for i,r in enumerate(rows):
        p=face(r["page"]);points=[first_region(p,reg) for reg in anchors[2*i:2*i+2]]
        for rev in (0,1):
            for mode in (0,1):
                y,end=writer.emit(p,points[::1-2*rev],face(STYLES[mode]))
                out.append(dict(scene=r["scene"],work=r["work"],lang=r["lang"],world=r["world"],reverse=rev,mode=mode,regions=points,reply=y.hex(),stopped=end,exact=end and y==face(target(r,rev,mode))))
    file=WORK/"parent_editor.json";record(file,out)
    src=ROOT/"references/evidence/yan_page_responses_20261119"
    hist=PUBLIC/"historical";page=read_image(src/"plate.png",34).tobytes()
    reg=[answer(net,page,read_image(src/(n+"_question.png"),None).tobytes()) for n in ("oracle","bronze")]
    gold=[read_image(src/(n+"_reference.png"),None).tobytes() for n in ("oracle","bronze")]
    full=[];coords=[first_region(page,im) for im in reg]
    for rev in (0,1):
        y,end=writer.emit(page,coords[::1-2*rev],face(STYLES[0]))
        a,b=gold[::1-2*rev];full.append(end and y==a+face("與")+b+face("。"))
        png(y,hist/f"parent_{rev}.png")
    stats={}
    for w in sorted({r["work"] for r in out}):
        for l in ("zh","en"):
            for m in (0,1):
                rs=[r for r in out if (r["work"],r["lang"],r["mode"])==(w,l,m)]
                stats[f"{w}/{l}/{m}"]=[sum(r["exact"] for r in rs),sum(all(r["exact"] for r in rs if r["scene"]==i) for i in {r["scene"] for r in rs})]
    record(PUBLIC/"parent_comparison.json",dict(book_scores=stats,book_replies=sha(file),historical_source_correct=[x==y for x,y in zip(reg,gold)],historical_regions=coords,historical_complete=full,weights=sha(weight)))


if __name__=="__main__":
    torch.set_num_threads(2);run()
