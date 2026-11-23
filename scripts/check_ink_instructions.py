"""Test unseen pictured instruction wordings with the same learned image editor."""
from collections import defaultdict
import torch
from scripts.prepare_grounded_ink_editor import PUBLIC,WORK,ROOT,read,record,sha
from scripts.evaluate_grounded_ink_editor import first_region,target
from scripts.respond_grounded_ink import load
from scripts.train_optical_role_adapter import BASE
from scripts.train_typeface_writer import Face,FACES
from scripts.figure_role_change_writer import png
from ilm.visual_lm.optical_role_reader import answer
from ilm.visual_lm.portable_writer import read_image


def main():
    net,writer=load(BASE,"cpu","original");face=Face(FACES[0])
    requests=(("把兩人的名字列出來","寫出二人姓名"),("分別敘述各自所做的事","寫明兩人各做了什麼"))
    positions=WORK/"regions.json";assert sha(positions)==read(PUBLIC/"reading.json")["positions"]
    rows=read(WORK/"held.json");data=defaultdict(list);out=[]
    for r,regions in zip(rows,read(positions),strict=True):
        page=face(r["page"])
        for mode,wordings in enumerate(requests):
            for style in wordings:
                for rev in (0,1):
                    y,end=writer.emit(page,regions["predicted"][::1-2*rev],face(style))
                    ok=end and y==face(target(r,rev,mode))
                    data[style,r["lang"]].append(ok)
                    out.append(dict(work=r["work"],scene=r["scene"],world=r["world"],lang=r["lang"],reverse=rev,instruction=style,exact=ok,stopped=end,reply=y.hex()))
    arr=WORK/"instruction_outputs.json";record(arr,out)
    src=ROOT/"references/evidence/yan_page_responses_20261119";p=read_image(src/"plate.png",34).tobytes()
    queried=[answer(net,p,read_image(src/(s+"_question.png"),None).tobytes()) for s in ("oracle","bronze")]
    refs=[read_image(src/(s+"_reference.png"),None).tobytes() for s in ("oracle","bronze")]
    points=[first_region(p,im) for im in queried]
    hist=[]
    for i,style in enumerate(("列出兩種字形","請並列這兩種字形","把兩種字形依次寫出來","只列出這兩個結果")):
        for rev in (0,1):
            y,end=writer.emit(p,points[::1-2*rev],face(style));a,b=refs[::1-2*rev]
            f=PUBLIC/"historical"/f"wording{i}_{rev}.png";png(y,f)
            hist.append(dict(style=style,reverse=rev,stopped=end,exact=end and y==a+face("與")+b+face("。"),file=f.name))
    record(PUBLIC/"instruction_transfer.json",dict(book_scores={"/".join(k):[sum(v),len(v)] for k,v in data.items()},book_replies=sha(arr),historical=hist))


if __name__=="__main__":
    torch.set_num_threads(2);main()
