"""Check two authentic forms of 言 with retrieved and supplied source regions."""
import torch
from ilm.visual_lm.optical_role_reader import answer
from ilm.visual_lm.portable_writer import read_image
from scripts.figure_role_change_writer import png
from scripts.evaluate_grounded_ink_editor import first_region
from scripts.prepare_grounded_ink_editor import ROOT, PUBLIC,STYLES,record,sha
from scripts.respond_grounded_ink import load
from scripts.train_optical_role_adapter import BASE
from scripts.train_typeface_writer import Face,FACES


def main():
    src=ROOT/"references/evidence/yan_page_responses_20261119"
    out=PUBLIC/"historical";out.mkdir(exist_ok=True)
    reader,writer=load(BASE,"cpu")
    page=read_image(src/"plate.png",34).tobytes()
    qs=[read_image(src/(s+"_question.png"),None).tobytes() for s in ("oracle","bronze")]
    targets=[read_image(src/(s+"_reference.png"),None).tobytes() for s in ("oracle","bronze")]
    predicted=[answer(reader,page,q) for q in qs];face=Face(FACES[0]);rows=[]
    for guide,cells in (("retrieved",predicted),("supplied",targets)):
        points=[first_region(page,c) for c in cells]
        for rev in (0,1):
            a,b=targets[::1-2*rev];gold=a+face("與")+b+face("。")
            actual,end=writer.emit(page,points[::1-2*rev],face(STYLES[0]))
            file=out/f"{guide}_{rev}.png";png(actual,file)
            rows.append(dict(guide=guide,reverse=rev,regions=points,exact=actual==gold and end,stopped=end,cells=len(actual)//128,file=file.name,sha256=sha(file)))
    record(out/"result.json",dict(input_hashes={p.name:sha(p) for p in sorted(src.glob("*.png"))},reader_correct=[x==y for x,y in zip(predicted,targets)],cases=rows))


if __name__=="__main__":
    torch.set_num_threads(2);main()
