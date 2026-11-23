"""Greedy text reference on the same pages with the expected answer format specified."""
from time import monotonic
import torch
from scripts.evaluate_role_question_binding import local_teacher
from scripts.measure_writer_energy import snapshot,consumed
from scripts.evaluate_grounded_ink_editor import target
from scripts.prepare_grounded_ink_editor import PUBLIC,WORK,read,record,sha
from scripts.train_span_choice import gpu_owner


def benchmark():
    dest=WORK/"text_reference";dest.mkdir(exist_ok=True)
    held=WORK/"held.json";assert sha(held)==read(PUBLIC/"corpus.json")["held"]
    out=[]
    with local_teacher("qwen3:8b-q8_0",dest) as (request,weights,_):
        start=monotonic();energy=snapshot()
        for r in read(held):
            for reverse in (0,1):
                qs=r["questions"][::1-2*reverse]
                for mode in (0,1):
                    rule=("請依兩題的先後，依原文寫出兩個人名，格式為「甲與乙。」"
                          if not mode else "請依兩題的先後抄出各自主語及其行為，格式為「第一個原文分句，而第二個原文分句。」")
                    y=request("依照提供的文字作答。保留原文字形，只給答案。",dict(role="user",content=r["page"]+"\n"+qs[0]+"\n"+qs[1]+"\n"+rule),think=False,options=dict(temperature=0,seed=241175,num_ctx=2048,num_predict=72))
                    end=y["stop"]=="stop";out.append(dict(scene=r["scene"],work=r["work"],world=r["world"],lang=r["lang"],reverse=reverse,mode=mode,guide="text",reply=y["text"],stopped=end,exact=end and y["text"]==target(r,reverse,mode)))
        elapsed=monotonic()-start;cost=consumed(energy,snapshot())
    f=WORK/"text_outputs.json";record(f,out);record(PUBLIC/"text_inference.json",dict(weights=weights,responses=sha(f),count=len(out),seconds=elapsed,energy=cost))


if __name__=="__main__":
    with gpu_owner():
        torch.set_num_threads(2);benchmark()
