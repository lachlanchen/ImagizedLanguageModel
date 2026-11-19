"""Summarize complete held boards, paired uncertainty and original response strips."""
import json
import shutil
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file
from scripts.grounded_proposition_corpus import WORK,PUBLIC,record,sha
from scripts.evaluate_grounded_propositions import release_notes,load_streams
from scripts.train_causal_glyph_writer import WORK as GPT,in_cells


def load(name):return json.loads((PUBLIC/name).read_text())


def readings():
    rows,meta=release_notes()
    dictionary=dict(zip(map(bytes,load_file(WORK/"planned/interface.safetensors")["packed"]),
                        json.loads((GPT/"audit_legend.json").read_text()),strict=True))
    for r,m in zip(rows,meta):
        for s,z in ((r.p,m["p"]),(r.q,m["q"])):
            for key,c in zip(in_cells(s),z,strict=True):dictionary[key]=c
    sources={}
    for k in ("matched","ordered","replay","varied","planned"):
        y,stops=load_streams(WORK/(k+"_notes.npz"))
        sources[k]=["".join(dictionary.get(c,"□") for c in in_cells(a)) for a in y]
    sources["qwen"]=load("qwen.json")["answers"]
    result=[dict(i=i,q=m["q"],target=m["y"],responses={k:v[i] for k,v in sources.items()}) for i,m in enumerate(meta)]
    record(PUBLIC/"programme_readings.json",result)


def main():
    plan=load("planned_result.json")
    chart_names=("ordered","unordered","varied","planned")
    series=[]
    for a in chart_names:
        c=plan["compound"]["predicted"]["combinations"] if a=="planned" else plan["monolithic_compounds"][a]
        series.append([round(100*c[k]["equivalent"]/c[k]["n"],1) for k in ("practiced","withheld")])
    # Paired bootstrap uses the entire original page and both assignments as the sampling unit.
    boards=json.loads((WORK/"boards.json").read_text())["development"]
    from scripts.extend_grounded_propositions import pair_key,WITHHELD
    conditions={}
    for a,file in (("ordered","ordered_compound_cases.json"),("planned","planned_predicted_compound_cases.json")):
        flags={}
        for r in json.loads((WORK/file).read_text()):flags.setdefault(r["board"],[]).append(r["equivalent"])
        conditions[a]={i:np.mean(v) for i,v in flags.items()}
    deltas={}
    for name,withheld in (("withheld",True),("practiced",False)):
        values=[v-conditions["ordered"][i] for i,v in conditions["planned"].items() if (pair_key(boards[i]) in WITHHELD)==withheld]
        g=np.random.default_rng(2346);sim=g.choice(values,(10000,len(values))).mean(1)
        deltas[name]=dict(boards=len(values),difference=float(np.mean(values)),percentile95=np.percentile(sim,[2.5,97.5]).tolist())
    record(PUBLIC/"board_intervals.json",deltas)
    table=[["Model","Programme / 32","Human CMRC / 29","Human DRCD / 19","Open / 571"]]
    for a,txt in (("matched","Single facts"),("varied","Varied wording"),("planned","Clause planner")):
        d=plan if a=="planned" else load(a+"_result.json")
        h=load(a+"_human.json")["scores"]
        note=d["notes"]["predicted"] if a=="planned" else d["notes"]
        reg=d["regression"]
        table.append([txt,str(note["scores"]["relation_0"]["equivalent"]),str(h["cmrc"]["statement"]),str(h["drcd"]["statement"]),
                      str(sum(reg[k]["exact"] for k in ("single_span","distributed_ink","absent_content_ink")))])
    record(PUBLIC/"figure.json",dict(table=table,series=series))
    ex=WORK/"release_examples"
    for n in ("compound_0_passage","compound_2_passage","compound_0_question","compound_0_answer","compound_2_answer"):
        shutil.copyfile(ex/(n+".png"),PUBLIC/(n+".png"))
    # First development board in the fixed seeded order. Only its emitted
    # factual statements are reproduced here; the corpus text stays local.
    from ilm.visual_lm.causal_glyph_writer import write_strip
    raw,_=load_streams(WORK/"planned_compound.npz")
    for i in (0,2):
        write_strip(np.frombuffer(raw[i],np.uint8).reshape(-1,128),PUBLIC/f"first_board_{i}.png")
    readings()


if __name__=="__main__":main()
