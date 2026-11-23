"""Build the table, chart series and signed original response pixels for the paper."""

from scripts.figure_role_change_writer import png
from scripts.prepare_role_question_binding import PUBLIC,WORK,read
from scripts.train_causal_glyph_writer import record,sha
from scripts.train_typeface_writer import Face,FACES


def figure():
    keys=["classical/zh","classical/en","novels/zh","novels/en"]
    systems=[("original","Previous reader"),("human","Answer loss"),("aligned","Answer + alignment"),("without_question","Generic question"),("text","Text Qwen3-8B")]
    series=[];table=[["Reader","Chinese exact","Chinese complete","English exact","English complete"]]
    for arm,title in systems:
        c=read(PUBLIC/(arm+"_scores.json"))["counts"]
        if arm in ("original","human","aligned"):
            series.append([round(100*c[k]["individual"]/c[k]["questions"],1) for k in keys])
        fields=[]
        for lang in ("zh","en"):
            row=c["classical/"+lang]
            fields.extend([f"{row['individual']} / {row['questions']}",f"{row['whole']} / {row['scenes']}"])
        table.append([title,*fields])
    f=WORK/"aligned_responses.json";assert sha(f)==read(PUBLIC/"aligned_scores.json")["response_sha"]
    raw=read(WORK/"classical.json")[0];face=Face(FACES[0])
    responses={ (r["world"],r["h"]):bytes.fromhex(r["reply"]) for r in read(f) if r["key"]==raw["key"] and r["lang"]=="zh" and r["group"]=="classical"}
    strips=[]
    for w,page in enumerate(raw["pages"]):
        png(face(page),PUBLIC/f"page{w}.png")
        for h,q in enumerate(raw["qs"]):
            base=f"reply{w}_{h}"
            png(face(q["zh"]),PUBLIC/(base+"_q.png"))
            png(face(raw["qs"][w^h]["answer"]),PUBLIC/(base+"_gold.png"))
            png(responses[w,h],PUBLIC/(base+"_out.png"))
            strips.append(dict(base=base,work="As printed" if w==0 else "Names exchanged"))
    record(PUBLIC/"figure.json",dict(table=table,series=series,strips=strips))


if __name__=="__main__":
    figure()
