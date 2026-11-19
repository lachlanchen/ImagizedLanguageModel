"""Bind the visual-interface chart, table and original image outputs for Figure 88."""

import json
from shutil import copyfile
from PIL import Image

from scripts.fit_font_invariant_reader import PUBLIC,PARENT_PUBLIC,WORK,write_json


def prepare():
    load=lambda p:json.loads(p.read_text())
    old=load(PARENT_PUBLIC/"result.json")["scores"]["fresh"]
    metric=load(PUBLIC/"result.json")["scores"]["fresh"]
    combined=load(PUBLIC/"responder.json")["arms"]
    qwen=load(PUBLIC/"qwen.json")
    table=[["System","Fresh / 64","Four correct / 16","Person QA / 387","Bronze prose / 32"],
           ["Calibrated writer","13","0","148","30"],
           ["Combined, regular",str(combined["fresh_sans"]["exact"]),str(combined["fresh_sans"]["all_four"]),
            str(combined["overlap_who"]["exact"]),str(combined["historic_describe"]["exact"])],
           ["CKIP, text",str(old["native"]["exact"]),str(old["native"]["all_four"]),"—","—"],
           ["Qwen3-8B, text",str(qwen["exact"]),str(qwen["all_four"]),"—","—"]]
    # Two matched series on the two common faces. Kai was evaluated for the second model only.
    series=[[arm[face]["exact"] for face in ("sans","bold")] for arm in (old,metric)]
    sizes={}
    for tag,index in (("fresh_bold",0),("fresh_bold",1),("historic_describe",0)):
        for kind in ("passage","question","answer"):
            name=f"{tag}_{index}_{kind}"
            if tag=="fresh_bold" and index==1 and kind=="passage":continue
            target=PUBLIC/(name+".png")
            copyfile(WORK/"responder"/target.name,target)
            with Image.open(target) as im:sizes[name]=list(im.size)
    write_json(PUBLIC/"figure.json",dict(series=series,table=table,dimensions=sizes))


if __name__=="__main__":
    prepare()
