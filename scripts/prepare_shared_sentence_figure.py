"""Regenerate Figure 86's native numeric series from the archived study and referenced images."""
from PIL import Image
from scripts.train_interval_writer import PREVIOUS,PUBLIC,load,write_json


def prepare():
    study=load(PUBLIC/"result.json")["arms"]
    calibrated=load(PUBLIC/"calibrated.json")["scores"]
    explicit=load(PUBLIC/"explicit_result.json")
    prior=load(PREVIOUS/"result.json")["arms"]["matched"]
    table=[["Writer","Who / 387","Reorder / 404","Bronze prose / 32","Roles / 96"]]
    for label,arm,model in (("Frozen writer","parent",prior),("Matched","matched",study["matched"]),
                            ("Calibrated","calibrated",calibrated)):
        who=calibrated["overlap_who"] if arm=="calibrated" else study[arm]["overlap_who"]
        role=calibrated["explicit"] if arm=="calibrated" else explicit[arm]
        table.append([label,str(who["exact"]),str(model["familiar_write"]["exact"]),
                      str(model["historic_describe"]["exact"]),str(role["exact"])])
    table.append(["Qwen3-8B","—","—","—",str(explicit["qwen_text"]["exact"])])
    systems=[study["parent"],study["randomized"],study["matched"],calibrated,study["passage_reader"]]
    dimensions={}
    for f in PUBLIC.glob("pair*.png"):
        with Image.open(f) as image:dimensions[f.stem]=image.size
    write_json(PUBLIC/"figure.json",dict(table=table,series=[[r["same_sentence"]["both"] for r in systems]],
                                        dimensions=dimensions))


if __name__=="__main__":
    prepare()
