"""Bind the results table and editable chart to the saved model evaluations."""
import json
from scripts.prepare_visual_facts import PUBLIC,write_json


def report():
    measured=json.loads((PUBLIC/"results.json").read_text())
    generated=json.loads((PUBLIC/"writer_results.json").read_text())
    groups=[("Human questions",("cmrc","drcd")),("Programme notes",("notes",)),("DuIE passages",("duie",))]
    series=[]
    for metric in ("subject","relation","object","statement"):
        values=[]
        for _,keys in groups:
            scores=measured["genuine"]["groups"]
            values.append(round(100*sum(scores[k][metric] for k in keys)/sum(scores[k]["n"] for k in keys),1))
        series.append(values)
    table=[["Questions","Inputs","Disrupted fit","Induced sentence","Both word orders"]]
    for title,keys in groups:
        a,b,c=measured["shuffled"]["groups"],measured["genuine"]["groups"],generated["adapted"]["predicted"]["groups"]
        table.append([title,*map(str,(sum(b[k]["n"] for k in keys),sum(a[k]["statement"] for k in keys),
            sum(b[k]["statement"] for k in keys),sum(c[k]["both_orders"] for k in keys)))])
    write_json(PUBLIC/"figure.json",dict(series=series,table=table))


if __name__=="__main__":report()
