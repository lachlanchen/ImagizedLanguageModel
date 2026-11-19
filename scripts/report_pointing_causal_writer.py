"""Make the numeric chart source and retain original, unretouched answer strips."""
import json
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PUBLIC=ROOT/"references/evidence/pointing_causal_writer_20261119"
WORK=ROOT/"artifacts/pointing_causal_writer_v1"


def main():
    load=lambda name:json.loads((PUBLIC/(name+".json")).read_text())
    initial=load("selection")["development"]["learned"]["scores"]
    follow=load("short_selection")
    fresh=load("transfer")["scores"]
    chart=[]
    for arm in ("learned","random_windows","sentences"):
        familiar=initial if arm=="learned" else follow[arm]["scores"]
        chart.append([familiar["role"]["exact"],fresh[arm]["exact"]])
    groups=("cmrc","drcd","historical","historical_rephrased","absent_content_ink")
    def values(sc):
        return [str(sc[k]["exact" if k=="absent_content_ink" else "both"]) for k in groups]
    table=[["Answer method","CMRC / 128","DRCD / 128","Original / 96","Reworded / 96","New ink / 300"],
           ["Causal writer",*values(load("scores")["pointing"]["scores"])],
           ["With span reader",*values(follow["sentences"]["scores"])]]
    (PUBLIC/"figure.json").write_text(json.dumps(dict(series=chart,table=table),indent=2)+"\n")
    ex=WORK/"short_release_examples"
    for n in ("yan_2_passage","yan_2_question","yan_3_question","yan_2_expected","yan_3_expected",
              "role_0_question","role_1_question","role_0_expected","role_1_expected"):
        shutil.copyfile(ex/(n+".png"),PUBLIC/(n+".png"))


if __name__=="__main__":main()
