"""Build Figure 90 inputs from the scored conditional and autonomous outputs."""
import json
from shutil import copyfile
from scripts.train_causal_glyph_writer import PUBLIC, WORK, record


def main():
    history=json.loads((PUBLIC/"correct_history.json").read_text())
    scores=json.loads((PUBLIC/"scores.json").read_text())
    compare=json.loads((PUBLIC/"first_departure_comparison.json").read_text())
    def profile(arm):
        rows=history[arm]
        return [round(100*rows[k]["correct"]/rows[k]["positions"],1)
                for k in ("in_page","outside_page","first_outside_page")]
    table=[["Training","Single span / 177","Dispersed / 94","Absent / 300","All / 571"]]
    for label,r in (("Shuffled pairs",scores["shuffled_open"]),("Matched pairs",scores["matched_open"]),
                    ("Further replay",compare["replay"]["open"]),("Extra first-glyph weight",compare["departure"]["open"])):
        table.append([label,*[str(r["groups"][k]["exact"]) for k in ("single_span","distributed_ink","absent_content_ink")],str(r["all"]["exact"])])
    g=history["matched"]["first_outside_page"]
    note=f'At the first missing glyph, {g["best_in_page"]} of {g["positions"]} decisions returned ink already visible in the prompt.'
    record(PUBLIC/"figure.json",dict(table=table,series=[profile("shuffled"),profile("matched")],note=note))
    for local,pub in (("0_passage","passage"),("0_question","q0"),("1_question","q1"),("0_answer","a0"),("1_answer","a1")):
        copyfile(WORK/("role_"+local+".png"),PUBLIC/("map_"+pub+".png"))


if __name__=="__main__":main()
