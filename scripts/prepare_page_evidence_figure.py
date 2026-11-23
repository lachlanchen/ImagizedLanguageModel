"""Collect original bitmap strips and measured board counts for Figure 98."""
import json
from pathlib import Path
import shutil

from scripts.page_evidence_corpus import DEFAULT
from scripts.train_page_writer_requests import VARIED
from scripts.prepare_visual_facts import ROOT
from scripts.train_causal_glyph_writer import record

PUBLIC=ROOT/"references/evidence/page_evidence_writer_20261124"


def load(path):
    return json.loads(path.read_text())


def main():
    PUBLIC.mkdir(exist_ok=True)
    for run,prefix,files in ((DEFAULT,"base",("corpus","page_fit","fields_fit","training","evaluation/results",
                                "evaluation/examples","familiar/results","transfer/results")),
                               (VARIED,"varied",("corpus","page_fit","evaluation/results",
                                "evaluation/examples","transfer/results"))):
        for n in files:
            shutil.copyfile(run/(n+".json"),PUBLIC/(prefix+"_"+n.replace("/","_")+".json"))
    for n, dest in (("audit","pixel_recount"),("clause_selection/results","competing_clauses"),
                    ("inscriptions/results","inscription_cases")):
        shutil.copyfile(VARIED/(n+".json"),PUBLIC/(dest+".json"))
    yan=PUBLIC/"yan"
    yan.mkdir(exist_ok=True)
    for path in (VARIED/"inscriptions").glob("*.png"):
        if "fields_parent" not in path.name:
            shutil.copyfile(path,yan/path.name)
    measured=load(VARIED/"evaluation/results.json")
    earlier=load(DEFAULT/"evaluation/results.json")["arms"]
    fresh=measured["varied"]["boards"]
    before=measured["one_instruction"]["boards"]
    table=[["Complete responses","Cases","Fields only¹","Page, one wording","Page, varied"]]
    for task,title in (("duie","DuIE, both word orders"),("cmrc","CMRC, both word orders"),
                       ("clause","Attested clause"),("single","Changed year, both replies"),("dual","Two paragraphs, all four")):
        fields=earlier["fields"]["groups"][task]
        table.append([title,str(fields["boards"]),str(fields["complete"]),
                      str(earlier["page"]["groups"][task]["complete"]),str(measured["previous_held"][task]["complete"])])
    counts=[[before["single"]["all_wordings"],fresh["single"]["all_wordings"]],
            [before["dual"]["all_wordings"],fresh["dual"]["all_wordings"]]]
    record(PUBLIC/"figure.json",dict(series=counts,table=table))
    for n in ("original_output","edit_output","failure_reference","failure_output"):
        shutil.copyfile(VARIED/"evaluation"/(n+".png"),PUBLIC/(n+".png"))


if __name__=="__main__":
    main()
