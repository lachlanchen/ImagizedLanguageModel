"""Check whether short selected passages omitted a reference available in the source documents."""
from collections import Counter, defaultdict
import json
from zipfile import ZipFile

from scripts.prepare_composed_chinese_answers import ARCHIVE
from scripts.train_causal_glyph_writer import OPEN, PUBLIC, WORK, record


def main():
    readings=json.loads((WORK/"private_readings.json").read_text())
    locations=json.loads((OPEN/"development/sources.json").read_text())
    members=defaultdict(dict)
    for i,s in enumerate(locations):
        members[s["member"]][s["line"]]=i
    counts=defaultdict(Counter)
    with ZipFile(ARCHIVE) as z:
        for member,items in members.items():
            end=max(items)
            with z.open(member) as stream:
                for k,raw in enumerate(stream):
                    if k>end:break
                    if k not in items:continue
                    i=items[k];row=json.loads(raw);r=readings[i]
                    assert str(row["question_id"])==locations[i]["question_id"]
                    context=r["passage"]
                    paragraphs=[p for d in row["documents"] for p in d["paragraphs"]]
                    alphabet=set("".join(paragraphs)+r["question"])
                    c=counts[r["group"]]
                    c["cases"]+=1
                    c["reference_in_selected_passage"]+=r["reference"] in context
                    c["reference_in_any_paragraph"]+=any(r["reference"] in p for p in paragraphs)
                    c["all_reference_characters_in_source_documents_or_question"]+=set(r["reference"])<=alphabet
                    c["any_human_reference_in_selected_passage"]+=any(t in context for t in r["all_references"])
                    c["any_human_reference_in_any_paragraph"]+=any(t in p for t in r["all_references"] for p in paragraphs)
    record(PUBLIC/"source_text_coverage.json",counts)
    print(json.dumps(counts,indent=2))


if __name__=="__main__":main()
