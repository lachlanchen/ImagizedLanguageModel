"""Author a held-out English request for each of 96 independently selected dictionary intents."""
import fcntl
import json
from pathlib import Path

from scripts.author_entry_requests import MODEL, choose
from scripts.local_translation_teacher import local_teacher
from scripts.prepare_pictured_entry_answers import DEFAULT, load, sha, write_json
from scripts.train_pictured_entry_writer import ROOT

WORK = ROOT / "artifacts/english_entry_authorship_v1"
EVIDENCE = ROOT / "references/evidence/english_request_writer_20261119"
SYSTEM = ("Write one natural English question a reader might ask about a Chinese dictionary entry. "
          "Use the given head character itself. Ask only for the specified fact and for the "
          "specified written form. If other information is already known, make the distinction "
          "clear. Return only the question, without any answer or explanation.")
STAGES = {"oracle": "oracle-bone inscriptions", "bronze": "bronze inscriptions",
          "seal": "seal script", "liushutong": "the Liushutong collection"}
FIELDS = {"本义": "earliest meaning", "读音": "main modern Mandarin pronunciation"}


def main():
    WORK.mkdir(exist_ok=True)
    EVIDENCE.mkdir(exist_ok=True)
    ledger = WORK / "requests.jsonl"
    out = EVIDENCE / "english.jsonl"
    if out.exists():
        raise FileExistsError(out)
    rows = load(DEFAULT / "development/index.json")
    schedule = list(choose())
    existing = [json.loads(l) for l in ledger.read_text().splitlines()] if ledger.exists() else []
    assert [m["row"] for m in existing] == [s["row"] for s in schedule[:len(existing)]]
    with (ROOT/"artifacts/ilm_gpu0.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with local_teacher(MODEL, WORK) as (ask, digest, size), ledger.open("a") as stream:
            for i in range(len(existing),len(schedule)):
                s = schedule[i]
                j = s["row"]
                m = rows[j]
                instruction = f"Head {m['head']}. Wants {FIELDS[m['field']]} and {STAGES[m['stage']]}."
                if s["group"] % 2:
                    instruction += (f" Already knows {FIELDS[rows[j ^ 2]['field']]} and has seen "
                                    f"{STAGES[rows[j ^ 1]['stage']]}. Explicitly distinguish these.")
                result = ask(SYSTEM,dict(role="user",content=instruction),think=False,
                             options=dict(seed=27619+i,temperature=.8,num_predict=130,num_ctx=2048))
                stream.write(json.dumps(dict(row=j,group=s["group"],instruction=instruction,**result),
                                        ensure_ascii=False)+"\n")
                stream.flush()
    out.write_bytes(ledger.read_bytes())
    assert len(out.read_text().splitlines())==len(schedule)==96
    write_json(EVIDENCE/"authorship.json",dict(model=MODEL,digest=digest,bytes=size,system=SYSTEM,
               n=len(schedule),sha256=sha(out),source=sha(Path(__file__))))


if __name__=="__main__":
    main()
