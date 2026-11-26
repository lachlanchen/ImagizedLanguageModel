"""Independently recount image-level scores, edited inputs and examples matching the source."""
from collections import Counter, defaultdict
from hashlib import sha256
import json
from zipfile import ZipFile

from scripts.evaluate_origin_value_reader import ARCHIVE, PUB, ROOT, dump, read, sha


def main():
    scored=read(PUB/"sealed.json")
    rep=read(PUB/"replay-and-rephrasing.json")
    extra=ARCHIVE.with_name("grounded_origin_rephrased.zip")
    assert sha(ARCHIVE)==scored["archive"]==rep["original"] and sha(extra)==rep["rephrased"]
    totals=defaultdict(Counter)
    unchanged_fields=Counter()
    with ZipFile(ARCHIVE) as z, ZipFile(extra) as changed:
        rec=json.loads(z.read("records.json"))
        expected={}
        for r in rec:
            i=r["i"];kind=r["kind"];base=f"{i:04d}/{kind}/"
            target=z.read(base+"target.png")
            assert sha256(target).hexdigest()==r["target_sha256"]
            if kind=="original": expected[i]=target
            else:
                first=f"{i:04d}/original/"
                equal_input=z.read(base+"question.png")==z.read(first+"question.png")
                equal_page=z.read(base+"page.png")==z.read(first+"page.png")
                assert (equal_input,equal_page)==((True,False) if kind=="values" else (False,True))
                unchanged_fields[kind+"_identical_target"]+=target==expected[i]
            for key,a in r["models"].items():
                raw=z.read(base+key+".png")
                assert sha256(raw).hexdigest()==a["sha256"]
                ok=a["stop"] and a["cells"]==r["n"] and raw==target
                assert ok==a["exact"]
                totals[kind][key]+=ok
        for r in json.loads(changed.read("records.json")):
            i=r["i"];prefix=f"{i:04d}/"
            assert changed.read(prefix+"question.png")!=z.read(prefix+"original/question.png")
            for key,a in r["models"].items():
                raw=changed.read(prefix+key+".png")
                assert sha256(raw).hexdigest()==a["sha256"]
                original=next(x for x in rec if x["i"]==i and x["kind"]=="original")
                ok=a["stop"] and a["cells"]==original["n"] and raw==expected[i]
                assert ok==a["exact"]
                totals["rephrased"][key]+=ok
    for kind,pair in totals.items():
        expected=rep["counts"] if kind=="rephrased" else scored["counts"][kind]
        assert all(expected[k]==v for k,v in pair.items())
    dump(PUB/"score-audit.json",dict(archives=[sha(ARCHIVE),sha(extra)],counts=totals,edits=unchanged_fields))
    print(totals,unchanged_fields,flush=True)


if __name__=="__main__": main()
