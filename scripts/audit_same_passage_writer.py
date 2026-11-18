"""Recount held-out answers directly from saved physical outputs and accepted reference intervals."""
from collections import Counter

import numpy as np

from scripts.audit_traditional_passage_training import archived
from scripts.compare_traditional_passage_training import TraditionalPassages
from scripts.evaluate_same_passage_writer import READER, compare_cluster
from scripts.prepare_pictured_entry_answers import load, sha, write_json
from scripts.train_grounded_glyph_writer import Split
from scripts.train_same_passage_writer import PAIRS, ROOT, WORK, ARMS

EVIDENCE = ROOT/"references/evidence/same_passage_writer_20261119"


def recount():
    record = load(WORK/"result.json")
    rows = np.load(PAIRS/"development.npy")
    ds = TraditionalPassages("development")
    du = Split("development")
    ids = np.load(WORK/"question_rows.npy")
    assert np.array_equal(ids,np.unique(rows[:,1:3]))
    refs = {}
    for i in ids:
        p = ds.passages[ds.groups[i]]
        refs[i] = {ds.bank[p[l:r]].tobytes() for l,r in ds.intervals[i]}
    du_refs = [du.tiles[y].tobytes() for y in du.y]
    pair_out, question_out, preferences = {}, {}, {}
    audit = {}
    for arm in ("parent",*ARMS):
        conditions = record["conditions"][arm]
        diagnostic = Counter()
        for mode in ("original","erased","dureader"):
            name = f"{arm}/{mode}.npz"
            img,end = archived(WORK/name, record["archives"][name])
            expect = conditions[mode]
            assert len(img)==expect["records"] and int(end.sum())==expect["stopped"]
            if mode=="dureader":
                correct = [bool(e and a==t) for a,e,t in zip(img,end,du_refs,strict=True)]
                assert sum(correct)==expect.get("exact",0)
                for group in set(du.groups):
                    assert sum(c for c,g in zip(correct,du.groups) if g==group)==expect[group+"_exact"]
            else:
                outputs = dict(zip(ids,img,strict=True))
                ok = {i:bool(e and im in refs[i]) for i,im,e in zip(ids,img,end,strict=True)}
                assert sum(ok.values())==expect.get("exact",0)
                joint = []
                for _,a,b,*_ in rows:
                    assert refs[a].isdisjoint(refs[b])
                    joint.append(ok[a] and ok[b])
                    if mode=="original":
                        diagnostic["at_least_one_correct"]+=int(ok[a] or ok[b])
                        diagnostic["siblings_with_identical_answer"]+=int(outputs[a]==outputs[b])
                assert sum(joint)==expect["both"]
                if mode=="original":
                    pair_out[arm] = np.array(joint)
                    question_out[arm] = [ok[i] for i in ids]
        scores = np.load(WORK/arm/"candidate_mean_logp.npy")
        assert sha(WORK/arm/"candidate_mean_logp.npy")==record["archives"][arm+"/candidate_mean_logp.npy"]
        prefs = (scores[:,0]>scores[:,1]) & (scores[:,3]>scores[:,2])
        assert int(prefs.sum())==conditions["provided_answers"]["both"]
        preferences[arm] = prefs
        diagnostic["recognizes_both_but_does_not_generate_both"] = int((prefs & ~pair_out[arm]).sum())
        diagnostic["generates_both_but_canonical_preference_fails"] = int((~prefs & pair_out[arm]).sum())
        audit[arm] = dict(diagnostic)
    dig=load(READER/"mixed_result.json")["scores"]["drcd"]["archive_sha256"]
    streams,stops=archived(READER/"mixed_drcd.npz",dig)
    ok={i:bool(stops[i] and streams[i] in refs[i]) for i in ids}
    assert sum(ok.values()) == record["conditions"]["released_span_reader"]["exact"]
    reader_pairs=np.array([ok[a] and ok[b] for _,a,b,*_ in rows])
    assert reader_pairs.sum()==record["conditions"]["released_span_reader"]["both"]
    comparisons = {arm:compare_cluster(pair_out[arm],reader_pairs,rows[:,0]) for arm in ARMS}
    EVIDENCE.mkdir(exist_ok=True)
    write_json(EVIDENCE/"result.json",record)
    write_json(EVIDENCE/"recount.json",dict(diagnostics=audit,vs_released_reader=comparisons,
               generation_vs_likelihood_on_unique_questions=compare_cluster(question_out["contrast"],
                                  question_out["likelihood"],ds.groups[ids]), distinct_questions=len(ids),
               distinct_passages=len(np.unique(rows[:,0])), source_sha256=sha(__file__)))
    print(audit,comparisons)


if __name__=="__main__":
    recount()
