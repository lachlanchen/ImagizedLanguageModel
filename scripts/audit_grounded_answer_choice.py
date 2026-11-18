"""Recount the fixed paired preferences and compare with bitmap overlap, without model execution."""
from collections import Counter
import numpy as np

from scripts.prepare_pictured_entry_answers import load, write_json
from scripts.probe_grounded_answer_choice import SRC, WORK, Split, choose_pairs
from scripts.audit_grounded_glyph_writer import paired_interval


def similarity(a, b):
    ax, bx = set(a), set(b)
    return len(ax & bx) / len(ax | bx)


def main():
    ds = Split("development")
    pairs = choose_pairs(ds)
    counts = load(SRC/"answer_choices.json")["counts"]
    successes = {}
    for arm in counts:
        saved = np.load(WORK/arm/"answer_choices.npz")
        assert np.array_equal(pairs,saved["pairs"])
        s = saved["mean_logp"]
        correct = np.stack((s[:,:,0,0]>s[:,:,0,1],s[:,:,1,1]>s[:,:,1,0]),-1)
        successes[arm] = correct.all(axis=(1,2))
        assert int(correct.sum()) == counts[arm]["all"]["individual"]
        assert int(successes[arm].sum()) == counts[arm]["all"]["both_questions_both_layouts"]
    totals = {"direct_jaccard":Counter(), "via_passages":Counter()}
    for i,j in pairs:
        for name in totals:
            actual = []
            for k, qrow in enumerate((i,j)):
                scores = []
                for ay in (i,j):
                    if name == "direct_jaccard":
                        score = similarity(ds.q[qrow], ds.y[ay])
                    else:
                        # Images of both component passages are supplied; no answer-to-passage labels.
                        score = max(similarity(ds.q[qrow],ds.p[p])*similarity(ds.y[ay],ds.p[p]) for p in (i,j))
                    scores.append(score)
                actual.append(scores[k] > scores[1-k])
            totals[name].update(dict(individual=2*sum(actual),both_questions_both_layouts=int(all(actual))))
    audit = dict(pairs=len(pairs), paired_vs_deranged=paired_interval(successes["paired"],successes["deranged"]),
                 bitmap_overlap=totals, ties_get_no_credit=True)
    write_json(SRC/"choice_recount.json",audit)
    print(audit)


if __name__=="__main__":
    main()
