"""Count paired gains, losses and primary adoption conditions on the fixed outputs."""

from collections import defaultdict
from scipy.stats import binomtest
from scripts.prepare_competing_facts import PUB, read, dump
from scripts.evaluate_visual_discourse import unpack


def compare():
    arms = ("parent", "likelihood", "contrast", "parent-bytes", "bytes")
    tables = {arm: unpack(PUB / (arm + "-outputs.json.gz")) for arm in arms}
    results = {}
    cases = {}
    for arm, control in (
        ("likelihood", "parent"),
        ("contrast", "parent"),
        ("contrast", "likelihood"),
        ("bytes", "parent-bytes"),
        ("bytes", "contrast"),
    ):
        groups = defaultdict(list)
        examples = defaultdict(dict)
        current = [v for v in tables[arm] if v["test"] == "human"]
        for a, b in zip(
            current, [v for v in tables[control] if v["test"] == "human"], strict=True
        ):
            if (a["key"], a["kind"]) != (b["key"], b["kind"]):
                raise ValueError("Unpaired comparisons")
            groups[(a["kind"], a["doc"])].append((a["correct"], b["correct"]))
        bins = defaultdict(list)
        for (k, doc), v in groups.items():
            assert len(v) == 2
            after, before = all(a for a, b in v), all(b for a, b in v)
            state = (
                "gain"
                if after and not before
                else "loss"
                if before and not after
                else "both"
                if after
                else "neither"
            )
            lang = "en" if k in ("coqa", "squad") else "zh"
            for key in (k, lang, "all"):
                bins[key].append((after, before))
            for key in (k, lang):
                examples[key].setdefault(state, doc)
        stats = {}
        for k, items in bins.items():
            gain = sum(a and not b for a, b in items)
            loss = sum(b and not a for a, b in items)
            stats[k] = dict(
                n=len(items),
                candidate=sum(a for a, b in items),
                control=sum(b for a, b in items),
                gain=gain,
                loss=loss,
                exact_p=binomtest(gain, gain + loss, p=0.5).pvalue
                if gain + loss
                else 1.0,
            )
        results[arm + "/" + control] = stats
        cases[arm + "/" + control] = examples
    metrics = {a: read(PUB / (a + "-scores.json"))["metrics"] for a in arms}
    passes = all(
        results["contrast/" + base][lang]["candidate"]
        > results["contrast/" + base][lang]["control"]
        for base in ("parent", "likelihood")
        for lang in ("en", "zh")
    )
    passes = (
        passes
        and metrics["contrast"]["qa2d"]["exact_statement"]
        >= metrics["parent"]["qa2d"]["exact_statement"]
    )
    dump(
        PUB / "comparison.json",
        dict(
            paired=results,
            ranked_adoption_rule=passes,
            examples=cases,
            scores={a: v["all"] for a, v in metrics.items()},
        ),
    )
    return cases


if __name__ == "__main__":
    compare()
