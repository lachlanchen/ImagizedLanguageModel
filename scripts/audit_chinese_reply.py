"""Compare fixed slot baselines and read the prose remaining after copying values."""

from collections import Counter, defaultdict
import gzip
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scripts.prepare_chinese_card_reply import (
    WORK,
    PUB,
    read,
    dump,
)


def main():
    train = read(WORK / "train.json")
    held = read(WORK / "test.json")
    raw = json.loads(gzip.decompress((PUB / "outputs.json.gz").read_bytes()))
    answers = raw["outputs"]["ink"]
    counts = defaultdict(Counter)
    for r in train:
        counts[r["domain"]][r["key"]] += 1
    vec = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), dtype=np.float32)
    fit = vec.fit_transform([r["q"] for r in train])
    test = vec.transform([r["q"] for r in held])
    near = []
    keywords = []
    for i, r in enumerate(held):
        indices = np.array(
            [
                j
                for j, s in enumerate(train)
                if s["domain"] == r["domain"] and s["key"] in r["card"]
            ]
        )
        similarities = (test[i] @ fit[indices].T).toarray()[0]
        near.append(train[indices[similarities.argmax()]]["key"] == r["key"])
        present = [k for k in r["card"] if k in r["q"]]
        predicted = (
            min(present, key=r["q"].index)
            if present
            else max(r["card"], key=counts[r["domain"]].get)
        )
        keywords.append(predicted == r["key"])
    qs = {r["q"] for r in train}
    entities = {r["name"] for r in train}
    splits = {
        "all": np.arange(len(held)),
        "no_literal_field": np.array(
            [i for i, r in enumerate(held) if r["key"] not in r["q"]]
        ),
        "new_question_no_literal": np.array(
            [
                i
                for i, r in enumerate(held)
                if r["q"] not in qs and r["key"] not in r["q"]
            ]
        ),
        "both_unseen": np.array(
            [
                i
                for i, r in enumerate(held)
                if r["q"] not in qs and r["name"] not in entities
            ]
        ),
    }
    summary = {}
    for name, index in splits.items():
        summary[name] = dict(
            n=len(index),
            nearest=int(np.asarray(near)[index].sum()),
            keyword=int(np.asarray(keywords)[index].sum()),
            image=sum(answers[i]["route"] for i in index),
            byte=sum(raw["outputs"]["byte"][i]["route"] for i in index),
        )
    known = {r["masked"] for r in train}

    def meaningful(s):
        return sum("一" <= c <= "\u9fff" for c in s)

    summary["phrasing"] = dict(
        masked_unique=len({r["masked"] for r in answers}),
        unseen_scaffold=sum(r["masked"] not in known for r in answers),
        at_least_two_written_han=sum(meaningful(r["masked"]) >= 2 for r in answers),
        human_at_least_two_written_han=sum(meaningful(r["masked"]) >= 2 for r in held),
        no_written_han=sum(meaningful(r["masked"]) == 0 for r in answers),
        frequent=Counter(r["masked"] for r in answers).most_common(12),
    )
    dump(PUB / "lexical-baselines.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
