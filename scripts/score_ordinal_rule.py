"""Compare an explicit recency-order shortcut with fixed natural ordinal decisions."""

import gzip
import json
from scripts.ordered_reference_data import natural
from scripts.prepare_story_mentions import PUBLIC, dump
from scripts.story_mention_options import collapse
from scripts.train_reference_order import sides


def compare():
    results = {}
    for language, rows in natural()[0].items():
        learned = json.loads(
            gzip.decompress(
                (PUBLIC / f"order-{language}-decisions.json.gz").read_bytes()
            )
        )["ink"]
        records = []
        for row, choice in zip(rows, learned, strict=True):
            text, q = row["text"], row["query"]
            idx, slot, spans, bounds = collapse(text, q, row["cast"])
            order = sides(slot, bounds[0])
            cue = text[slice(*q)]
            position = None
            if order is not None:
                key = order[int(cue.lower() in ("latter", "后者", "後者"))]
                position = next(
                    spans[j] for j in reversed(range(bounds[0])) if slot[j] == key
                )
            target = (
                None
                if row["truth"] is None
                else text[slice(*row["cast"][row["truth"]])]
            )
            extract = lambda pair: None if pair is None else text[slice(*pair)]
            records.append(
                dict(
                    id=row["id"],
                    cue=cue,
                    rule=extract(position) == target,
                    learned=extract(choice) == target,
                    negative=target is None,
                )
            )
        results[language] = dict(
            n=len(rows),
            rule=sum(r["rule"] for r in records),
            learned=sum(r["learned"] for r in records),
            decisions=records,
        )
    return results


if __name__ == "__main__":
    scores = compare()
    dump(PUBLIC / "ordinal-rule.json", scores)
    print({key: (v["rule"], v["learned"]) for key, v in scores.items()})
