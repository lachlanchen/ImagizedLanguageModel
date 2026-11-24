"""Measure when a real answer can be assembled by legal visual source edits."""

from collections import Counter
from functools import lru_cache
import json
from scripts.answer_dialogue_surface import load
from scripts.dialogue_ink_data import read_data
from scripts.dialogue_ink_images import trims
from scripts.dialogue_teacher_stories import absent, new_stories
from scripts.evaluate_optical_passage_qa import normalized
from scripts.qualify_dialogue_teacher import PUB, WORK, dump, sha
from scripts.train_dialogue_evidence import RELEASE
from scripts.train_dialogue_surface import representation


def path_to(reference, choices, alphabet, max_words, max_letters):
    words = tuple(normalized(reference).split())

    @lru_cache(maxsize=None)
    def solve(i, remaining):
        if i == len(words):
            return 0, ()
        if remaining == 0:
            return None
        best = None

        def offer(cost, action, j):
            nonlocal best
            tail = solve(j, remaining - 1)
            if tail is None:
                return
            candidate = (cost + tail[0], (action,) + tail[1])
            if best is None or (candidate[0], len(candidate[1])) < (
                best[0],
                len(best[1]),
            ):
                best = candidate

        for tokens, action in choices.get(words[i], ()):
            if words[i : i + len(tokens)] == tokens:
                offer(0, action, i + len(tokens))
        w = words[i]
        if len(w) <= max_letters and all(c in alphabet for c in w):
            offer(1, ("spell", "".join(alphabet[c] for c in w)), i + 1)
        return best

    return solve(0, max_words)


def measure():
    registry = read_data()
    folder = RELEASE / "ink/human"
    net = load(folder)
    face, unit, _ = representation("ink")
    decode = {face(chr(i)): chr(i) for i in range(33, 127)}
    alphabet = {
        decode[p.numpy().tobytes()].lower(): decode[p.numpy().tobytes()]
        for p in net.palette[1:]
    }
    details = []
    groups = {}
    for group in ("long", "longer"):
        ds, markers, evaluator = new_stories(registry, group)
        counts = Counter()
        for i, r in enumerate(ds["rows"]):
            sources = (ds["pages"][r[1]], r[3], r[4])
            refs = evaluator.gold_data[markers[i]]
            if not absent(refs, sources):
                continue
            choices = {}
            seen = set()
            for source in sources:
                for word in source.split():
                    for raw in trims(face(word), unit):
                        surface = "".join(
                            decode[raw[k : k + unit]] for k in range(0, len(raw), unit)
                        )
                        tokens = tuple(normalized(surface).split())
                        if tokens and tokens not in seen:
                            choices.setdefault(tokens[0], []).append(
                                (tokens, ("copy", surface))
                            )
                            seen.add(tokens)
            paths = []
            for ref in refs:
                path = path_to(ref, choices, alphabet, net.max_words, net.max_letters)
                if path is not None:
                    n, actions = path
                    if len(actions) > net.max_words or normalized(
                        " ".join(w for _, w in actions)
                    ) != normalized(ref):
                        raise ValueError("Unrealizable witness")
                    if sum(a == "spell" for a, _ in actions) != n:
                        raise ValueError("Wrong spelling count")
                paths.append(path)
            finite = [p for p in paths if p is not None]
            best = min([p[0] for p in finite], default=None)
            counts["turns"] += 1
            counts["references"] += len(refs)
            counts["reachable_references"] += len(finite)
            counts["copyable_references"] += sum(p[0] == 0 for p in finite)
            counts["any_copyable"] += best == 0
            counts["requires_spelling"] += best is not None and best > 0
            counts["unreachable"] += best is None
            counts["contains_repeated_word_in_reference"] += any(
                len(z) != len(set(z)) for z in (normalized(ref).split() for ref in refs)
            )
            counts["minimum_spelled_" + str(best)] += 1
            details.append(dict(marker=markers[i], group=group, paths=paths))
        groups[group] = dict(counts)
    file = WORK / "reachability-witnesses.json"
    dump(file, details)
    dump(
        PUB / "reachability.json",
        dict(
            groups=groups,
            witness_sha256=sha(file),
            model=sha(folder / "writer.safetensors"),
        ),
    )
    print(json.dumps(groups))


if __name__ == "__main__":
    measure()
