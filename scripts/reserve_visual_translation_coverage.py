"""Reserve fresh archival passages and cross-check the numerical edit task on new wording."""

from collections import Counter
import hashlib
from scripts.prepare_anchor_writer_holdout import main as reserve
from scripts.prepare_screened_visual_targets import PRIOR, ROSTER, EVIDENCE as SCREENED
from scripts.prepare_visual_translation_coverage import DEST, load, write_new
from scripts.probe_screened_visual_changes import DIGITS, NUMBER
from scripts.train_screened_visual_targets import forced_matches, Rasterizer, sha
from scripts.evaluate_anchor_gap_writer import read_rows
from scripts.prepare_phrase_boundary_trial import order


def run():
    # All previous held-out chapter passages are excluded even if they were never read.
    path = DEST / "holdout.json"
    if not path.exists():
        extras = [r["target"] for r in load(SCREENED / "accepted.json")] + [
            r["draft"]
            for p in (PRIOR / "teacher_drafts.json", SCREENED / "teacher_drafts.json")
            for r in load(p)["drafts"]
        ]
        reserve(
            prior_rosters=(ROSTER, SCREENED / "registration.json"),
            prefix="translation-coverage-1",
            quota=64,
            maximum_per_chapter=3,
            evidence=path,
            extra_text=extras,
        )
    hold = load(path)["rows"]
    render = Rasterizer()

    def pixels(s):
        return tuple(map(render.tile, s))

    def digest(s):
        return hashlib.sha256(b"".join(pixels(s))).digest()

    blocked = {
        digest(r[field]) for r in read_rows("train") for field in ("source", "old", "new")
    }
    blocked.update(
        digest(r[field])
        for r in load(SCREENED / "registration.json")["rows"]
        for field in ("source", "target")
    )
    blocked.update(digest(r["target"]) for r in load(SCREENED / "accepted.json"))
    blocked.update(digest(r[field]) for r in hold for field in ("source", "target"))
    pairs = []
    for i, r in enumerate(hold):
        s, t = r["source"], r["target"]
        _, matches = forced_matches(pixels(s), pixels(t))
        links = dict(matches)
        for hit in NUMBER.finditer(s):
            a = hit.start()
            b = links.get(a)
            if b is None or s[a : a + 2] != t[b : b + 2]:
                continue
            new = DIGITS[(DIGITS.index(s[a]) + 2) % len(DIGITS)]
            x, y = s[:a] + new + s[a + 1 :], t[:b] + new + t[b + 1 :]
            hashes = [digest(x), digest(y)]
            if any(h in blocked for h in hashes):
                continue
            blocked.update(hashes)
            pairs.append(
                dict(
                    row=i,
                    original_digit=s[a],
                    new_digit=new,
                    input_slot=a,
                    target_slot=b,
                    source=x,
                    target=y,
                )
            )
            break
    order_rows = sorted(
        range(len(hold)),
        key=lambda i: order("coverage-reading/" + hold[i]["source_sha"]),
    )
    chosen = []
    chapters = set()
    for band in range(4):
        count = 0
        for i in order_rows:
            r = hold[i]
            if r["length_band"] == band and r["chapter"] not in chapters:
                chapters.add(r["chapter"])
                chosen.append(i)
                count += 1
                if count == 6:
                    break
    write_new(
        DEST / "fresh_assessment.json",
        dict(
            holdout_sha=sha(path),
            reading=chosen,
            number_pairs=pairs,
            by_band=dict(Counter(hold[i]["length_band"] for i in chosen)),
            student_updates=0,
        ),
    )
    print("Reading passages:", len(chosen), "quantity pairs:", len(pairs))


if __name__ == "__main__":
    run()
