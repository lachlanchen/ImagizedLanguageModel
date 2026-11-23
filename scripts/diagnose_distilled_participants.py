"""Locate repeated answers and rival-name errors in the frozen book replies."""

from collections import Counter

from scripts.audit_book_span_distillation import read, verified
from scripts.prepare_book_span_distillation import LAST, PREVIOUS, PUBLIC, WORK
from scripts.train_causal_glyph_writer import record, sha
from scripts.train_typeface_writer import Face, FACES


def diagnose():
    path = LAST / "reserved.json"
    assert sha(path) == read(PREVIOUS / "sources.json")["reserved"]["sha256"]
    pages = read(path)
    face = Face(FACES[0])
    groups = {}
    for arm in ("matched", "exchanged", "human", "teacher"):
        if arm == "teacher":
            teacher = verified(
                WORK / "teacher_reserved.json", PUBLIC / "teacher_reserved.json"
            )
        else:
            detail = verified(
                WORK / f"{arm}_responses.json", PUBLIC / f"{arm}_reserved.json"
            )
        by_lang = {}
        for lang, offset in (("zh", 0), ("en", 2)):
            counts = Counter()
            for i, row in enumerate(pages):
                if arm == "teacher":
                    replies = teacher[i * 4 + offset : i * 4 + offset + 2]
                else:
                    assert row["page_sha"] == detail[i]["key"]
                    replies = [
                        v["image"] for v in detail[i]["replies"][offset : offset + 2]
                    ]
                refs = [face(name).hex() for name in row["names"]]
                assert refs[0] != refs[1]
                same = replies[0] == replies[1]
                counts["same_reply_to_both_questions"] += same
                counts["different_replies"] += not same
                counts["both_correct"] += replies == refs
                counts["both_exchanged"] += replies == refs[::-1]
                for h, pred in enumerate(replies):
                    counts[
                        "requested_name"
                        if pred == refs[h]
                        else "other_name"
                        if pred == refs[1 - h]
                        else "neither_name"
                    ] += 1
            assert counts["same_reply_to_both_questions"] + counts[
                "different_replies"
            ] == len(pages)
            assert sum(
                counts[k] for k in ("requested_name", "other_name", "neither_name")
            ) == 2 * len(pages)
            by_lang[lang] = dict(counts)
        groups[arm] = by_lang
    record(PUBLIC / "participant_errors.json", groups)


if __name__ == "__main__":
    diagnose()
