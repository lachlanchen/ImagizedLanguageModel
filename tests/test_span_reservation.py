from scripts.span_reservation import Reservation, normalized


def index():
    idx = object.__new__(Reservation)
    idx.by_question = {"reserved": "p"}
    idx.by_paragraph = {"p": "a"}
    idx.named_sources = {
        "train": ("DRCD_training.json", False),
        "shared": ("DRCD_dev.json", True),
    }
    idx.titles = {normalized("預留")}
    idx.contexts = set()
    idx.windows = {"甲" * 64}
    return idx


def test_shared_passage_segment_vetoes_an_unrelated_source_id():
    idx = index()
    assert idx.rejects(
        {"group": "paired", "key": "other", "p": "句" + "甲" * 64 + "。"}
    )


def test_id_scope_and_title_screen():
    idx = index()
    assert idx.rejects({"group": "drcd", "key": "reserved", "p": "乙"})
    assert idx.rejects({"group": "drcd", "key": "shared", "p": "乙"})
    assert not idx.rejects({"group": "drcd", "key": "train", "p": "乙"})
    assert not idx.rejects({"group": "exchange", "key": "reserved", "p": "乙"})


def test_development_excludes_reserved_articles_and_question_ids():
    idx = index()
    p = {"id": "s", "context": "乙", "qas": [{"id": "train"}]}
    bad = dict(p, qas=[{"id": "reserved"}])
    assert list(
        idx.filter_development(
            [
                {"title": "预留", "paragraphs": [p]},
                {"title": "丙", "paragraphs": [bad, p]},
            ]
        )
    ) == [p]
