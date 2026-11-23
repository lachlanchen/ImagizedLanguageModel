import pytest

from scripts.evaluate_page_evidence_writer import score
from scripts.page_evidence_corpus import dual_cases, single_cases, unique_year
from scripts.train_page_evidence_writer import source


def entry(key, name, person, year):
    return dict(key=key, p=f"《{name}》由{person}作词，{year}年出版。", q="谁写的词？",
                s=name, o=person, r="作词", group="duie")


def face(text):
    return b"".join(ord(c).to_bytes(4,"little")+bytes(124) for c in text)


@pytest.fixture
def pair(monkeypatch):
    class EmptyReservation:
        def overlaps(self, text):
            return False
    monkeypatch.setattr("scripts.page_evidence_corpus.reservation", EmptyReservation)
    return entry("a","湖雨","陈东","1995"), entry("b","山歌","姜明","2003")


def test_date_pair_depends_on_both_observations(pair):
    cases = dual_cases(*pair, held=True)
    assert len(cases) == 4
    for original, changed in zip(cases[:2], cases[2:]):
        assert source(original,"fields") == source(changed,"fields")
        assert original["q"] == changed["q"] and original["p"] != changed["p"]
        assert original["y"] != changed["y"] and original["year"] == changed["rival"]
    assert cases[0]["p"] == cases[1]["p"] and cases[0]["q"] != cases[1]["q"]
    outputs = [(face(x["y"]),True) for x in cases]
    assert score(cases,outputs,face)["dual"]["complete"] == 1
    outputs[-1] = (outputs[-1][0],False)
    assert score(cases,outputs,face)["dual"]["complete"] == 0
    with pytest.raises(ValueError,match="Incomplete"):
        score(cases[:3],outputs[:3],face)


def test_scope_of_year_edits(pair):
    a,b = pair
    assert not dual_cases(a,a,True)
    assert not unique_year(dict(b,p=b["p"]+"另见1990年。"))
    assert not unique_year(dict(a,s=a["s"]+"1995"))
    old,new=single_cases(a,True)
    assert old["p"].replace("1995年","1996年")==new["p"]
    assert old["q"]==new["q"]
    assert old["year"] in old["y"] and new["year"] in new["y"]
