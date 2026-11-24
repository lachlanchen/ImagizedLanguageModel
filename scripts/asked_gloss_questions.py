"""Counterbalanced requests with exactly the same local windows of pictured ink."""

from collections import Counter

from scripts.prepare_asked_gloss import QUESTIONS

CN = ("主要動詞", "直接賓語")
EN = ("main verb", "direct object")


def make(frame, roles, answer_first):
    # Each pair is ordered as (predicate question, object question).
    left, right = roles if answer_first else roles[::-1]
    values = (frame.format(left, right), frame.format(right, left))

    def triples(s):
        padded = (None, *s, None)
        return Counter(zip(padded, padded[1:], padded[2:]))

    assert triples(values[0]) == triples(values[1]) and values[0] != values[1]
    return values


TRAIN = QUESTIONS[:4] + [
    make("請翻譯句中{}的意思，略過句中{}的意思", CN, True),
    make("略過句中{}的意思，請翻譯句中{}的意思", CN, False),
    make("請寫出文中{}的英文，不用回答文中{}的英文", CN, True),
    make("不用回答文中{}的英文，請寫出文中{}的英文", CN, False),
    make("請解釋這個{}的含義，不必解釋這個{}的含義", CN, True),
    make("不必解釋這個{}的含義，請解釋這個{}的含義", CN, False),
    make("Translate the {} in this passage and skip the {} in this passage", EN, True),
    make("Skip the {} in this passage and translate the {} in this passage", EN, False),
    make("Give the {} in English and ignore the {} in English", EN, True),
    make("Ignore the {} in English and give the {} in English", EN, False),
    make("Explain the {} in this line but omit the {} in this line", EN, True),
    make("Omit the {} in this line but explain the {} in this line", EN, False),
]
HELD = [
    make("請翻譯文中{}的含義，不必解釋文中{}的含義", CN, True),
    make("不必解釋文中{}的含義，請翻譯文中{}的含義", CN, False),
    make("Translate the {} in this line and ignore the {} in this line", EN, True),
    make("Ignore the {} in this line and translate the {} in this line", EN, False),
]
assert not {q for qs in TRAIN for q in qs} & {q for qs in HELD for q in qs}
