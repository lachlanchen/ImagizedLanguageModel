"""Phrasings for training and sealed word-order interventions on pictured dictionary pages."""
from scripts.prepare_grounded_origin_pairs import STAGES

# In each pair, a means the value to be output first and b the value to be output last.
TRAIN = {
    True: (
        "For {h}, please show {a} first, followed by {b}.",
        "Read the entry for {h}. Begin with {a} and end with {b}.",
        "Give {b} after {a} for {h}.",
        "{h}: I want {a}, and next {b}.",
        "Put {b} last. First copy {a} from the entry for {h}.",
        "Which is {a} for {h}? Follow it with {b}.",
        "For {h}, the last item should be {b}. Start with {a}.",
    ),
    False: (
        "查「{h}」：先給我{a}，接著給我{b}。",
        "我要看「{h}」的{a}，然後再看{b}。",
        "「{h}」的{b}排在{a}之後。",
        "抄錄「{h}」的條目，從{a}開始，以{b}收尾。",
        "請把{b}留到最後，先列「{h}」的{a}。",
        "「{h}」的{a}是甚麼？下一項附上{b}。",
        "末項應為{b}。請先抄出「{h}」的{a}。",
    ),
}
HELD = {
    True: ("Under {h}, start with {a}. Put {b} second.",
           "For the entry {h}, put {b} after {a}."),
    False: ("就「{h}」這一條，先抄{a}，第二項列{b}。",
            "讀「{h}」的條目，把{b}放在{a}之後。"),
}


def question(row, template, swap=False):
    s=row["supervision"]
    stage=s["form"]
    positions=list(s["value_rows"])
    if swap:
        positions.reverse()
    form_first=len(row["page"][positions[0]]["value"])==1
    form,meaning=(("the "+stage+" form","the original meaning") if s["english"] else (STAGES[stage]+"字形","本義"))
    a,b=(form,meaning) if form_first else (meaning,form)
    return template.format(h=s["head"],a=a,b=b),positions
