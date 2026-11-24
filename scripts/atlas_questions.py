"""Bilingual training and held question frames with pixel-position supervision."""
import numpy as np

FIELDS = ("oracle","bronze","seal","liushutong","original_meaning","decomposition")
NAMES = (
    (("甲骨文","甲骨字形"),("金文","金文字形"),("篆書","小篆"),("六書通","六書通字形"),
     ("本義","原義"),("字形分解","構形說明"),("字源","演變")),
    (("oracle bone","oracle"),("bronze","bronze form"),("seal","seal script"),("Liushutong","liushu"),
     ("original meaning","meaning"),("decomposition","structure"),("origin","evolution")),
)
TRAIN = (
    ("請列出{h}的{f}。","{f}：查{h}。","看過{d}後，請查{h}的{f}。","查{h}的{f}，另有{d}。","{d}和{h}之中，請列出後者的{f}。","{h}和{d}之中，請列出前者的{f}。"),
    ("Show the {f} of {h}.","For {h}, give the {f}.","After reading {d}, show the {f} of {h}.","Look up {h} for its {f}, then leave {d} aside.","Between {d} and {h}, give the latter's {f}.","Between {h} and {d}, give the former's {f}."),
)
HELD = (
    ("書中有{d}，我想查的是{h}的{f}。","要查{h}的{f}，旁邊還寫著{d}。"),
    ("I saw {d} in the book. What is the {f} of {h}?","What is the {f} of {h}? The other entry is {d}."),
)
LIMIT = 128


def render_question(frame,head,foil,fields,lang,variant):
    pieces,labels = [],[]
    head_position = None
    for word in frame.replace("{","\x01{").replace("}","}\x01").split("\x01"):
        if word == "{h}":
            head_position = len(labels)
            word = head
        elif word == "{d}":
            word = foil
        elif word == "{f}":
            for i,field in enumerate(fields):
                if i:
                    sep = ("、"," and ")[lang]
                    pieces.append(sep)
                    labels.extend([0]*len(sep))
                term = NAMES[lang][field][variant]
                pieces.append(term)
                labels.extend([1+field]+[0]*(len(term)-1))
            continue
        pieces.append(word)
        labels.extend([0]*len(word))
    text = "".join(pieces)
    if len(text)>LIMIT or head_position is None:
        raise ValueError(text)
    return text,head_position,np.asarray(labels,np.uint8)


def corpus_chars():
    strings = [v for group in (*TRAIN,*HELD) for v in group]
    strings += [s for group in NAMES for field in group for s in field]
    return set("".join(strings)+"、 and ")
