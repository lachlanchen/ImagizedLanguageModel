"""Blindly sample actual fitted bilingual targets and test exact proposition alignment."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import numpy as np
from scripts.evaluate_ordered_translation import chat, server, JUDGE, MODELS, STORE
from scripts.prepare_visual_rollins import BASE, BOOK, PREVIOUS, ROOT, WORK, PUB, read, dump, sha, training_rows
from scripts.train_span_choice import gpu_owner
from scripts.verse_gpu_runtime import configure_cuda_cli

QUESTIONS = {
    "same": "Both sides express the same complete event or statement, allowing idiom and ordinary paraphrase.",
    "english_more": "English adds a significant fact, clause or event absent from Chinese.",
    "chinese_more": "Chinese adds a significant fact, clause or event absent from English.",
    "different": "The pair contradicts, changes a significant detail, is untranslatable, or the sides describe different things.",
}
def sample():
    rows = training_rows()
    domains = np.load(BOOK / "domains.npy")
    included = np.ones(len(rows), bool)
    included[np.load(PREVIOUS / "replacement.npz")["indices"]] = False
    lookup = read(BASE / "readback.json")
    rng = np.random.default_rng(624320)
    examples = []
    for domain, name in enumerate(("book", "talk")):
        choices = rng.choice(np.flatnonzero(included & (domains == domain)), size=96, replace=False)
        items = []
        for i in choices:
            strings = ["".join(lookup[str(int(k))] for k in side if k).replace("\u3000", " ") for side in rows[i]]
            items.append((name, int(i), *strings))
        examples.extend(items)
        for i in range(12):
            examples.append(("foil", items[i][1], items[i][2], items[(i+1)%12][3]))
    examples.extend([("calibration", i, en, zh) for i, (en,zh) in enumerate([
        ("The train arrived yesterday.", "火车昨天到了。"), ("He closed the door.", "他把门关上了。"),
        ("She has two pens.", "她有两支笔。"), ("The room is empty.", "房间里没人，也没有东西。"),
        ("Can you hear me?", "你听得到我吗？"), ("The book is on the table.", "书在桌上。"),
        ("They were waiting outside.", "他们当时正在外面等。"), ("I have never been there.", "我从来没去过那里。"),
    ])])
    rng.shuffle(examples)
    return examples

def main():
    configure_cuda_cli()
    cases = sample()
    if sha(STORE / "blobs" / ("sha256-" + MODELS[JUDGE])) != MODELS[JUDGE]:
        raise ValueError("Unexpected judge weights")
    schema = dict(type="object", properties={"verdict":dict(type="string",enum=list(QUESTIONS))}, required=["verdict"])
    prompt = "Audit a bilingual training pair. Compare only the supplied statements. Return one verdict. " + json.dumps(QUESTIONS)
    jobs = []
    for _, _, en, zh in cases:
        for a in (dict(English=en, Chinese=zh), dict(Chinese=zh, English=en)):
            jobs.append(prompt+"\n"+json.dumps(a, ensure_ascii=False))
    with gpu_owner(), server() as url, ThreadPoolExecutor(max_workers=4) as pool:
        def ask(msg):
            r = chat(url, JUDGE, msg, schema)
            v = json.loads(r["text"])["verdict"]
            if v not in QUESTIONS or not r["stop"]:
                raise ValueError("Invalid judge result")
            return v
        votes = list(pool.map(ask,jobs))
    counts = defaultdict(Counter)
    records = []
    for j,(group,idx,en,zh) in enumerate(cases):
        a,b = votes[2*j:2*j+2]
        c = counts[group]
        c["pairs"]+=1
        c["same_twice"]+=a==b=="same"
        c["mismatch_twice"]+=a!="same" and b!="same"
        c["split_on_equivalence"]+=(a=="same") != (b=="same")
        records.append(dict(group=group, index=idx, en=en, zh=zh, votes=[a,b]))
    dump(WORK/"alignment-meaning.json", records)
    dump(PUB/"alignment-meaning.json",dict(seed=624320, judge=MODELS[JUDGE], counts=counts,
        program=sha(ROOT/"scripts/audit_visual_pair_meanings.py"), full=sha(WORK/"alignment-meaning.json"),
        records=[{k:v for k,v in r.items() if k not in ("en","zh")} for r in records]))
    print(dict(counts))

if __name__ == "__main__":
    main()
