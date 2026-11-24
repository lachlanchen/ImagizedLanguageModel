"""Ask a locally installed teacher for training-only bilingual visual question pairs."""

import json
import re
from scripts.local_translation_teacher import local_teacher
from scripts.train_span_choice import gpu_owner
from scripts.prepare_visual_clause_roles import WORK, PUBLIC, dump, read
from scripts.everyday_gloss_questions import TRAIN, BROAD, TEST
from scripts.visual_role_transfer_questions import HELD_PAIRS

MODEL="qwen3:8b-q8_0"
REGISTERS=("spoken to a young reader","an ordinary reader asking for a short explanation",
           "a concise exercise in literary interpretation","a reader contrasting the deed with its target",
           "a student asking a tutor without using grammar terminology",
           "a teacher asking indirect questions in everyday language")
SCHEMA={"type":"object","properties":{"pairs":{"type":"array","items":{"type":"object","properties":{
    "action":{"type":"string"},"object":{"type":"string"}},"required":["action","object"]}}},"required":["pairs"]}
SYSTEM="Write naturally worded questions for a bilingual reading exercise. Each pair asks two DIFFERENT questions about an unspecified line of classical Chinese with a transitive verb. The first asks for the ACTION that someone carries out (the verb). The second asks for the person or thing acted upon (the direct object). The answer should be one translated word from the line. Neither question asks for the subject, purpose, instrument or place. Do not give answers or placeholders, do not quote a particular story. Vary wording, length and sentence shape. Use the requested language. Return JSON matching the schema."

def normalized(q):
    return re.sub(r"[\W_]+","",q.casefold())

def main():
    folder=WORK/"teacher";folder.mkdir(exist_ok=True)
    reserved={normalized(s) for pair in TEST+HELD_PAIRS for s in pair}
    seen={normalized(s) for pair in TRAIN+BROAD for s in pair}
    kept=[];drops=[]
    with gpu_owner(),local_teacher(MODEL,folder) as (ask,digest,weight_bytes):
        for i,style in enumerate(REGISTERS):
            for j,lang in enumerate(("English","Traditional Chinese")):
                file=folder/f"raw-{i}-{j}.json"
                if file.exists():reply=read(file)
                else:
                    prompt=f"Give eight diverse pairs in {lang}. Register: {style}. Do not prefix or label questions."
                    reply=ask(SYSTEM,dict(role="user",content=prompt),think=False,format=SCHEMA,
                              options=dict(temperature=.85,seed=8379+i*3+j,num_predict=1350))
                    dump(file,reply)
                for pair in json.loads(reply["text"])["pairs"]:
                    vals=[pair[k].strip() for k in ("action","object")]
                    keys=[normalized(q) for q in vals]
                    if len(set(keys))<2 or min(map(len,keys))<3 or max(map(len,vals))>180 or any(k in reserved|seen for k in keys):
                        drops.append(vals)
                    else:
                        kept.append(dict(language=lang,register=style,pair=vals))
                        seen.update(keys)
    dump(PUBLIC/"teacher_questions.json",dict(model=MODEL,manifest=digest,installed_weight_bytes=weight_bytes,
         system=SYSTEM,registers=REGISTERS,kept=kept,dropped=drops))
    print(len(kept),"retained",len(drops),"excluded")

if __name__=="__main__":main()
