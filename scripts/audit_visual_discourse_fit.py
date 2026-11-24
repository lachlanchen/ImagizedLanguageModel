"""Recount the cells actually scored, and independently verify matched training schedules."""

from collections import Counter
import hashlib
from scripts.prepare_visual_discourse import PUBLIC, dump, read
from scripts.train_visual_discourse import Corpus, WARM,TUNE


def main():
    d=Corpus()
    results={}
    for name,fine,steps in (("warm",False,WARM),("continued",True,TUNE)):
        digest=bytes(32);c=Counter()
        for step in range(1,steps+1):
            ids,mask=d.batch(step,fine)
            digest=hashlib.sha256(digest+ids.tobytes()+mask.tobytes()).digest()
            c["source_and_target_cells"]+=int((ids>0).sum())
            c["supervised_cells"]+=int(mask[:32].sum())
            c["foil_cells"]+=int(mask[32:].sum())
            c["correct_sequences"]+=32
        arms=("ink-warm","byte-warm") if not fine else ("ink-pair","ink-independent","byte-pair")
        expected=digest.hex()
        for arm in arms:
            if read(PUBLIC/(arm+".json"))["schedule"]!=expected:raise ValueError("Mismatched draw schedule for "+arm)
        results[name]=dict(schedule=expected,arms=arms,**c)
    dump(PUBLIC/"matched-draws.json",results);print(results)


if __name__=="__main__":main()
