"""Separate exact answer extraction from outputs that already contain an acceptable answer."""
import numpy as np
from scripts.audit_traditional_passage_training import ARM_NAMES, EVIDENCE, archived
from scripts.compare_traditional_passage_training import OUT, TraditionalPassages
from scripts.prepare_pictured_entry_answers import load,write_json


def contains(output, reference):
    offset = output.find(reference)
    while offset>=0:
        if offset%128==0:
            return True
        offset=output.find(reference,offset+1)
    return False


def main():
    data=TraditionalPassages("development")
    report=load(OUT/"result.json")
    aliases=[{data.bank[data.passages[g][l:r]].tobytes() for l,r in spans}
             for g,spans in zip(data.groups,data.intervals,strict=True)]
    measures={}
    for arm in ARM_NAMES:
        images,stop=archived(OUT/(arm+"_drcd.npz"),report[arm]["scores"]["drcd"]["archive_sha256"])
        categories=[]
        for i,refs in enumerate(aliases):
            output=images[i]
            if stop[i] and output in refs:
                categories.append(2)
            elif stop[i] and any(contains(output,reference) for reference in refs):
                categories.append(1)
            else:
                categories.append(0)
        measures[arm]=np.asarray(categories)
    result={arm:dict(exact=int((a==2).sum()),contains_additional_ink=int((a==1).sum()),
                     lacks_complete_reference_or_stop=int((a==0).sum())) for arm,a in measures.items()}
    for arm in ("released","replay"):
        mask=(measures["mixed"]==2)&(measures[arm]!=2)
        result["gained_over_"+arm]=dict(total=int(mask.sum()),previously_contained=int((mask&(measures[arm]==1)).sum()),
                                      previously_lacked=int((mask&(measures[arm]==0)).sum()))
    write_json(EVIDENCE/"containment.json",result)
    print(result)


if __name__=="__main__":
    main()
