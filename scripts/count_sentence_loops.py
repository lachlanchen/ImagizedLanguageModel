"""Count exact threefold physical suffix repetitions in autonomous Chinese translations."""
import numpy as np
from scripts.report_first_emission import EVIDENCE, load, require, roster, sha, write_new


def period(part):
    for width in range(1,min(12,len(part)//3)+1):
        if np.array_equal(part[-width:],part[-2*width:-width]) and np.array_equal(part[-width:],part[-3*width:-2*width]):
            return width
    return 0


def run():
    rows,bank=roster('reserved')
    stats=load(EVIDENCE/'sentence_records.json')
    result={}
    for arm in ('frozen','preference'):
        path=EVIDENCE/('sentences_'+arm+'.npz')
        require(sha(path)==stats[arm]['archive_sha256'],'Saved autonomous output changed')
        with np.load(path) as data:
            parts=np.split(data['pixels'],data['offsets'][1:-1])
        endings=[r['end'] for r in stats[arm]['rows']]
        p=np.array(list(map(period,parts)))
        ref=np.array([period(bank[r['y']]) for r in rows])
        result[arm]={}
        for end in sorted(set(endings)):
            included=np.array(endings)==end
            result[arm][end]=dict(sites=int(included.sum()),threefold_suffix=int((p[included]>0).sum()),
                                  reference_threefold_suffix=int((ref[included]>0).sum()))
    write_new(EVIDENCE/'cycles.json',result)
    print(result)


if __name__=='__main__':
    run()
