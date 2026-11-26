"""Reproduce saved book completions on CPU by reading ink without an input vocabulary."""
from hashlib import sha256
import json
from zipfile import ZipFile
import torch

from ilm.visual_lm.book_ink_inference import BookInkWriter
from scripts.evaluate_book_order_prior import PUB,BOOK,base,png,unpack


def main():
    torch.set_num_threads(2)
    checked={}
    assert base.sha(BOOK)==base.read(PUB/'prose.json')['archive']
    with ZipFile(BOOK) as archive:
        rows=json.loads(archive.read('records.json'))
        for arm in ('intact','scrambled'):
            for stage in ('prior','adapted'):
                engine=BookInkWriter(arm,stage)
                failed=[]
                for row in rows:
                    i=row['window']
                    prefix=unpack(archive.read(f'{i}/prefix.png'),row['prefix']['cells'])
                    out,stop=engine.write(prefix)
                    expected=row['answers'][arm+'_'+stage]
                    if sha256(png(out)).hexdigest()!=expected['sha256'] or stop!=expected['stop']:
                        failed.append(i)
                checked[arm+'_'+stage]=dict(cases=len(rows),different=failed)
    base.dump(PUB/'optical_replay.json',dict(archive=base.sha(BOOK),results=checked))
    print(checked)


if __name__=='__main__':
    main()
