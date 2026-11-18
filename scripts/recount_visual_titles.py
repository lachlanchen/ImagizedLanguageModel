"""Score the complete archive independently of decoding and certify the span control."""
from collections import Counter, defaultdict
import hashlib
import json
from zipfile import ZipFile

import numpy as np

from scripts.diagnose_visual_context_gradients import load, require, sha, write_new
from scripts.train_visual_title_transducer import DATA, OUT


def main():
    report=load(OUT/'report.json'); rows=load(OUT/'rows.json')
    roster=load(DATA/'dev_roster.json')
    streams={k:(DATA/f'dev_{k}.bin').read_bytes() for k in ('passages','titles')}
    question=(DATA/'question.bin').read_bytes()
    counters=defaultdict(Counter)
    with ZipFile(OUT/'outputs.zip') as archive:
        require(len(archive.infolist())==3*len(rows)==2502,'Incomplete outputs')
        for i,(meta,case) in enumerate(zip(roster,rows,strict=True)):
            def get(key, size):
                off=meta[key+'_byte_offset']
                return streams[key+'s'][off:off+128*size]
            passage=get('passage',meta['passage_cells'])
            truth=get('title',meta['title_cells'])
            require(hashlib.sha256(truth).hexdigest()==meta['title_sha256'],'Target mismatch')
            source=passage+question
            for mode in ('normal','contiguous','exchange'):
                raw=archive.read(f'{mode}/{i}.bin'); rec=case[mode]
                require(len(raw)==128*rec['cells'],'Length mismatch')
                exact=raw==truth and rec['stopped']
                require(exact==rec['exact'],'Equality mismatch')
                if mode=='contiguous' and raw:
                    starts=range(0,len(source)-len(raw)+1,128)
                    require(any(source.startswith(raw,j) for j in starts),'Noncontiguous control')
                for group in ['all',case['stratum']]+(['unknown'] if case['unknown'] else []):
                    counters[group][mode]+=int(exact)
    for group,values in counters.items():
        for mode,exact in values.items():
            require(exact==report['scores'][group][mode]['exact'],'Group score mismatch')
    write_new(OUT/'recount.json',dict(exact={k:dict(v) for k,v in counters.items()},
        whole_physical_streams=2502,contiguous_streams_verified=834,archive_sha256=sha(OUT/'outputs.zip'),
        stop_scope='Use the recorded autonomous stop decisions; independently compare every uncompressed image.'))


if __name__=='__main__':
    main()
