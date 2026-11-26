"""Compare pretrained optical and native text translation on identical held requests."""
from collections import Counter,defaultdict
from hashlib import sha256
import json
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import MarianMTModel,MarianTokenizer
from ilm.visual_lm.pictured_path_writer import PicturedPathWriter, COUNTS
from scripts.audit_glyph_draft_editor import unpack
from scripts.evaluate_visual_training_breadth import ROOT,PUB,WORK,ARCHIVE,png,blind_votes,base,gpu_owner
from scripts.verse_gpu_runtime import configure_cuda_cli

RAW=ROOT/'publication/ilm-nature/evidence/breadth_pretrained_images.zip'


def grids(z,rows):
    face=base.Ink(np.load(base.BASE/'palette.npy'))
    return [np.unpackbits(unpack(z.read(f'{i:04d}/source.png'),len(face.encode(r['source']))),axis=-1).reshape(-1,32,32)
            for i,r in enumerate(rows)]


@torch.inference_mode()
def generate():
    existing=PUB/'transfer-generation.json'
    if existing.exists():
        assert base.read(existing)['archive']==base.sha(RAW)
        return
    assert base.sha(ARCHIVE)==base.read(PUB/'assessment.json')['archive']
    palette=np.load(base.BASE/'palette.npy')
    names=base.read(base.BASE/'readback.json')
    decode={v.tobytes():names[str(i)] for i,v in enumerate(palette) if str(i) in names}
    with ZipFile(ARCHIVE) as z:
        rows=json.loads(z.read('records.json'))[:96]
        images=grids(z,rows)
    outputs=[{} for _ in rows]
    partial=RAW.with_suffix('.partial')
    with ZipFile(partial,'w',ZIP_DEFLATED) as out:
        for t,lang in enumerate(('en','zh')):
            cfg=base.read(COUNTS/(lang+'-fit.json'))
            path=snapshot_download(cfg['model'],revision=cfg['revision'],local_files_only=True)
            tokenizer=MarianTokenizer.from_pretrained(path,local_files_only=True)
            native=MarianMTModel.from_pretrained(path,local_files_only=True,attn_implementation='eager').eval().cuda()
            visual=PicturedPathWriter(lang,'cuda')
            order=[i for i,r in enumerate(rows) if r['t']==t]
            for j in range(0,len(order),16):
                ids=order[j:j+16]
                rendered=visual.write([images[i] for i in ids])
                src=tokenizer([rows[i]['source'] for i in ids],padding=True,return_tensors='pt').to('cuda')
                predictions=native.generate(**src,num_beams=1,do_sample=False,max_new_tokens=128,forced_eos_token_id=None)
                for i,(pixels,stop,spans),seq in zip(ids,rendered,predictions.cpu(),strict=True):
                    bits=np.packbits(pixels.reshape(len(pixels),1024),axis=1)
                    blob=png(bits)
                    out.writestr(f'{i:04d}/pretrained.png',blob)
                    outputs[i]=dict(visual=dict(text=''.join(decode[x.tobytes()] for x in bits),stop=bool(stop),
                                         cells=len(bits),spans=spans,png=sha256(blob).hexdigest()),
                                    native=dict(text=tokenizer.decode(seq,skip_special_tokens=True),
                                                stop=native.config.eos_token_id in seq[1:].tolist()))
            del visual,native
        out.writestr('records.json',json.dumps(outputs,ensure_ascii=False))
    partial.replace(RAW)
    base.dump(existing,dict(primary=base.sha(ARCHIVE),archive=base.sha(RAW),
              protocol=base.sha(ROOT/'references/visual_training_breadth_transfer_control.md'),
              directional_fits={k:base.sha(COUNTS/(k+'-fit.json')) for k in ('en','zh')}))


def assess():
    with ZipFile(RAW) as z:
        outputs=json.loads(z.read('records.json'))
        for i,r in enumerate(outputs):
            assert sha256(z.read(f'{i:04d}/pretrained.png')).hexdigest()==r['visual']['png']
    with ZipFile(ARCHIVE) as z:
        rows=json.loads(z.read('records.json'))[:96]
    primary=base.read(PUB/'assessment.json')
    candidates=[dict(pair,broad=r['answers']['broad'],qwen=q,reference=dict(text=r['ref'],stop=True),
                     foil=dict(text=r['foil'],stop=True)) for pair,r,q in zip(outputs,rows,primary['comparator'],strict=True)]
    votes=blind_votes(rows,candidates)
    counts=defaultdict(Counter); grades=[]
    for i,(r,ans) in enumerate(zip(rows,candidates,strict=True)):
        g={m:bool(a['stop'] and votes[2*i][m] and votes[2*i+1][m]) for m,a in ans.items()}
        grades.append(g)
        for group in ('all',r['doc'],('en','zh')[r['t']]):
            counts[group].update(g)
    with ZipFile(ARCHIVE) as z:images=grids(z,rows)
    matches=[]
    for t,lang in enumerate(('en','zh')):
        model=PicturedPathWriter(lang,'cpu')
        ids=[i for i in (0,1,24,25,48,49,72,73) if rows[i]['t']==t]
        for i,(pixels,stop,bounds) in zip(ids,model.write([images[i] for i in ids]),strict=True):
            old=outputs[i]['visual']
            bits=np.packbits(pixels.reshape(len(pixels),1024),axis=1)
            matches.append(dict(i=i,same=stop==old['stop'] and bounds==old['spans'] and sha256(png(bits)).hexdigest()==old['png']))
    base.dump(PUB/'transfer.json',dict(archive=base.sha(RAW),primary=base.sha(PUB/'assessment.json'),counts=counts,
              grades=grades,votes=votes,cpu=matches,stopped={m:sum(r[m]['stop'] for r in outputs) for m in ('visual','native')}))
    print('transfer',dict(counts['all']),'cpu',sum(r['same'] for r in matches),len(matches),flush=True)


if __name__=='__main__':
    configure_cuda_cli()
    with gpu_owner():
        generate()
        if not (PUB/'transfer.json').exists():
            assess()
