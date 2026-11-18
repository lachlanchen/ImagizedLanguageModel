"""Trace fixed complete outputs and decompose code-to-visible-feedback error."""
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
import torch

from ilm.visual_lm.continuous_autoregressive_language import ContinuousImageLanguage
from ilm.visual_lm.continuous_glyph_codec import ContinuousGlyphCodec, ContinuousGlyphCodecConfig
from scripts.diagnose_visual_context_gradients import load, require, sha, write_new
from scripts.infer_continuous_image_language import image_cells
from scripts.train_continuous_image_language import DATA, REG, ROOT, RUN

EVIDENCE = ROOT/'references/evidence/native_yan_cold_20261118'
OUT = ROOT/'artifacts/native_yan_code_cycle_v1'


def guard(start, entry=False):
    mem = next(s for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))
    require(int(mem.split()[1])/1024**2 >= (48 if entry else 24), 'Host reserve')
    require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2 < 4, 'Process RSS ceiling')
    require(time.monotonic()-start <= 900, 'CPU diagnostic deadline')


def geometry(a, b):
    a,b = a.double(),b.double()
    ma,mb = a.mean(-1,keepdim=True),b.mean(-1,keepdim=True)
    va,vb = a-ma,b-mb
    sa,sb = va.square().mean(-1).sqrt(),vb.square().mean(-1).sqrt()
    shift = (ma-mb).squeeze(-1).square()
    scale = (sa-sb).square()
    angular = 2*(sa*sb-(va*vb).mean(-1))
    mse = (a-b).square().mean(-1)
    torch.testing.assert_close(shift+scale+angular,mse,rtol=1e-10,atol=1e-12)
    require(bool((angular>=-1e-12).all()), 'Invalid centred angular discrepancy')
    return torch.stack((mse.sqrt(),shift,scale,angular,sa,sb),dim=1).numpy()


def collect_controls(start):
    reg = load(REG)
    require(sha(DATA/'report.json') == reg['input_sha256'][str((DATA/'report.json').relative_to(ROOT))], 'Corpus changed')
    tiles = set()
    for shard in load(DATA/'report.json')['shards']:
        if shard['split'] == 'train':
            guard(start)
            file, = [p for p in shard['files'] if p.endswith('_pixels.npz')]
            require(sha(DATA/file) == shard['files'][file] == reg['input_sha256'][str((DATA/file).relative_to(ROOT))], 'Pixel shard differs')
            with np.load(DATA/file, allow_pickle=False) as source:
                tiles.update(x.tobytes() for x in np.unique(source['ink'].view('V128')))
    require(len(tiles)==9581, 'Incomplete original bitmap control')
    return np.frombuffer(b''.join(sorted(tiles)),dtype=np.uint8).reshape(-1,128)


def quantiles(x):
    return dict(zip(('minimum','median','p90','p99','maximum'),np.quantile(x,[0,.5,.9,.99,1]).tolist()))


@torch.inference_mode()
def main():
    start=time.monotonic()
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU only')
    guard(start, entry=True)
    OUT.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    greedy=load(EVIDENCE/'result.json')
    sampled=load(EVIDENCE/'sampled_result.json')
    checkpoint=RUN/'k8/model_0013785.pt'
    require(sha(checkpoint)==greedy['checkpoint_sha256']==sampled['receipt']['checkpoint_sha256'], 'Final model differs')
    model=ContinuousImageLanguage(ContinuousGlyphCodec(ContinuousGlyphCodecConfig()),components=8).eval()
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=True)['model'],strict=True)
    source=EVIDENCE/'prompt.png'
    require(sha(source)==greedy['prompt_sha256'],'Question changed')
    prompt=torch.from_numpy(image_cells(source))
    arrays={}
    for mode,receipt,filename in [('greedy',greedy['routes']['pictured']['receipt'],'greedy_answer.png'),
                                  ('sampled',sampled['receipt'],'sampled_answer.png')]:
        guard(start)
        captured=[]
        def capture(_module, inputs):
            captured.append(inputs[0].detach().clone())
        hook=model.codec.decoder.register_forward_pre_hook(capture)
        try:
            ink, meta=model.generate(prompt,maximum_new=128,stochastic=receipt['stochastic'],
                                     generator=torch.Generator().manual_seed(receipt['seed']))
        finally:
            hook.remove()
        png=EVIDENCE/filename
        expected_sha=sampled['answer_png_sha256'] if mode=='sampled' else greedy['routes']['pictured']['output_png_sha256']
        require(sha(png)==expected_sha and all(meta[k]==receipt[k] for k in meta),'Original generation differs')
        require(len(captured)==len(ink)==128 and np.array_equal(ink.numpy(),image_cells(png)), 'Original pixels differ')
        proposed=torch.cat(captured)
        observed=model.encode_ink(ink)
        arrays[mode]=geometry(proposed,observed)
        np.savez_compressed(OUT/(mode+'_codes.npz'),proposed=proposed.numpy(),feedback=observed.numpy(),geometry=arrays[mode])
        print(mode,'original pixels reproduced',flush=True)
    packed=collect_controls(start)
    control=[]
    exact=0
    for first in range(0,len(packed),48):
        guard(start)
        ink=torch.from_numpy(np.unpackbits(packed[first:first+48],axis=1).reshape(-1,1,32,32))
        proposed=model.encode_ink(ink)
        drawn=(model.codec.decode(proposed)<0).to(torch.uint8)
        exact+=int((drawn==ink).reshape(len(ink),-1).all(-1).sum())
        control.append(geometry(proposed,model.encode_ink(drawn)))
    arrays['control']=np.concatenate(control)
    np.savez_compressed(OUT/'control.npz',packed=packed,geometry=arrays['control'])
    threshold=float(np.quantile(arrays['control'][:,0],.99))
    groups={}
    for label,values in arrays.items():
        squared=values[:,1:4].sum(0)
        total=float(squared.sum())
        groups[label]=dict(count=len(values),rms=quantiles(values[:,0]),above_control_p99=int((values[:,0]>threshold).sum()),
                           proposed_centered_sd=quantiles(values[:,4]),feedback_centered_sd=quantiles(values[:,5]),
                           fraction_shift_scale=float((squared[0]+squared[1])/total) if total>0 else None)
    mismatch=all(groups[k]['above_control_p99']/groups[k]['count']>=.9 for k in ('greedy','sampled'))
    scale_test=mismatch and all(groups[k]['fraction_shift_scale']>.5 for k in ('greedy','sampled'))
    guard(start)
    report=dict(groups=groups,train_control_exact_reconstructions=exact,case_mismatch=mismatch,
                rescaling_test_qualified=scale_test,original_pixels_reproduced=256,checkpoint_sha256=sha(checkpoint),
                code_sha256=sha(Path(__file__)),protocol_sha256=sha(ROOT/'references/native_yan_code_cycle_protocol.md'),
                files_sha256={p.name:sha(p) for p in OUT.glob('*.npz')},elapsed_seconds=time.monotonic()-start,
                peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2)
    write_new(OUT/'report.json',report)
    print(json.dumps(dict(groups=groups,mismatch=mismatch,rescaling_test_qualified=scale_test)))


if __name__=='__main__':
    main()
