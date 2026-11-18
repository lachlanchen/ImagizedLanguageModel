"""Paired whole-record CPU timing and gradient checks on the trained native ILM."""
import json
import os
from pathlib import Path
import resource
import time

import torch

from ilm.visual_lm.cached_training_encoder import CachedTrainingImageLanguage
from ilm.visual_lm.continuous_autoregressive_language import ContinuousImageLanguage
from ilm.visual_lm.continuous_glyph_codec import ContinuousGlyphCodec, ContinuousGlyphCodecConfig
from ilm.visual_lm.visual_sequence_corpus import VisualSequenceCorpus
from scripts.diagnose_visual_context_gradients import load, require, sha, write_new
from scripts.train_continuous_image_language import ROOT, RUN, DATA, REG

OUT = ROOT/'artifacts/frozen_training_encoder_cpu_v1'


def host():
    line = next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
    return int(line.split()[1]) / 1024**2


def guard():
    require(host() >= 24 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2 < 4, 'CPU memory limit')


def timed(model, record):
    guard()
    start = time.perf_counter()
    model.zero_grad(set_to_none=True)
    terms = model.training_terms(record['ink'], record['prompt_glyphs'])
    (terms['code_nll'] + terms['stop_bce']).backward()
    elapsed = time.perf_counter() - start
    guard()
    return elapsed, [terms[k].detach().item() for k in ('code_nll','stop_bce')]


def main():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and host() >= 48, 'CPU-only host admission')
    fit = load(RUN/'report.json')
    audit = load(RUN/'audit.json')
    require(audit['complete'] and audit['report_sha256'] == sha(RUN/'report.json'), 'Audited fit required')
    snap = next(a for a in fit['arms'] if a['components'] == 8)['snapshots'][-1]
    checkpoint = ROOT/snap['file']
    require(snap['update'] == 13785 and sha(checkpoint) == snap['sha256'], 'Trained final model differs')
    state = torch.load(checkpoint, weights_only=True, map_location='cpu', mmap=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    corpus = VisualSequenceCorpus(DATA, split='train', expected_report_sha256=load(REG)['input_sha256']['data/visual_sequence_corpus_v1/report.json'])
    models = [cls(ContinuousGlyphCodec(ContinuousGlyphCodecConfig()), components=8).train()
              for cls in (ContinuousImageLanguage, CachedTrainingImageLanguage)]
    for model in models:
        model.load_state_dict(state['model'], strict=True)
    OUT.mkdir(exist_ok=False)
    cases = load(ROOT/'references/evidence/continuous_autoregressive_preflight_registration_20261118.json')['cases']
    rows = []
    for case in cases:
        record = corpus[case['corpus_index']]
        require(len(record['ink']) == case['glyphs'] and record['prompt_glyphs'] == case['prompt_glyphs'], 'Original record differs')
        models[1].clear_training_cache()
        cold = [timed(m, record)[0] for m in models]
        cold_encodings = models[1].encoded_training_cells
        runs = [[], []]
        for repeat in range(3):
            for i in ([0,1] if repeat % 2 == 0 else [1,0]):
                elapsed, losses = timed(models[i], record)
                runs[i].append(dict(seconds=elapsed,losses=losses))
        with torch.no_grad():
            code_error=(models[0].encode_ink(record['ink'])-models[1].encode_ink(record['ink'])).abs().max().item()
        gradients=[]
        for (n,p),(other,q) in zip(models[0].named_parameters(),models[1].named_parameters()):
            require(n == other and (p.grad is None) == (q.grad is None), 'Parameter participation differs')
            if p.requires_grad:
                require(p.grad is not None and q.grad is not None, 'Missing trainable gradient')
                a,b = p.grad.double(),q.grad.double()
                relative = (a-b).norm().item()/max(a.norm().item(),b.norm().item(),1e-8)
                gradients.append((relative,n))
        loss_error=max(abs(a-b) for u,v in zip(*runs) for a,b in zip(u['losses'],v['losses']))
        keep = code_error <= 2e-4 and loss_error <= 2e-4 and max(gradients)[0] <= 1e-3
        row=dict(index=case['corpus_index'],role=case['role'],images=case['glyphs'],cold_encodings=cold_encodings,
                 extra_warm_encodings=models[1].encoded_training_cells-cold_encodings,cold_seconds=cold,
                 runs=runs,code_max_absolute=code_error,loss_max_absolute=loss_error,
                 gradients_checked=len(gradients),worst_gradient=max(gradients),numerically_eligible=keep)
        rows.append(row)
        write_new(OUT/('case_'+str(case['corpus_index'])+'.json'),row)
        print(json.dumps({k:row[k] for k in ('role','images','cold_encodings','worst_gradient','numerically_eligible')}),flush=True)
    write_new(OUT/'report.json',dict(cases=rows,checkpoint_sha256=snap['sha256'],registration_sha256=sha(REG),
              script_sha256=sha(Path(__file__)),cache_source_sha256=sha(ROOT/'ilm/visual_lm/cached_training_encoder.py'),
              protocol_sha256=sha(ROOT/'references/frozen_training_encoder_cost_protocol.md'),torch=str(torch.__version__),
              peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,device='cpu',threads=2,
              optimizer_updates=0,generated_answers=0,numerically_eligible=all(r['numerically_eligible'] for r in rows)))


if __name__ == '__main__':
    main()
