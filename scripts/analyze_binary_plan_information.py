#!/usr/bin/env python3
"""Post-hoc information accounting of saved codec plans; no neural calls."""
from collections import Counter,defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time

from scripts import diagnose_visual_context_gradients as original

ROOT=original.ROOT
OUT=ROOT/'references/evidence/binary_plan_information_20261118.json'


def entropy(masses):
    total=sum(masses)
    return -sum((v/total)*math.log2(v/total) for v in masses if v>0)


def describe(values,groups,select,binary):
    chosen=[i for i,on in enumerate(select) if on];forms=Counter(int(groups[i]) for i in chosen)
    table=defaultdict(Counter);form_codes=defaultdict(set)
    for i in chosen:
        key=values[i].tobytes();g=int(groups[i]);table[key][g]+=1;form_codes[g].add(key)
    n=len(chosen);collisions=[c for c in table.values() if len(c)>1]
    exact_upper=sum(max(c.values()) for c in table.values())/n
    macro_upper=sum(max(count/forms[g] for g,count in c.items()) for c in table.values())/len(forms)
    h_form=entropy(list(forms.values()));h_code=entropy([sum(c.values()) for c in table.values()])
    conditional=sum(sum(c.values())/n*entropy(list(c.values())) for c in table.values())
    macro_code=[sum(count/forms[g] for g,count in c.items()) for c in table.values()]
    out=dict(positions=n,distinct_physical_forms=len(forms),distinct_actual_codes=len(table),
        codes_with_conflicting_physical_forms=len(collisions),positions_in_conflicting_codes=sum(sum(c.values()) for c in collisions),
        physical_forms_in_conflicting_codes=len({g for c in collisions for g in c}),
        maximum_forms_sharing_one_code=max(len(c) for c in table.values()),
        forms_with_multiple_actual_codes=sum(len(c)>1 for c in form_codes.values()),
        perfect_arbitrary_code_decoder_cell_exact_upper_bound=exact_upper,
        perfect_arbitrary_code_decoder_macro_exact_upper_bound=macro_upper,
        source_form_entropy_bits=h_form,actual_code_entropy_bits=h_code,
        source_form_given_code_entropy_bits=conditional,equal_form_weighted_code_entropy_bits=entropy(macro_code))
    if binary:
        import numpy as np
        bits=np.unpackbits(values[chosen],axis=1,bitorder='little').astype(float)
        probabilities=bits.mean(0);marginal=sum(entropy([float(p),float(1-p)]) for p in probabilities)
        out['binary_bit_statistics']=dict(nonconstant_bits=int(((probabilities>0)&(probabilities<1)).sum()),
            minimum_ink_probability=float(probabilities.min()),maximum_ink_probability=float(probabilities.max()),
            sum_marginal_bit_entropies=marginal,
            marginal_entropy_excess_over_actual_joint=marginal-h_code)
    return out


def main():
    original.require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU saved-array analysis only')
    original.require(not OUT.exists(),'Preserve every analysis')
    import fcntl
    import numpy as np
    import torch
    from scripts.score_pixel_language_first_blocks import host_resources
    torch.set_num_threads(2);began=time.perf_counter();samples=[]
    def guard():
        r=host_resources();samples.append(r)
        original.require(r['available_gib']>=24 and time.perf_counter()-began<180,'Information analysis reserve/deadline')
        original.require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<2*1024**3,'Information analysis RSS cap')
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        entry=host_resources();original.require(entry['available_gib']>=48 and entry['some_pressure_avg10']<=.5 and entry['full_pressure_avg10']<=.1,'Information analysis admission')
        public=ROOT/'references/evidence/binary_codec_comparison_result_20261118.json';raw=original.load(public)
        original.require(raw['all_275792_reconstructions_recounted'] and not raw['support'],'Use the complete audited failed comparison')
        partitionpath=ROOT/'artifacts/binary_codec_preparation_v1/partition.pt'
        reg=original.load(ROOT/'references/evidence/binary_codec_comparison_registration_20261118.json')
        original.require(original.sha(partitionpath)==reg['input_sha256'][str(partitionpath.relative_to(ROOT))],'Physical partition changed')
        partition=torch.load(partitionpath,weights_only=True,map_location='cpu');groups=partition['group'].numpy();held=partition['held'].numpy()
        inputs={str(p.relative_to(ROOT)):original.sha(p) for p in [public,partitionpath,Path(__file__)]};reports=[]
        for arm in raw['arms']:
            for stage in ['initial','final']:
                guard();path=ROOT/arm[stage]['raw_file'];original.require(original.sha(path)==arm[stage]['raw_sha256'],'Saved code bytes changed')
                inputs[str(path.relative_to(ROOT))]=original.sha(path)
                with np.load(path) as saved:values=saved['plan']
                report=dict(mode=arm['mode'],stage=stage,subsets={})
                for name,select in [('all',np.ones(len(groups),dtype=bool)),('train',~held),('held',held)]:
                    guard();report['subsets'][name]=describe(values,groups,select,arm['mode']=='binary')
                    observed=arm[stage]['metrics'][name]
                    cell=observed['pixels']['exact']/observed['pixels']['positions'];macro=observed['macro_form_exact_rate']
                    bounds=report['subsets'][name]
                    original.require(cell<=bounds['perfect_arbitrary_code_decoder_cell_exact_upper_bound']+1e-12
                        and macro<=bounds['perfect_arbitrary_code_decoder_macro_exact_upper_bound']+1e-12,'Observed decoder exceeds actual-code bound')
                reports.append(report)
        guard();result=dict(schema='ilm-binary-plan-information-v1',input_sha256=inputs,reports=reports,
            scope='Empirical exact-image equivalence in saved representation values; not Unicode labels, a source-alias re-audit, semantics or a universal capacity theorem',
            bounds_scope='Best arbitrary deterministic decoder of actual saved code alone on this finite support; no trained decoder or language success implied',
            continuous_code_scope='Exact saved FP32 bytes, not a 64-bit channel or an entropy-coding cost',
            all_original_source_labels_and_images_preserved=True,no_quality_rule_changed=True,neural_calls=0,optimizer_updates=0,gpu_used=False,
            resources=dict(elapsed_seconds=time.perf_counter()-began,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                minimum_available_host_gib=min(s['available_gib'] for s in samples),cpu_threads=2),full_language_fit_authorized=False,goal_achieved=False)
        original.write_new(OUT,result)
        print(json.dumps(dict(output=str(OUT.relative_to(ROOT)),sha256=original.sha(OUT),
            binary_final=next(v for v in reports if v['mode']=='binary' and v['stage']=='final')['subsets'])),flush=True)


if __name__=='__main__':main()
