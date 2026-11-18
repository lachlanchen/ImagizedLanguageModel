"""Export the confirmed context/writer tensors and test CPU entry points before release."""
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
from safetensors.torch import load_file, save_file

from scripts.confirm_joint_reference_writer import OUT as CONFIRM
from scripts.diagnose_visual_context_gradients import load, require, sha, write_new
from scripts.fit_local_teacher_writer import DATA, ROOT, load_split
from scripts.fit_physical_answer_preference import OUT as JOINT
from scripts.fit_source_plate_reader import MODELS
from scripts.report_expression_preference import DEST
from scripts.train_visual_extractive_reader import save_page


def main():
    checks = load(CONFIRM/'result.json')
    require(checks['qualified_for_cpu_check'], 'Confirmation did not qualify this writer')
    work = CONFIRM/'cpu'
    work.mkdir()
    base = MODELS/'reader_writer.safetensors'
    source = MODELS/'historical_reader.safetensors'
    final = MODELS/'joint_reference_writer.safetensors'
    require(not final.exists(), 'Preserve existing release')
    weights = JOINT/'reference/weights.pt'
    require(sha(weights) == load(weights.parent/'fit.json')['weights_sha256'], 'Reference weights changed')
    trained = torch.load(weights, weights_only=True, map_location='cpu')
    original = load_file(str(base))
    changed = {k: v for k, v in trained.items() if k != 'landmarks' and not k.startswith('motor.')}
    require(trained.keys() == original.keys() and all(torch.equal(trained[k], original[k])
            for k in trained.keys() - changed.keys()), 'Base cannot reconstruct the fitted model')
    checkpoint = work/final.name
    save_file(changed, str(checkpoint), metadata=dict(format='ilm-context-writer-v2', base_sha256=sha(base),
              qualified_source_sha256=sha(source), fit_sha256=sha(weights)))
    samples = load(DEST/'summary.json')['samples']
    validation = load_split(DATA/'validation.npz')
    px = np.load(DATA/'pixels.npy')
    emitted = load_split(JOINT/'evaluation/validation/reference/generated.npz')
    old = load(ROOT/'references/evidence/historical_plates_20261118/cpu_inference.json')
    reports = {}
    def run(key, args, expected, route):
        folder = work/key
        subprocess.run([sys.executable, '-m','scripts.infer_pictured_idiom','--weights',str(base),
                        '--writer',str(checkpoint),'--source-reader',str(source),'--output',str(folder),*args],
                       check=True, cwd=ROOT, env=os.environ | {'CUDA_VISIBLE_DEVICES':''})
        data = (folder/'answer.bin').read_bytes()
        receipt = load(folder/'receipt.json')
        require(data == expected and receipt == dict(cells=len(expected)//128,stopped=True,route=route),
                'CPU entry point differs: '+key)
        reports[key] = dict(**receipt, physical_sha256=sha(folder/'answer.bin'))
    for label,sample in samples.items():
        i = sample['i']
        expected = emitted['packed'][i,:emitted['length'][i]].tobytes()
        run(label+'_typed',['--before-text',sample['before'],'--after-text',sample['after']],expected,'writer')
        sides = []
        for side,values in [('before',validation['context'][i,:24]),('after',validation['context'][i,24:])]:
            image = work/(label+'_'+side+'.png')
            save_page(image,px[values])
            sides.extend(['--'+side+'-image',str(image)])
        run(label+'_pictured',sides,expected,'writer')
    for label in ('yan','ma'):
        evidence = MODELS/'examples'
        args = ['--question-text', old['yan']['prompt']] if label == 'yan' else ['--question-image',str(evidence/'ma_question.png')]
        args += ['--plate-image',str(evidence/(label+'_plate.png'))]
        rows = load(evidence/'source.json')[label]['source']
        bank = np.load(ROOT/'data/source_plates_v1/pixels.npy')
        from hashlib import sha256
        indices = {sha256(p.tobytes()).hexdigest():i for i,p in enumerate(bank)}
        expected = b''.join(bank[indices[next(r['raster'] for r in rows if r['label']==wanted)]].tobytes()
                            for wanted in old[label]['labels'])
        run(label,args,expected,'source')
    # Install only after every fresh subprocess has reproduced its physical stream.
    import shutil
    shutil.copyfile(checkpoint,final)
    require(sha(checkpoint)==sha(final),'Installed checkpoint changed')
    write_new(DEST/'release.json',dict(checkpoint=final.name,sha256=sha(final),bytes=final.stat().st_size,
              base_sha256=sha(base),historical_sha256=sha(source),cpu=reports))


if __name__ == '__main__':
    main()
