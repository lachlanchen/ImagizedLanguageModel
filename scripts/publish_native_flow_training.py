#!/usr/bin/env python3
"""Publish audited training costs and every fixed monitoring image, not QA scores."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from ilm.image_edit.baseline import ROOT, save_json, sha


def publish(run, audit, output):
    source = json.loads((audit/'report.json').read_text())
    if source['implementation_sha256'] != sha(ROOT/'scripts/audit_native_flow_training.py'):
        raise ValueError('completed audit implementation differs')
    if (source['actual_optimizer_updates'] != 1536 or source['verified_fixed_monitoring_previews'] != 24
            or not source['all_monitoring_PNGs_equal_actual_decoder_tensors'] or source['heldout_assessment_completed']):
        raise ValueError('complete training-only audit required')
    for name, key in (('report.json','completed_report_sha256'),('preparation.json','training_preparation_sha256'),
            ('checkpoint.pt','final_checkpoint_sha256')):
        if sha(run/name) != source[key]: raise ValueError('owning completed training changed')
    for name, digest in source['code_sha256'].items():
        if sha(ROOT/name) != digest: raise ValueError('bound training implementation changed')
    observed, inspections = {}, []
    for path in sorted(run.glob('attempt_*/previews/inspection_*.json')):
        review = json.loads(path.read_text())
        if not review['original_pages_viewed_complete'] or not review['training_monitoring_only'] or review['heldout_assessment']:
            raise ValueError('raw monitoring review scope differs')
        for row in review['records']:
            image = path.parent/row['preview']/'answer.png'
            record = image.parent/'record.json'
            if sha(image) != row['PNG_sha256'] or sha(record) != row['record_sha256']:
                raise ValueError('actual reviewed preview binding changed')
            key = (row['after_update'],row['id'])
            if key in observed and observed[key] != row['PNG_sha256']:
                raise ValueError('conflicting raw inspection binding')
            observed[key] = row['PNG_sha256']
        inspections.append(review)
    expected = {(row['step'],row['id']):row['PNG_sha256'] for row in source['previews']}
    if observed != expected: raise ValueError('all24 actual fixed previews must be visually reviewed')
    output.mkdir(parents=True,exist_ok=False)
    files = {}

    def copy(path, name, digest):
        if sha(path) != digest: raise ValueError('published source bytes changed')
        target = output/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
        if sha(target) != digest: raise ValueError('published copy differs')
        files[name] = digest

    copy(audit/'report.json','training_audit.json',sha(audit/'report.json'))
    copy(run/'report.json','training_report.json',source['completed_report_sha256'])
    previews = []
    for row in source['previews']:
        name = 'images/'+Path(row['image']).parent.name+'.png'
        copy(run/row['image'],name,row['PNG_sha256'])
        previews.append({**row,'image':name})
    train = {r['id']:r for r in map(json.loads,(run/'training_pairs.jsonl').read_text().splitlines())}
    monitored = []
    for i, id in enumerate(dict.fromkeys(row['id'] for row in source['previews'])):
        original = train[id];images = {}
        for kind in ('input','target'):
            name = f'images/example{i}_{kind}.png'
            copy(run/original[kind],name,original[kind+'_sha256']);images[kind] = name
        monitored.append(dict(id=id,images=images,split='train'))
    attribution = ROOT/'references/evidence/native_paired_qa_result'
    for name in ('SOURCE_ATTRIBUTION.md','CMRC2018_LICENCE'):
        copy(attribution/name,name,sha(attribution/name))
    save_json(output/'raw_visual_review.json',dict(complete_monitoring_images=24,records=inspections,
        all_original_monitoring_pages_viewed=True,training_examples_only=True,manual_accuracy_assigned=False,
        independent_or_perfect_blinding_claim=False,heldout_assessment_completed=False))
    files['raw_visual_review.json'] = sha(output/'raw_visual_review.json')
    (output/'README.md').write_text('''# Completed native-flow training; language assessment pending

This package retains all 24 original fixed training-preview PNGs at the twelve
registered updates, plus both complete input/reference images. Every preview
is authenticated against its saved actual decoder tensor by the CPU audit.
All 24 original pages have been visually inspected. The final two previews
still give incorrect short answers. No image, transcript or contrast is repaired.

These are seen training examples with fixed monitoring seeds. They are not
held-out results or a checkpoint-selection rule. The complete initial/final
question, erasure and copying assessment is separate and still pending.

The final checkpoint contains 1,536 actual AdamW updates over 384 image pairs,
with 852 adapter tensors and 37,380,992 trainable parameters. The complete
pretrained runtime has 6,240,279,775 parameters. Its inherited language/image
pretraining remains part of the system. Training takes 11,450.839 seconds
(3.18 hours), at 15.697 GiB peak allocated GPU memory. Execution on one card
does not establish foundation training or efficiency at useful language quality.

`training_audit.json` preserves actual optimizer, schedule, RNG and pixel
authentication. `training_report.json` is the original completed report.
`raw_visual_review.json` binds all monitoring-image observations without a
blinded-review or quantitative manual-accuracy claim. `result.json` binds
every published file. The full checkpoint and actual tensors remain local
under the experiment directory; no weights or private runtime logs are copied.
The source data attribution and licence are retained alongside the images.

No new independent neural replay, correct answering, historical-script
understanding or independent ILM qualification is claimed by this package.
''')
    files['README.md'] = sha(output/'README.md')
    save_json(output/'result.json',dict(source_training_audit_sha256=sha(audit/'report.json'),
        publisher_sha256=sha(Path(__file__)),actual_optimizer_updates=1536,previews=previews,monitored_examples=monitored,
        files_sha256=files,all24_monitoring_PNG_bytes_unchanged=True,raw_monitoring_review_complete=True,
        heldout_assessment_completed=False,semantic_learning_validated=False,general_ilm_goal_complete=False))
    print('Published all24 original monitoring pages and actual training audit; no held-out language score.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run',type=Path,required=True)
    parser.add_argument('--audit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    publish(args.training_run.resolve(),args.audit.resolve(),args.output.resolve())
