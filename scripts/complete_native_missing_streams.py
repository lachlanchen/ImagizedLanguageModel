"""Finish only missing native cases after an authenticated generation deadline."""
import argparse
import os
from pathlib import Path
import resource
import signal
from tempfile import TemporaryDirectory

from scripts.diagnose_visual_context_gradients import load, require, sha, write_new
from scripts.evaluate_continuous_image_language import plan
from scripts.review_continuous_image_language import checked_page
from scripts.train_continuous_image_language import CODEC, EVAL, REG, ROOT, RUN, validate

ORIGINAL = RUN/'evaluation'
WORK = RUN/'missing_stream_completion_v1'
BIND = WORK/'inventory.json'
PIPELINE = ROOT/'artifacts/native_continuous_pipeline_20261118.json'
PROTOCOL = ROOT/'references/native_missing_stream_completion_protocol.md'


def alive(identity):
    stat = Path(f"/proc/{identity['pid']}/stat")
    if not stat.exists() or Path('/proc/sys/kernel/random/boot_id').read_text().strip()!=identity['boot_id']:
        return False
    try:
        raw = stat.read_text()
    except FileNotFoundError:
        return False
    fields = raw[raw.rfind(')')+2:].split()
    return fields[0]!='Z' and fields[19]==str(identity['start_ticks'])


def original_schedule():
    pipeline=load(PIPELINE)
    require(pipeline['current_stage']=='terminal_failure_evaluate', 'Original evaluator is not terminal with a failure')
    failed=load(ORIGINAL/'failure.json')
    require(failed['error'] in ("ValueError('glyph reader stage time cap reached')",
            "TimeoutError('Autonomous generation deadline')","RuntimeError('glyph reader stage time cap reached')"),
            'Only the original generation time limit permits completion')
    owners=[pipeline['supervisor'],pipeline['phases'][-1]['owner'],load(ORIGINAL/'status.json')]
    require(not any(alive(o) for o in owners),'An original generation owner is still alive')
    require([p['exit_code'] for p in pipeline['phases'][:2]]==[0,0] and pipeline['phases'][-1]['exit_code']!=0,
            'Original fit and audit must be successful')
    fit=load(RUN/'report.json')
    audit=load(RUN/'audit.json')
    require(audit['complete'] and audit['report_sha256']==sha(RUN/'report.json') and
            audit['registration_sha256']==fit['registration_sha256']==sha(REG),'Original fit or audit differs')
    snapshots={(a['components'],s['update']):s['sha256'] for a in fit['arms'] for s in a['snapshots']}
    reg=load(REG)
    schedule=[]
    for k in reg['components']:
        for job in plan(reg):
            step,cap=job['step'],job['cap']
            folder=ORIGINAL/f'k{k}_step{step:07d}_cap{cap}'
            for row in job['inputs']:
                schedule.append((k,step,cap,row,folder,snapshots[k,step]))
    require(len(schedule)==792,'Incomplete schedule')
    return schedule, failed['completed_streams']


def inventory():
    schedule, count=original_schedule()
    require(0<count<len(schedule) and not (ORIGINAL/'report.json').exists(),'No incomplete generation to finish')
    records={}
    missing=[]
    found=set()
    for index,(k,step,cap,row,folder,snapshot) in enumerate(schedule):
        file=folder/(row['id']+'.json')
        key=str(file.relative_to(ORIGINAL))
        if index>=count:
            require(not any((folder/(row['id']+ext)).exists() for ext in ('.json','.npz','.png')),
                    'Orphaned or unexpected case requires separate inspection')
            missing.append(key)
            continue
        rec=load(file)
        require((rec['components'],rec['step'],rec['cap'],rec['input'],rec['snapshot_sha256'])==
                (k,step,cap,row,snapshot),'Original receipt differs from schedule')
        checked_page(folder,rec)
        require(sha(EVAL/row['image'])==row['image_sha256'],'Original prompt differs')
        records[key]=sha(file)
        found.update(folder/(row['id']+ext) for ext in ('.json','.png','.npz'))
    actual={p for p in ORIGINAL.glob('k*/*') if p.is_file()}
    require(actual==found,'Unexpected original case artifact')
    files=(REG,RUN/'report.json',RUN/'audit.json',PIPELINE,ORIGINAL/'failure.json',ORIGINAL/'status.json',
           Path(__file__).resolve(),PROTOCOL)
    return dict(original={str(p.relative_to(ROOT)):sha(p) for p in files},
                retained=records, missing=missing, new_budget_seconds=10800)


def publish(record,folder,pixels):
    import numpy as np
    from PIL import Image
    case=record['input']['id']
    folder.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=WORK) as temp:
        tmp=Path(temp)
        raw=tmp/(case+'.npz')
        png=tmp/(case+'.png')
        np.savez_compressed(raw,ink=pixels)
        n=len(pixels)
        cols=min(32,max(1,n))
        rows=max(1,(n+cols-1)//cols)
        ink=np.zeros((rows*cols,32,32),np.uint8)
        ink[:n]=pixels[:,0]
        page=ink.reshape(rows,cols,32,32).transpose(0,2,1,3).reshape(rows*32,cols*32)
        Image.fromarray(255-255*page).save(png)
        for path,key in ((png,'output'),(raw,'raw')):
            record[key+'_file']=str((folder/path.name).relative_to(ROOT))
            record[key+'_sha256']=sha(path)
        receipt=tmp/(case+'.json')
        write_new(receipt,record)
        for path in (raw,png,receipt):
            os.link(path,folder/path.name)
    checked_page(folder,record)


def generate():
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0' and os.environ.get('CUDA_DEVICE_ORDER')=='PCI_BUS_ID'
            and os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','Exact original GPU configuration required')
    bound=load(BIND)
    require(inventory()==bound,'Original terminal inventory changed')
    _,head=validate()
    schedule,cut=original_schedule()
    import torch
    from ilm.visual_lm.codec_block_writer import load_qualified_writing_codec
    from ilm.visual_lm.continuous_autoregressive_language import ContinuousImageLanguage
    from ilm.visual_lm.glyph_reader_monitor import GlyphReaderMonitor
    from ilm.visual_lm.visual_context_deployment_resources import full_reader_gpu
    from scripts.infer_continuous_image_language import image_cells
    require(not (WORK/'owner.json').exists(),'A completion execution already exists')
    fields=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
    write_new(WORK/'owner.json',dict(pid=os.getpid(),start_ticks=fields[19],
              boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),inventory_sha256=sha(BIND)))
    monitor=GlyphReaderMonitor(WORK,bound['new_budget_seconds'])
    monitor.pressure_wait_allowance_seconds=300
    signal.signal(signal.SIGTERM,lambda *_: (_ for _ in ()).throw(TimeoutError('Completion deadline')))
    added=0
    try:
        with full_reader_gpu() as (device,admission):
            monitor.admission=admission
            torch.set_num_threads(2)
            torch.use_deterministic_algorithms(True,warn_only=False)
            torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.allow_tf32=False
            torch.backends.cudnn.benchmark=False
            torch.backends.cudnn.deterministic=True
            codec=load_qualified_writing_codec(CODEC,device)
            current=None
            for k,step,cap,row,folder,snapshot in schedule[cut:]:
                monitor.position=dict(components=k,step=step,cap=cap,case=row['id'])
                monitor.status('completing_missing_native_streams')
                monitor.guard()
                require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2<8,'Generation RSS ceiling')
                if current!=(k,step,cap):
                    if current is not None:
                        del model
                        torch.cuda.empty_cache()
                    checkpoint=RUN/f'k{k}'/f'model_{step:07d}.pt'
                    require(sha(checkpoint)==snapshot,'Checkpoint differs')
                    model=ContinuousImageLanguage(codec,components=k).to(device).eval()
                    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
                    model.load_state_dict(state['model'],strict=True)
                    del state
                    current=(k,step,cap)
                source=EVAL/row['image']
                require(sha(source)==row['image_sha256'],'Original prompt differs')
                ink=torch.from_numpy(image_cells(source,row['glyphs'])).to(device)
                emitted,info=model.generate(ink,maximum_new=cap)
                record=dict(components=k,step=step,cap=cap,input=row,snapshot_sha256=snapshot,**info)
                publish(record,folder,emitted.cpu().numpy())
                added+=1
            records=[]
            for k,step,cap,row,folder,snapshot in schedule:
                path=folder/(row['id']+'.json')
                key=str(path.relative_to(ORIGINAL))
                if key in bound['retained']:
                    require(sha(path)==bound['retained'][key],'Preserved original receipt changed')
                rec=load(path)
                require((rec['components'],rec['step'],rec['cap'],rec['input'],rec['snapshot_sha256'])==
                        (k,step,cap,row,snapshot),'Final schedule mismatch')
                checked_page(folder,rec)
                records.append(rec)
            require(added==len(bound['missing']) and len(records)==792,'Missing completion outputs')
            resources=monitor.receipt()
            write_new(ORIGINAL/'report.json',dict(registration_sha256=sha(REG),source_commit=head,
                    streams=records,resources=resources,original_failed_generation=load(ORIGINAL/'failure.json'),
                    original_streams=cut,additional_streams=added,completion_inventory_sha256=sha(BIND),
                    review_frozen=False,language_quality_measured=False,goal_achieved=False))
            monitor.status('complete_generation_awaiting_frozen_review')
    except BaseException as exc:
        write_new(WORK/'failure.json',dict(error=repr(exc),added=added,resources=monitor.receipt(),automatic_retry=False))
        raise


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('action',choices=('inventory','generate'))
    if cli.parse_args().action=='inventory':
        require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Audit on CPU')
        write_new(BIND,inventory())
    else:
        generate()
