#!/usr/bin/env python3
"""Derive a hashed path-SVG inventory; preserve legacy stage aliases separately.

Never modifies the legacy database, moves assets, or makes network/model calls.
Source category/structural validation is not semantic or chronological proof.
"""
import argparse
from collections import Counter,defaultdict
import math
import os
from pathlib import Path
import re
import time
import xml.etree.ElementTree as ET

from ilm.image_edit.baseline import ROOT,sha,save_json
from scripts.score_pixel_language_first_blocks import require,host_resources
from scripts.audit_historic_glyph_inventory import load_report

PREFIX={'J':'oracle','B':'bronze','S':'seal','L':'liushutong'}


def main(inventory,asset_root,output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only source curation')
    require(not output.exists(),'preserve prior derived inventories')
    require(host_resources()['available_gib']>=48,'curation host reserve')
    import hashlib,json
    source=load_report(inventory)
    require(sha(inventory/'records.json')==source['records_sha256'],'audited source rows changed')
    rows=json.loads((inventory/'records.json').read_text());asset_root=asset_root.resolve()
    output.mkdir(parents=True);began=time.monotonic();hashes={};structures={};accepted=[];quarantine=[]
    def content(row):
        path=Path(row['source_local_path']);path=(asset_root/path).resolve() if not path.is_absolute() else path.resolve()
        require(asset_root in path.parents and path.is_file(),'source file unavailable/outside declared root')
        if str(path) not in hashes:
            require(path.stat().st_size<=2*1024*1024,'oversized SVG requires separate inspection')
            payload=path.read_bytes();hashes[str(path)]=hashlib.sha256(payload).hexdigest()
            try:
                root=ET.fromstring(payload);tags=[n.tag.rsplit('}',1)[-1] for n in root.iter()]
                refs=[v for n in root.iter() for k,v in n.attrib.items() if k.rsplit('}',1)[-1]=='href' and v and not v.startswith(('#','data:'))]
                box=[float(v) for v in root.attrib.get('viewBox','').replace(',',' ').split()]
                valid=(root.tag.rsplit('}',1)[-1]=='svg' and tags.count('path')>0 and
                    not any(t in tags for t in ['text','script','foreignObject','image']) and not refs and
                    len(box)==4 and all(math.isfinite(v) for v in box) and box[2]>0 and box[3]>0)
                structures[str(path)]=dict(standalone_path_SVG=valid,viewBox=box,paths=tags.count('path'))
            except (ET.ParseError,ValueError):structures[str(path)]=dict(standalone_path_SVG=False)
        return hashes[str(path)],structures[str(path)]
    canonical=defaultdict(list)
    for i,row in enumerate(rows):
        label=row['label'] or ''
        if Path(row['source_local_path']).suffix.lower()=='.svg' and re.fullmatch('[JBSL][0-9]{3,6}',label) and row['stage']==PREFIX[label[0]]:
            canonical[row['char'],label].append(i)
    for i,row in enumerate(rows):
        if i%1024==0:
            require(host_resources()['available_gib']>=24 and time.monotonic()-began<600,'curation live reserve/deadline')
            save_json(output/'status.json',dict(phase='checking_source_vectors',processed=i,total=len(rows),pid=os.getpid()))
        label=row['label'] or '';base=dict(source_record_index=i,**row)
        if Path(row['source_local_path']).suffix.lower()!='.svg':
            quarantine.append(dict(base,reason='non_SVG_source_record'));continue
        if not re.fullmatch('[JBSL][0-9]{3,6}',label):
            quarantine.append(dict(base,reason='unrecognized_source_label'));continue
        digest,structure=content(row)
        if row['stage']!=PREFIX[label[0]]:
            matches=[j for j in canonical[row['char'],label] if content(rows[j])[0]==digest]
            quarantine.append(dict(base,reason='wrong_stage_duplicate' if matches else 'wrong_stage_without_byte_identical_canonical',
                source_expected_stage=PREFIX[label[0]],sha256=digest,matching_canonical_source_records=matches));continue
        if not structure['standalone_path_SVG']:
            quarantine.append(dict(base,reason='requires_separate_SVG_structure_review',sha256=digest));continue
        accepted.append(dict(base,sha256=digest,SVG_structure=structure,
            semantic_identity_independently_verified=False,source_stage_from_heading_and_label=True))
    by_label=defaultdict(set);by_image=defaultdict(set)
    for r in accepted:
        by_label[r['stage'],r['label']].add(r['char']);by_image[r['sha256']].add(r['char'])
    save_json(output/'records.json',accepted);save_json(output/'quarantine.json',quarantine)
    save_json(output/'report.json',dict(source_inventory_report_sha256=sha(inventory/'report.json'),
        source_records_sha256=source['records_sha256'],input_records=len(rows),accepted_records=len(accepted),
        accepted_characters=len({r['char'] for r in accepted}),accepted_stages=dict(Counter(r['stage'] for r in accepted)),
        accepted_characters_by_stage={s:len({r['char'] for r in accepted if r['stage']==s}) for s in PREFIX.values()},
        quarantined_records=len(quarantine),quarantine_reasons=dict(Counter(r['reason'] for r in quarantine)),
        accepted_unique_SVG_hashes=len(by_image),accepted_cross_character_label_groups=sum(len(v)>1 for v in by_label.values()),
        accepted_cross_character_identical_SVG_groups=sum(len(v)>1 for v in by_image.values()),
        all_accepted_records_content_hashed_and_XML_checked=True,records_sha256=sha(output/'records.json'),
        quarantine_sha256=sha(output/'quarantine.json'),source_assets_copied=False,legacy_database_modified=False,
        network_requests=0,model_calls=0,rasterization_performed=False,
        linguistic_meanings_or_dates_verified=False,trained_or_evaluated_model=False,
        elapsed_seconds=time.monotonic()-began))
    save_json(output/'status.json',dict(phase='completed',processed=len(rows),pid=None))
    print((output/'report.json').read_text(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory',type=Path,default=ROOT/'artifacts/historic_glyph_inventory_v1')
    p.add_argument('--asset-root',type=Path,default=ROOT.parent/'incoder')
    p.add_argument('--output',type=Path,default=ROOT/'artifacts/historic_stage_checked_v1')
    a=p.parse_args();main(a.inventory.resolve(),a.asset_root,a.output.resolve())
