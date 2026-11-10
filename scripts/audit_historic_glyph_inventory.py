#!/usr/bin/env python3
"""Read-only availability/provenance audit of the existing historical corpus.

No crawling, neural model, rendering, relabelling or copying of source assets.
File/label existence is not a linguistic or archaeological validity judgment.
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
from urllib.parse import quote
import xml.etree.ElementTree as ET

from ilm.image_edit.baseline import ROOT,sha,save_json
from scripts.score_pixel_language_first_blocks import require,host_resources


def main(db,asset_root,output):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only asset audit')
    require(not output.exists(),'preserve previous inventories')
    require(host_resources()['available_gib']>=48,'asset audit host reserve')
    began=time.monotonic();asset_root=asset_root.resolve();db=db.resolve();before=db.stat()
    c=sqlite3.connect('file:'+quote(str(db),safe='/')+'?mode=ro',uri=True)
    c.execute('BEGIN')
    chars=c.execute('select count(*) from chars').fetchone()[0]
    rows=c.execute('select c.char,g.stage,g.label,g.local_path,g.source_site,g.width,g.height from glyphs g join chars c on c.id=g.char_id order by g.id').fetchall()
    c.commit();c.close()
    stages=Counter();valid=Counter();missing=Counter();formats=Counter();labels=defaultdict(set);unique_paths=set()
    sample=[];manifest=[]
    for index,row in enumerate(rows):
        char,stage,label,local,source,width,height=row;stages[stage]+=1
        if index%1024==0:
            require(host_resources()['available_gib']>=24 and time.monotonic()-began<300,'asset audit live reserve/deadline')
        p=(asset_root/local).resolve() if not Path(local).is_absolute() else Path(local).resolve()
        inside=p==asset_root or asset_root in p.parents
        exists=inside and p.is_file()
        if exists:valid[stage]+=1;formats[p.suffix.lower()]+=1;unique_paths.add(str(p))
        else:missing[stage]+=1
        labels[(stage,label)].add(char)
        record=dict(char=char,stage=stage,label=label,source_site=source,source_local_path=local,
            path_inside_asset_root=inside,file_exists=exists,stored_width=width,stored_height=height)
        manifest.append(record)
        if exists and (char in ['言','语','中','水'] or index%1024==0):
            require(p.stat().st_size<=2*1024*1024,'oversized sampled glyph; inspect separately')
            payload=p.read_bytes();item=dict(index=index,char=char,stage=stage,label=label,
                source_local_path=local,sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload))
            if p.suffix.lower()=='.svg':
                try:
                    root=ET.fromstring(payload)
                    external=[v for node in root.iter() for k,v in node.attrib.items() if k.rsplit('}',1)[-1]=='href' and v and not v.startswith(('#','data:'))]
                    item.update(xml_parseable=True,svg_root=root.tag.rsplit('}',1)[-1]=='svg',
                        viewBox=root.attrib.get('viewBox'),width=root.attrib.get('width'),height=root.attrib.get('height'),
                        paths=sum(n.tag.rsplit('}',1)[-1]=='path' for n in root.iter()),external_href_count=len(external),
                        active_elements=sum(n.tag.rsplit('}',1)[-1] in ['script','foreignObject'] for n in root.iter()))
                except ET.ParseError as error:item.update(xml_parseable=False,error=str(error))
            sample.append(item)
    after=db.stat();output.mkdir(parents=True)
    save_json(output/'records.json',manifest);save_json(output/'sampled_files.json',sample)
    conflicts=[dict(stage=s,label=l,characters=sorted(v)) for (s,l),v in labels.items() if len(v)>1]
    save_json(output/'cross_character_label_groups.json',conflicts)
    save_json(output/'report.json',dict(database=str(db),asset_root=str(asset_root),database_opened_read_only=True,
        database_consistent_read_transaction=True,database_bytes=before.st_size,
        database_base_file_stat_unchanged=(before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),
        characters=chars,glyph_records=len(rows),characters_with_glyph_records=len({r[0] for r in rows}),
        stages=dict(stages),existing_files_by_stage=dict(valid),missing_or_outside_root_by_stage=dict(missing),
        existing_file_formats=dict(formats),unique_existing_paths=len(unique_paths),
        cross_character_label_groups=len(conflicts),sampled_files=len(sample),
        sampled_parse_errors=sum(v.get('xml_parseable') is False for v in sample),
        sampled_external_or_active_SVGs=sum(bool(v.get('external_href_count') or v.get('active_elements')) for v in sample),
        records_sha256=sha(output/'records.json'),sampled_files_sha256=sha(output/'sampled_files.json'),
        cross_character_groups_sha256=sha(output/'cross_character_label_groups.json'),
        source_assets_copied=False,source_database_modified=False,network_requests=0,model_calls=0,
        individual_glyph_meanings_verified=False,unsampled_SVG_content_verified=False,
        labels_not_assumed_independent_split_groups=True,usable_language_or_historical_generation_demonstrated=False,
        elapsed_seconds=time.monotonic()-began))
    print(json.dumps(load_report(output),ensure_ascii=False),flush=True)


def load_report(output):return json.loads((output/'report.json').read_text())


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--db',type=Path,default=ROOT.parent/'incoder/data/historic/etymology.sqlite3')
    p.add_argument('--asset-root',type=Path,default=ROOT.parent/'incoder')
    p.add_argument('--output',type=Path,default=ROOT/'artifacts/historic_glyph_inventory_v1')
    a=p.parse_args();main(a.db,a.asset_root,a.output.resolve())
