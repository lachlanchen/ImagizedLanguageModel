#!/usr/bin/env python3
"""Verify Figure 23 native data and directed geometry; no Office/model execution."""
import argparse
from io import BytesIO
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

from openpyxl import load_workbook

from verify_motor_diagnostics_figure import relationships,cell_values,cache_values,NS

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):return json.loads(Path(path).read_text())
def require(condition,message):
    if not condition:raise ValueError(message)


def main(stage,ppt):
    data_path=HERE/'evidence/continuation_history_measured_data.json';data=load(data_path)
    binding_path=(stage/'figure_bindings.json' if stage else HERE/'evidence/continuation_history_figure_bindings.json')
    validation_path=(stage/'presentation_validation.json' if stage else HERE/'evidence/continuation_history_native_validation.json')
    binding=load(binding_path);validation=load(validation_path)
    require(binding['measured_data_sha256']==sha(data_path),'Bound complete records changed')
    require(binding['source_sha256']==data['source_sha256'],'Complete-record source bindings differ')
    for name,digest in data['source_sha256'].items():require(sha(ROOT/name)==digest,'Bound study changed: '+name)
    require(binding['builder_sha256']==sha(HERE/'continuation_history_figures.mjs') and
        binding['label_formatter_sha256']==sha(HERE/'format_continuation_history_chart_labels.py'),
        'Authoring source changed after native export')
    require(validation['finalSha256']==sha(ppt) and validation['firstPartyImport']['passed'],
        'Exact finalized native file and actual import required')
    require(validation['nativeChartValidation']['passed'] and validation['presentationLayout']['finding_count']==0,
        'Native structure/layout validation incomplete')
    require(binding['new_model_calls']==0 and not binding['semantic_alignment_claimed']
        and binding['preflight_not_language_evidence'] and binding['unmeasured_language_score_not_zero'],
        'Preserve scientific claim boundaries')
    layouts=['fixed_cells','proportional'];rounded=lambda v:round(v,6)
    expected=[dict(title='Mean F1 on shared indices',categories=['Fixed cells','Proportional'],series=[
        dict(name='Generated history',values=[rounded(data['history_groups'][k]['autonomous_shared']['mean_f1']) for k in layouts]),
        dict(name='Reference history',values=[rounded(data['history_groups'][k]['forced_shared']['mean_f1']) for k in layouts])]),
        dict(title='Later writing with reference history',categories=['Fixed cells','Proportional'],series=[
        dict(name='Median F1',values=[rounded(data['history_groups'][k]['later']['median_f1']) for k in layouts]),
        dict(name='Fraction F1 ≥0.95',values=[rounded(data['history_groups'][k]['later']['at_least_095']/713) for k in layouts])])]
    require(binding['charts']==expected,'Declared chart data differ from independent source calculation')
    names=['Shortest QA','Largest prompt','Longest QA','Longest book']
    expected_table=[['Case','Prompt','Target']]+[[name,f"{case['prompt_glyphs']:,}",f"{case['complete_target_glyphs']:,}"]
        for name,case in zip(names,data['preflight_cases'])]
    expected_edges={('prompt_pixels','frozen_prefill'),('frozen_prefill','immutable_KV'),
        ('immutable_KV','continuation'),('initial_query','continuation'),('continuation','writer_stop'),
        ('writer_stop','emitted_pixels'),('emitted_pixels','image_feedback'),('image_feedback','continuation')}
    charts=[]
    with ZipFile(ppt) as archive:
        part='ppt/slides/slide1.xml';slide=ET.fromstring(archive.read(part));rels=relationships(archive,part)
        require(not slide.findall('.//p:pic',NS),'Measured diagrams/charts may not be screenshots')
        chart_nodes=slide.findall('.//c:chart',NS);require(len(chart_nodes)==2,'Two native charts required')
        for node,source in zip(chart_nodes,expected):
            chart_part=rels[node.attrib['{'+NS['r']+'}id']];xml=ET.fromstring(archive.read(chart_part))
            require(''.join(xml.find('.//c:title',NS).itertext())==source['title'],'Native chart title differs')
            series=xml.findall('.//c:ser',NS);require(len(series)==2,'Complete native series required')
            external=xml.find('c:externalData',NS);require(external is not None,'Literal chart workbook required')
            workbook_part=relationships(archive,chart_part)[external.attrib['{'+NS['r']+'}id']]
            book=load_workbook(BytesIO(archive.read(workbook_part)),data_only=True)
            for actual,e in zip(series,source['series']):
                categories=cache_values(actual.find('c:cat',NS),'string');values=cache_values(actual.find('c:val',NS),'number')
                require(actual.find('c:tx/c:v',NS).text==e['name'] and categories==source['categories'] and values==e['values'],
                    'Native cache differs from complete source records')
                require(cell_values(book,actual.find('c:cat/c:strRef/c:f',NS).text)==categories and
                    cell_values(book,actual.find('c:val/c:numRef/c:f',NS).text)==values,'Workbook cells and caches differ')
            for fmt in xml.findall('.//c:dLbls/c:numFmt',NS):
                require(fmt.attrib=={'formatCode':'0.000000','sourceLinked':'0'},'Six-decimal labels required')
            require(xml.find('.//c:dLbls/c:numFmt',NS) is not None,'Explicit label format absent')
            charts.append(dict(part=chart_part,workbook=workbook_part,series=2,source_cache_literal_cells_match=True))
        tables=slide.findall('.//a:tbl',NS);require(len(tables)==1,'One native preflight table required')
        observed=[[''.join(t.itertext()) for t in row.findall('a:tc/a:txBody',NS)] for row in tables[0].findall('a:tr',NS)]
        require(observed==expected_table==binding['tableValues'],'Native preflight table differs from complete cases')
        identities={}
        for shape in slide.findall('.//p:sp',NS):
            node=shape.find('p:nvSpPr/p:cNvPr',NS)
            if node is not None and node.get('name'):identities[node.get('id')]=node.get('name')
        observed_edges=[]
        for connector in slide.findall('.//p:cxnSp',NS):
            start=connector.find('.//a:stCxn',NS);end=connector.find('.//a:endCxn',NS)
            tail=connector.find('.//a:tailEnd',NS);head=connector.find('.//a:headEnd',NS)
            require(start is not None and end is not None and tail is not None and tail.get('type')=='triangle'
                and (head is None or head.get('type')=='none'),'Arrow must point toward its encoded destination')
            observed_edges.append((identities[start.get('id')],identities[end.get('id')]))
        require(len(observed_edges)==8 and set(observed_edges)==expected_edges,'Complete directed pixel-feedback route required')
    preserved=load(HERE/'evidence/continuation_history_preserved_assets.json')
    for name,digest in preserved['files_sha256'].items():require(sha(HERE/name)==digest,'Earlier asset changed: '+name)
    if stage:
        for name,digest in data['baseline_PDF_sha256'].items():require(sha(HERE/name)==digest,'Active article PDF changed')
    result=dict(schema='continuation-history-native-figure-audit-v1',PPT_sha256=sha(ppt),
        verifier_sha256=sha(Path(__file__)),binding_sha256=sha(binding_path),
        measured_data_sha256=sha(data_path),native_charts=charts,native_table_exact=True,
        all_bound_complete_study_hashes_verified=True,
        native_table_values=observed,native_directed_connections=sorted(observed_edges),
        native_chart_points=8,six_decimal_native_labels=True,native_screenshot_objects=0,
        prior_assets_preserved=len(preserved['files_sha256']),active_article_PDFs_unchanged=True if stage else None,
        native_package_and_import_checks_pass=True,visual_QA_still_required=True,
        import_check_is_recorded_validation=True,new_ArtifactTool_import_performed=False,
        Office_export_verified=False,native_PowerPoint_application_opened=False,
        article_activation_and_full_build_pending=True,new_model_calls=0,goal_achieved=False)
    if stage:(stage/'native_figure_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(native_charts=2,exact_native_table=True,directed_connections=8,
        prior_assets_preserved=result['prior_assets_preserved'],Office_export_verified=False)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',type=Path)
    p.add_argument('--pptx',type=Path);a=p.parse_args()
    main(a.stage,a.pptx or (a.stage/'output/Continuation_history_figures.pptx' if a.stage
        else HERE/'figures/Continuation_history_figures.pptx'))
