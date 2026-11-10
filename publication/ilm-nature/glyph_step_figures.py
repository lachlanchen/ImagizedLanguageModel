"""Native editable S40 charts and unrepaired, explicitly cropped output streams."""
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

from generate_assignment_editable_figures import NS, Slide, chart_part, element, q, relpath, resolve, xml
from glyph_step_data import HERE, ROOT, collect, require, sha

FIGURE = 'Figure_S40_copying_and_feedback_steps'


def append(paper):
    require(paper.resolve() != HERE.resolve(), 'stage outside the active publication')
    data = collect()
    with zipfile.ZipFile(paper / 'ILM_Figures_editable.pptx') as archive:
        original = {name: archive.read(name) for name in archive.namelist()}
    parts = dict(original)
    presentation = ET.fromstring(parts['ppt/presentation.xml'])
    ids = presentation.find('p:sldIdLst', NS)
    require(len(ids) == 54, 'stage the reviewed S39 extension before S40')
    require(len([name for name in parts if re.fullmatch(r'ppt/charts/chart\d+\.xml', name)]) == 59,
            'S39 predecessor must contain 59 native charts')
    template = 'ppt/slides/slide54.xml'
    slide = Slide(parts[template])
    template_rels = ET.fromstring(parts[relpath(template)])
    types = ET.fromstring(parts['[Content_Types].xml'])
    relationships = ET.fromstring(parts['ppt/_rels/presentation.xml.rels'])
    chart_template = next(raw for name, raw in parts.items()
        if re.fullmatch(r'ppt/charts/chart\d+\.xml', name)
        and ET.fromstring(raw).find('c:chart/c:plotArea/c:barChart', NS) is not None)
    charts, media = [], []

    def chart(number, categories, values, series, maximum, unit, x):
        node, workbook = chart_part(chart_template, categories, values, series, 0, maximum, '527A98')
        for item in node.findall('.//c:numFmt', NS):
            item.set('formatCode', '0'); item.set('sourceLinked', '0')
        for item in node.findall('.//c:formatCode', NS): item.text = '0'
        for item in node.findall('.//a:defRPr', NS): item.set('sz', '1400')
        for axis in node.findall('c:chart/c:plotArea/c:valAx', NS):
            major = axis.find('c:majorUnit', NS)
            if major is None: major = element('c:majorUnit', axis)
            major.set('val', str(unit))
        part = f'ppt/charts/chart{number}.xml'
        book = f'ppt/embeddings/glyph_step_{number}.xlsx'
        require(part not in parts and book not in parts, 'new chart collision')
        node.find('c:externalData', NS).set(q('r:id'), 'rId1')
        rels = element('pkg:Relationships')
        element('pkg:Relationship', rels, Id='rId1', Type=NS['r']+'/package',
                Target='../embeddings/'+Path(book).name)
        parts[part], parts[book], parts[relpath(part)] = xml(node), workbook, xml(rels)
        element('ct:Override', types, PartName='/'+part,
                ContentType='application/vnd.openxmlformats-officedocument.drawingml.chart+xml')
        slide.chart(slide.relation('chart', '../charts/'+Path(part).name), x, 118, 720, 190)
        charts.append(dict(part=part, workbook=book, categories=categories, values=values,
                           series=series, axis_range=[0, maximum], major_unit=unit))

    def picture(row, x, y):
        path = ROOT / row['file']
        require(sha(path) == row['sha256'], 'raw stream changed')
        part = 'ppt/media/glyph_step_' + path.name
        require(part not in parts, 'new output-media collision')
        parts[part] = path.read_bytes()
        width = max(1, row['displayed_glyphs']) * 24
        slide.picture(slide.relation('image', '../media/'+Path(part).name),
                      path.stem+' actual output prefix', x, y, width, 24)
        if row['glyphs'] > 10:
            # The entire unmodified PNG remains inside the editable deck.
            # Only its display viewport shows the declared first-ten prefix.
            fill = list(slide.tree)[-1].find('p:blipFill', NS)
            viewport = element('a:srcRect', r=round(100000*(1-10/row['glyphs'])))
            fill.insert(1, viewport)
        media.append(dict(part=part, **row))

    slide.shape('Glyph copying, source-step mismatch and complete-answer failure', 45, 20, 1510, 50, 27, True)
    slide.shape('a  Held-wording copies reaching F1 0.95', 45, 80, 720, 32, 20, True)
    summaries = [data['copy_qualification'][i]['summary'][key] for i, key in
                 [(0, 'genuine/copy_held'), (1, 'candidate/copy_held')]]
    counts = [round(row['cases']*row['fraction_f1_at_least095']) for row in summaries]
    require(counts == [194, 229], 'copy qualification counts differ')
    chart(60, ['Anchor + pixels', 'Pixels only'], counts, 'Known-form copies / 256', 256, 64, 45)
    slide.shape('Same G and copy data. Gate requires at least 205/256.\nMean F1: 0.951778 vs 0.972856. This is not language accuracy.',
                45, 311, 720, 61, 14)
    slide.shape('b  Same-prefill source-policy diagnostic', 820, 80, 720, 32, 20, True)
    keys = ['native_units', 'exact_character_vector', 'canonical_image_encoder']
    counts = [data['feedback']['development/'+key]['literal_reference_exact'] for key in keys]
    require(counts == [12, 3, 3], 'feedback diagnosis counts differ')
    chart(61, ['Native unit', 'Exact char', 'Image E'], counts, 'Literal matches / 16', 16, 4, 820)
    slide.shape('Source head attached: diagnostic only, not ILM answers.\nEqual counts do not mean equal trajectories; failures retained.',
                820, 311, 720, 61, 14)
    slide.shape('Native unit: one or more characters', 45, 382, 450, 43, 18, fill='EAF1F6')
    slide.arrow(498, 403, 529, 403)
    slide.shape('Glyph feedback: one character per step', 535, 382, 505, 43, 18, fill='EAF1F6')
    slide.arrow(1043, 403, 1074, 403)
    slide.shape('Changed sequence advancement', 1080, 382, 475, 43, 18, fill='EAF1F6')
    slide.shape('c  Fresh C3 outputs: neither model gives a correct complete phrase (0/32 each)',
                45, 442, 1510, 32, 19, True)
    media_by_key = {(r['document'], r['order'], r['style'], r['version']): r for r in data['cells']}
    rows = [(order, style, version) for order in ['original', 'rotated']
            for style in ['letter', 'phrase'] for version in ['baseline', 'candidate']]
    for column, document in enumerate(data['fixed_documents']):
        x = 45 + column * 380
        slide.shape(f'Document {document:02} | fixed before export', x, 480, 365, 26, 15, True)
        for index, (order, style, version) in enumerate(rows):
            row = media_by_key[document, order, style, version]
            y = 512 + index * 44
            end = 'cap' if row['capped'] else 'stop'
            label = f"{version[0].upper()} {order[0]} {style[0]} | n={row['glyphs']} {end}"
            if row['display_is_prefix_excerpt']: label += ' | first 10'
            slide.shape(label, x, y, 365, 18, 11)
            picture(row, x+4, y+18)
    slide.shape('B: original; C: pixel-only writer. o/r: option order; l/p: letter/phrase. Prefix excerpts only where marked; full raw PNGs embedded.',
                45, 868, 1510, 28, 12)

    slide.relation('slideLayout', next(n.get('Target') for n in template_rels
                                      if n.get('Type').endswith('/slideLayout')))
    part = 'ppt/slides/slide55.xml'; note = 'ppt/notesSlides/glyph_step_notes55.xml'
    slide.relation('notesSlide', '../notesSlides/'+Path(note).name)
    old_note = next(resolve(template, n.get('Target')) for n in template_rels
                    if n.get('Type').endswith('/notesSlide'))
    notes = ET.fromstring(parts[old_note]); note_rels = ET.fromstring(parts[relpath(old_note)])
    caption = ('Figure S40. Completed copy-objective, source-feedback and fresh C3 controls. '
        'The two native charts show measured counts with their separate denominators. '
        'The source-head diagnostic is not the image-only ILM runtime and supplies no training examples. '
        'C3 documents0,5,10,15were fixed before export; both models, option orders and instructions are retained. '
        'Displayed prefix excerpts retain the original complete PNG bytes in the deck and explicitly label full length and cap/stop. '
        'No strokes or answer images are repaired. C3 phrase matches are zero for both models; letter matches are6/32and11/32, '
        'with both option orders correct for1/16and3/16documents. Agent reading is not independent human grading. '
        'No unfinished contextual-adaptation language result is included.')
    for shape in notes.findall('p:cSld/p:spTree/p:sp', NS):
        ph = shape.find('p:nvSpPr/p:nvPr/p:ph', NS)
        if ph is not None and ph.get('type') == 'body':
            for i, node in enumerate(shape.findall('.//a:t', NS)): node.text = caption if i == 0 else ''
    for node in note_rels:
        if node.get('Type').endswith('/slide'): node.set('Target', '../slides/slide55.xml')
    for name, kind in [(part, 'slide'), (note, 'notesSlide')]:
        require(name not in parts, 'new slide/note collision')
        element('ct:Override', types, PartName='/'+name,
                ContentType='application/vnd.openxmlformats-officedocument.presentationml.'+kind+'+xml')
    parts[part], parts[relpath(part)] = xml(slide.root), xml(slide.rels)
    parts[note], parts[relpath(note)] = xml(notes), xml(note_rels)
    used = {n.get('Id') for n in relationships}
    rid = next(f'rId{i}' for i in range(1, 1000) if f'rId{i}' not in used)
    element('pkg:Relationship', relationships, Id=rid, Type=NS['r']+'/slide', Target='slides/slide55.xml')
    element('p:sldId', ids, id=max(int(n.get('id')) for n in ids)+1).set(q('r:id'), rid)
    mutable = {'[Content_Types].xml', 'ppt/presentation.xml', 'ppt/_rels/presentation.xml.rels'}
    for name, node in [('[Content_Types].xml', types), ('ppt/presentation.xml', presentation),
                       ('ppt/_rels/presentation.xml.rels', relationships)]: parts[name] = xml(node)
    require(all(parts[name] == raw for name, raw in original.items() if name not in mutable),
            'predecessor slide, chart or media part changed')
    with zipfile.ZipFile(paper/'ILM_Figures_editable.pptx', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, raw in parts.items(): archive.writestr(name, raw)
    return dict(native_slides=55, native_charts=61, prior54_slide_parts_preserved=True,
        prior59_charts_preserved=True, charts=charts, media=media, sources=data,
        runtime_outputs_repaired=False, unfinished_tail_language_results_included=False)
