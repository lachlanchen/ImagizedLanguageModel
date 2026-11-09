"""Append editable complete-learning evidence while retaining every old part."""
import hashlib
import io
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile

from PIL import Image

from generate_assignment_editable_figures import NS, Slide, chart_part, element, q, relpath, resolve, xml
from native_flow_manuscript import HERE, ROOT, require, sha

FIGURES = ('Figure_S34_complete_pixel_learning', 'Figure_S35_complete_pixel_examples')
EVIDENCE = ROOT / 'references/evidence/complete_pixel_language_result'


def append(paper, audit, monitors):
    with zipfile.ZipFile(HERE / 'ILM_Figures_editable.pptx') as archive:
        original = {n: archive.read(n) for n in archive.namelist()}
    parts = dict(original)
    presentation = ET.fromstring(parts['ppt/presentation.xml'])
    ids = presentation.find('p:sldIdLst', NS)
    require(len(ids) == 48, 'reviewed 48-slide predecessor required')
    types = ET.fromstring(parts['[Content_Types].xml'])
    relationships = ET.fromstring(parts['ppt/_rels/presentation.xml.rels'])
    template = 'ppt/slides/slide48.xml'
    template_rels = ET.fromstring(parts[relpath(template)])
    media = []

    def register(slide, number, caption):
        layout = next(n.get('Target') for n in template_rels if n.get('Type').endswith('/slideLayout'))
        slide.relation('slideLayout', layout)
        part = f'ppt/slides/slide{number}.xml'
        note = f'ppt/notesSlides/complete_pixel_notes{number}.xml'
        slide.relation('notesSlide', '../notesSlides/' + Path(note).name)
        old = next(resolve(template, n.get('Target')) for n in template_rels if n.get('Type').endswith('/notesSlide'))
        notes = ET.fromstring(parts[old])
        note_rels = ET.fromstring(parts[relpath(old)])
        for shape in notes.findall('p:cSld/p:spTree/p:sp', NS):
            placeholder = shape.find('p:nvSpPr/p:nvPr/p:ph', NS)
            if placeholder is not None and placeholder.get('type') == 'body':
                for i, node in enumerate(shape.findall('.//a:t', NS)):
                    node.text = caption if i == 0 else ''
        for node in note_rels:
            if node.get('Type').endswith('/slide'):
                node.set('Target', f'../slides/slide{number}.xml')
        for name, kind in ((part, 'slide'), (note, 'notesSlide')):
            require(name not in parts, 'slide collision')
            element('ct:Override', types, PartName='/' + name,
                    ContentType='application/vnd.openxmlformats-officedocument.presentationml.' + kind + '+xml')
        parts[part], parts[relpath(part)] = xml(slide.root), xml(slide.rels)
        parts[note], parts[relpath(note)] = xml(notes), xml(note_rels)
        used = {n.get('Id') for n in relationships}
        rid = next(f'rId{i}' for i in range(1, 1000) if f'rId{i}' not in used)
        element('pkg:Relationship', relationships, Id=rid, Type=NS['r'] + '/slide', Target=f'slides/slide{number}.xml')
        element('p:sldId', ids, id=max(int(n.get('id')) for n in ids) + 1).set(q('r:id'), rid)

    def picture(slide, path, name, x, y, scale, crop=None):
        raw = path.read_bytes()
        with Image.open(io.BytesIO(raw)) as source:
            image = source.convert('RGB')
        if crop:
            image = image.crop(crop)
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            raw = buffer.getvalue()
        part = 'ppt/media/complete_pixel_' + name + '.png'
        require(part not in parts, 'new media collision')
        parts[part] = raw
        slide.picture(slide.relation('image', '../media/' + Path(part).name), name,
                      x, y, image.width * scale, image.height * scale)
        media.append(dict(part=part, source=str(path.relative_to(ROOT)), source_sha256=sha(path),
                          crop_box=crop, RGB_size=list(image.size),
                          RGB_pixels_sha256=hashlib.sha256(image.tobytes()).hexdigest()))

    slide = Slide(parts[template])
    slide.shape('Complete image-answer learning fits seen questions and fails transfer', 45, 24, 1510, 60, 29, True)
    slide.shape('a  Native image context and a causal physical writer', 45, 110, 1510, 40, 22, True)
    for x, width, text in [(45, 275, 'Original prompt PNG\nFull visual context'),
                           (390, 350, 'Native contextual body\n4,096-dimensional state\nFrozen body, suffix LoRA'),
                           (815, 350, 'Physical pixel density\n256 causal eight-bit pieces'),
                           (1235, 320, '32 × 64 emitted image\nLearned stopping')]:
        slide.shape(text, x, 165, width, 95, 18, center=True,
                    fill='EAF1F6' if x in (390, 815) else 'F0F3F5')
    for a, b in [(325, 385), (745, 810), (1170, 1230)]:
        slide.arrow(a, 210, b, 210)
    slide.arrow(1395, 265, 1395, 315, head=False)
    slide.arrow(1395, 315, 570, 315, head=False)
    slide.arrow(570, 315, 570, 265)
    slide.shape('Read actual emitted pixels through the frozen history encoder and learned input projection',
                600, 272, 790, 38, 16, center=True)
    slide.shape('Training supplies true preceding images. Inference receives its own pixels and no desired answer length.',
                45, 340, 1510, 40, 17)
    slide.shape('b  Whole-training image NLL per physical piece', 45, 405, 680, 40, 21, True)
    template_chart = next(raw for name, raw in parts.items()
                          if re.fullmatch(r'ppt/charts/chart\d+\.xml', name)
                          and ET.fromstring(raw).find('c:chart/c:plotArea/c:barChart', NS) is not None)
    chart, workbook = chart_part(template_chart, [str(r['update']) for r in monitors],
                                [r['image_nll'] for r in monitors], 'Training image NLL', 1e-9, 10, '527A98')
    bar = chart.find('c:chart/c:plotArea/c:barChart', NS)
    labels = bar.find('c:dLbls', NS)
    if labels is not None:
        bar.remove(labels)
    for axis in chart.findall('c:chart/c:plotArea/c:valAx', NS):
        element('c:logBase', axis.find('c:scaling', NS), val='10')
        unit = axis.find('c:majorUnit', NS)
        if unit is not None:
            axis.remove(unit)
        fmt = axis.find('c:numFmt', NS)
        if fmt is not None:
            fmt.set('formatCode', '0E+00')
            fmt.set('sourceLinked', '0')
    for node in chart.findall('.//a:defRPr', NS):
        node.set('sz', '1600')
    chartname = 'ppt/charts/chart54.xml'
    bookname = 'ppt/embeddings/complete_pixel_NLL.xlsx'
    require(chartname not in parts and bookname not in parts, 'native chart collision')
    external = chart.find('c:externalData', NS)
    require(external is not None, 'editable chart workbook link required')
    external.set(q('r:id'), 'rId1')
    parts[chartname], parts[bookname] = xml(chart), workbook
    rels = element('pkg:Relationships')
    element('pkg:Relationship', rels, Id='rId1', Type=NS['r'] + '/package', Target='../embeddings/' + Path(bookname).name)
    parts[relpath(chartname)] = xml(rels)
    element('ct:Override', types, PartName='/' + chartname,
            ContentType='application/vnd.openxmlformats-officedocument.drawingml.chart+xml')
    slide.chart(slide.relation('chart', '../charts/chart54.xml'), 45, 455, 680, 320)
    slide.shape('Optimizer updates     Log scale     Correct preceding images', 75, 790, 660, 32, 16)
    slide.shape('c  Complete autonomous answers', 800, 405, 755, 40, 21, True)
    groups = audit['groups']
    rows = [['Assessment', 'Exact', 'F1'], ['Initial training', '0/32', '0.130'],
            ['Training, update 1,000', '17/32', '0.944'], ['Training, update 2,000', '29/32', '0.996'],
            ['Final training, fresh PNG', '32/32', '1.000'],
            ['Chinese development', '0/16', f'{groups["development_cmrc_normal"]["foreground_f1"]:.3f}'],
            ['English transfer', '0/8', f'{groups["development_dolly_normal"]["foreground_f1"]:.3f}']]
    for i, row in enumerate(rows):
        x = 800
        for j, (text, width) in enumerate(zip(row, [455, 145, 150])):
            slide.shape(text, x, 460 + i * 39, width, 38, 18, bold=i == 0, center=j > 0)
            x += width
    slide.shape('All outputs stop below the cap. Erased controls also fail.\nPixel scores do not assign semantic correctness.',
                800, 755, 755, 65, 17)
    slide.shape('7.544B total / 13.856M trained. Peak 18.191 GiB, 63.2 min active optimization. External pretraining excluded. Generalization remains unresolved.',
                45, 845, 1510, 52, 15)
    register(slide, 49, 'Figure S34. Complete native image-answer learning. Full native prompt keys remain available. '
             'Suffix adapters, image-history projection, start/stop interfaces and physical density learn from complete answer images. '
             'The density uses physical eight-bit patterns, not character identities. The loss averages piece NLL within blocks and questions plus balanced stopping. '
             'The editable log-scale chart shows matched whole-cohort correct-history NLL at0/1000/2000/4000, not autonomous accuracy. '
             'Fresh-PNG final training is32/32 exact; reused development is0/16 Chinese and0/8 English transfer. '
             'All48 normal/erased development outputs are retained. Sources: references/evidence/complete_pixel_language_result/independent_audit.json and complete_pixel_language_update4000/review_and_manifest.json.')

    slide = Slide(parts[template])
    slide.shape('Source-order examples separate learned writing from unseen questions', 45, 24, 1510, 60, 29, True)
    slide.shape('Input excerpts show the final 64 image rows. The actor reads each complete original page.', 45, 88, 1510, 35, 18)
    for i, index in enumerate((0, 1, 32, 33)):
        name = f'{index:02}'
        y = 140 + 174 * i
        kind = 'Training' if index < 32 else 'Development'
        slide.shape(f'{kind} question {1 + index % 2}', 45, y, 900, 32, 20, True)
        path = EVIDENCE / name / 'input_normal.png'
        with Image.open(path) as source:
            crop = (0, max(0, source.height - 64), source.width, source.height)
        picture(slide, path, f'prompt_{name}', 45, y + 43, .88, crop=crop)
        for offset, label, file in ((8, 'Output', 'answer_normal.png'), (75, 'Reference', 'reference_00.png')):
            slide.shape(label, 975, y + offset, 125, 32, 17, True)
            picture(slide, EVIDENCE / name / file, f'{file[:-4]}_{name}', 1110, y + offset, 1.4)
    slide.shape('First training and development passage pairs, without quality ranking. All complete outputs and erased controls remain in the evidence archive. No generated pixel is repaired.',
                45, 845, 1510, 52, 15)
    register(slide, 50, 'Figure S35. The first two training and first two development questions in the fixed source order. '
             'Input images display only the last64 physical rows, with exact crop boxes and original image hashes in the preparation receipt. '
             'The actor receives the full original page. Complete outputs and references retain their original pixels at a common1.4 display scale. '
             'The two seen answers are exact; both unseen answers have broken glyphs. No output quality selects these cases. '
             'All full inputs, references, outputs and erasure controls: references/evidence/complete_pixel_language_result/.')
    for name, node in [('[Content_Types].xml', types), ('ppt/presentation.xml', presentation),
                       ('ppt/_rels/presentation.xml.rels', relationships)]:
        parts[name] = xml(node)
    mutable = {'[Content_Types].xml', 'ppt/presentation.xml', 'ppt/_rels/presentation.xml.rels'}
    require(all(parts[n] == raw for n, raw in original.items() if n not in mutable), 'prior native package part changed')
    require(sum(bool(re.fullmatch(r'ppt/charts/chart\d+\.xml', n)) for n in parts) == 54, '54 native charts required')
    with zipfile.ZipFile(paper / 'ILM_Figures_editable.pptx', 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, raw in parts.items():
            archive.writestr(name, raw)
    return dict(media=media, prior48_slide_parts_preserved=True, prior53_chart_parts_preserved=True,
                native_slides=50, native_charts=54, artifact_tool_available=False,
                method='existing native ZIP/XML append with preserved prior parts and a linked chart workbook')
