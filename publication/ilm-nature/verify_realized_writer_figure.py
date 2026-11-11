#!/usr/bin/env python3
"""Check Figure 22 native values, literal workbooks and preserved prior assets."""
from collections import Counter
import argparse
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import fitz
from openpyxl import load_workbook

from verify_motor_diagnostics_figure import relationships, cell_values, cache_values, NS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STAGE = Path(os.environ.get('ILM_REALIZED_PAPER_STAGE',
                            str(ROOT / 'artifacts/realized_writer_publication_v1')))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def locate_figure(path, expected_labels):
    doc = fitz.open(path)
    matches = []
    for index, page in enumerate(doc):
        text = ' '.join(page.get_text().split())
        if not all(label in text for label in ['Physical writing and complete-answer transfer',
            'Balanced writer: exact tiles', 'Both correct (16 pairs)']):
            continue
        counts = Counter(word[4] for word in page.get_text('words'))
        if not all(counts[label] >= number for label, number in expected_labels.items()):
            raise ValueError('Missing six-decimal physical data labels: ' + str(path))
        matches.append(index + 1)
    if len(matches) != 1:
        raise ValueError('Exactly one complete Figure22 page required: ' + str(path))
    return dict(pages=len(doc), figure_page=matches[0], sha256=sha(path),
                physical_F1_labels_verified=True, visual_QA_still_required=True)


def main(figure_only=False):
    bindings_path = STAGE / 'figure_bindings.json'
    if not bindings_path.exists():
        bindings_path = HERE / 'evidence/realized_writer_figure_bindings.json'
    bindings = load(bindings_path)
    data_path = STAGE / 'measured_data.json'
    if not data_path.exists():
        data_path = HERE / 'evidence/realized_writer_measured_data.json'
    if sha(data_path) != bindings['measured_data_sha256']:
        raise ValueError('Source values changed after native authoring')
    if not bindings['unmeasured_language_score_not_zero']:
        raise ValueError('Unmeasured outcomes cannot be plotted as zero')
    ppt = HERE / 'figures/Realized_writer_figures.pptx'
    charts = []
    with ZipFile(ppt) as archive:
        part = 'ppt/slides/slide1.xml'
        slide = ET.fromstring(archive.read(part)); rels = relationships(archive, part)
        nodes = slide.findall('.//c:chart', NS)
        if len(nodes) != 4 or len(bindings['charts']) != 4:
            raise ValueError('Four editable native charts required')
        if slide.findall('.//p:pic', NS):
            raise ValueError('Quantitative charts cannot be replaced by screenshots')
        for number, (node, expected) in enumerate(zip(nodes, bindings['charts'])):
            chart_part = rels[node.attrib['{' + NS['r'] + '}id']]
            xml = ET.fromstring(archive.read(chart_part))
            series = xml.findall('.//c:ser', NS)
            external = xml.find('c:externalData', NS)
            if external is None or len(series) != len(expected['series']):
                raise ValueError('Complete native workbook and series required')
            book_part = relationships(archive, chart_part)[external.attrib['{' + NS['r'] + '}id']]
            book = load_workbook(BytesIO(archive.read(book_part)), data_only=True)
            for actual, source in zip(series, expected['series']):
                name = actual.find('c:tx/c:v', NS).text
                categories = cache_values(actual.find('c:cat', NS), 'string')
                values = cache_values(actual.find('c:val', NS), 'number')
                if (name != source['name'] or categories != expected['categories']
                        or values != source['values']):
                    raise ValueError('Native cache differs from bound complete records')
                cat_formula = actual.find('c:cat/c:strRef/c:f', NS).text
                val_formula = actual.find('c:val/c:numRef/c:f', NS).text
                if cell_values(book, cat_formula) != categories or cell_values(book, val_formula) != values:
                    raise ValueError('Literal workbook differs from native chart cache')
            if number == 0:
                fmt = xml.find('.//c:dLbls/c:numFmt', NS)
                if fmt is None or fmt.attrib != {'formatCode': '0.000000', 'sourceLinked': '0'}:
                    raise ValueError('Physical F1 labels require six explicit decimals')
            charts.append(dict(part=chart_part, workbook=book_part,
                               cache_and_literal_cells_match=True, series=len(series)))
    labels = Counter(f'{value:.6f}' for series in bindings['charts'][0]['series'] for value in series['values'])
    paths = [HERE / 'figures/Figure_22_realized_writer.pdf']
    if not figure_only:
        paths += [STAGE / 'reading/manuscript.pdf', STAGE / 'review/manuscript_review.pdf',
                  STAGE / 'redlines/redlined_manuscript/redlined_manuscript.pdf']
    PDFs = {str(path.relative_to(ROOT)): locate_figure(path, labels) for path in paths}
    preserved = load(STAGE / 'preserved_assets.json')
    for name, expected in preserved['files_sha256'].items():
        if sha(HERE / name) != expected:
            raise ValueError('Earlier figure/style/supplement changed: ' + name)
    result = dict(native_charts=charts, PDFS=PDFs, PPT_sha256=sha(ppt),
        source_sha256=sha(Path(__file__)), figure_binding_sha256=sha(bindings_path),
        prior_assets_preserved=len(preserved['files_sha256']), previous_assets_unchanged=True,
        native_PowerPoint_application_opened=False, structural_verification_only=True,
        visual_QA_still_required=True, comparisons_not_learning_curve=True,
        matched_article_PDFs_verified=not figure_only)
    name = 'figure_only_audit.json' if figure_only else 'figure_audit.json'
    (STAGE / name).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(native_charts=4, PDFs=len(PDFs), prior_assets_preserved=len(preserved['files_sha256']))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figure-only', action='store_true',
                        help='Verify the native figure; leave matching article PDFs explicitly unverified')
    main(parser.parse_args().figure_only)
