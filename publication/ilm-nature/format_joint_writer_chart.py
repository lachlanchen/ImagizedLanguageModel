#!/usr/bin/env python3
"""Make the fixed prerequisite a native combo-chart line; preserve literal data."""
from pathlib import Path
import sys
from zipfile import ZipFile

from lxml import etree as E

C = 'http://schemas.openxmlformats.org/drawingml/2006/chart'
A = 'http://schemas.openxmlformats.org/drawingml/2006/main'
def c(name): return '{' + C + '}' + name
def a(name): return '{' + A + '}' + name

path = Path(sys.argv[1]); found = set()
with ZipFile(path) as z:
    entries = [(info, z.read(info.filename)) for info in z.infolist()]
for i, (info, raw) in enumerate(entries):
    if '/charts/' not in info.filename or not info.filename.endswith('.xml'):
        continue
    root = E.fromstring(raw); title = ''.join(root.find('.//' + c('title')).itertext())
    assert title not in found
    if title == 'Frozen review credits with supplied image history':
        bar = root.find('.//' + c('barChart')); series = bar.findall(c('ser')); assert len(series) == 3
        threshold = series[-1]; assert threshold.find(c('tx') + '/' + c('v')).text == 'Required in every group'
        bar.remove(threshold)
        line = E.Element(c('lineChart')); E.SubElement(line, c('grouping'), val='standard')
        line.append(threshold)
        sp = threshold.find(c('spPr'))
        if sp is None:
            sp = E.Element(c('spPr')); threshold.insert(3, sp)
        for child in list(sp): sp.remove(child)
        ln = E.SubElement(sp, a('ln'), w='19050')
        solid = E.SubElement(ln, a('solidFill')); E.SubElement(solid, a('srgbClr'), val='A74337')
        E.SubElement(ln, a('prstDash'), val='dash')
        for child in list(threshold):
            if E.QName(child).localname in ['invertIfNegative', 'pictureOptions', 'shape']:
                threshold.remove(child)
        marker = E.Element(c('marker')); E.SubElement(marker, c('symbol'), val='none')
        threshold.insert(threshold.index(sp) + 1, marker)
        for labels in threshold.findall(c('dLbls')): threshold.remove(labels)
        labels = E.SubElement(line, c('dLbls')); E.SubElement(labels, c('delete'), val='1')
        for axis in bar.findall(c('axId')):
            E.SubElement(line, c('axId'), val=axis.attrib['val'])
        parent = bar.getparent(); parent.insert(parent.index(bar) + 1, line)
        fmt = '0.0%'
    else:
        assert title == 'Source-preferred selected false blanks'; fmt = '0'
    for bar_group in root.findall('.//' + c('barChart')):
      for labels in bar_group.findall('.//' + c('dLbls')):
        num = labels.find(c('numFmt'))
        if num is None:
            num = E.Element(c('numFmt')); at = 0
            while at < len(labels) and E.QName(labels[at]).localname in ['dLbl', 'delete']:
                at += 1
            labels.insert(at, num)
        num.set('formatCode', fmt); num.set('sourceLinked', '0')
    entries[i] = (info, E.tostring(root, encoding='utf-8', xml_declaration=True)); found.add(title)
assert len(found) == 2
temporary = path.with_suffix('.formatting.pptx')
with ZipFile(temporary, 'w') as z:
    for info, raw in entries: z.writestr(info, raw)
temporary.replace(path)
