#!/usr/bin/env python3
"""Give both new native diagnostic charts explicit six-decimal data labels."""
from pathlib import Path
import sys
from zipfile import ZipFile
from lxml import etree as ET

path=Path(sys.argv[1]);ns='http://schemas.openxmlformats.org/drawingml/2006/chart'
titles={'Mean F1 on shared indices','Later writing with reference history'}
with ZipFile(path) as source:entries=[(info,source.read(info.filename)) for info in source.infolist()]
found=set()
for i,(info,raw) in enumerate(entries):
    if '/charts/' not in info.filename or not info.filename.endswith('.xml'):continue
    root=ET.fromstring(raw);node=root.find('.//{'+ns+'}title')
    title=''.join(node.itertext()) if node is not None else ''
    if title not in titles:raise ValueError('Unexpected chart: '+title)
    if title in found:raise ValueError('Repeated chart: '+title)
    labels=root.findall('.//{'+ns+'}dLbls')
    if not labels:raise ValueError('Native data labels absent')
    for group in labels:
        fmt=group.find('{'+ns+'}numFmt')
        if fmt is None:
            fmt=ET.Element('{'+ns+'}numFmt');at=0
            while at<len(group) and ET.QName(group[at]).localname in ['dLbl','delete']:at+=1
            group.insert(at,fmt)
        fmt.set('formatCode','0.000000');fmt.set('sourceLinked','0')
    entries[i]=(info,ET.tostring(root,encoding='utf-8',xml_declaration=True));found.add(title)
if found!=titles:raise ValueError('Both diagnostic charts required')
temporary=path.with_suffix('.formatting.pptx')
with ZipFile(temporary,'w') as output:
    for info,raw in entries:output.writestr(info,raw)
temporary.replace(path)
