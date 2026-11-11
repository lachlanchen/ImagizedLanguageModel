#!/usr/bin/env python3
"""Set the native six-decimal label format absent from the chart authoring API."""
from pathlib import Path
import sys
from lxml import etree as ET
from zipfile import ZipFile

path=Path(sys.argv[1]);ns='http://schemas.openxmlformats.org/drawingml/2006/chart'
with ZipFile(path) as source:
    entries=[(info,source.read(info.filename)) for info in source.infolist()]
found=False
for i,(info,raw) in enumerate(entries):
    if '/charts/' not in info.filename or not info.filename.endswith('.xml'):continue
    root=ET.fromstring(raw)
    title=root.find('.//{'+ns+'}title')
    if title is None or 'Mean first-tile foreground F1' not in ''.join(title.itertext()):continue
    if found:raise ValueError('Physical-F1 native chart repeated')
    labels=root.findall('.//{'+ns+'}dLbls')
    if len(labels)!=1:raise ValueError('Exactly one physical-F1 label group required')
    old=labels[0].find('{'+ns+'}numFmt')
    if old is None:
        old=ET.Element('{'+ns+'}numFmt')
        at=0
        while at<len(labels[0]) and ET.QName(labels[0][at]).localname in ['dLbl','delete']:at+=1
        labels[0].insert(at,old)
    old.set('formatCode','0.000000');old.set('sourceLinked','0')
    entries[i]=(info,ET.tostring(root,encoding='utf-8',xml_declaration=True));found=True
if not found:raise ValueError('Physical-F1 native chart absent')
temporary=path.with_suffix('.formatting.pptx')
with ZipFile(temporary,'w') as output:
    for info,raw in entries:output.writestr(info,raw)
temporary.replace(path)
