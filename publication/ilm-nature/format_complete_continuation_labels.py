#!/usr/bin/env python3
"""Set native count/percentage label formats without changing chart data."""
from pathlib import Path
import sys
from zipfile import ZipFile
from lxml import etree as ET

path=Path(sys.argv[1]);ns='http://schemas.openxmlformats.org/drawingml/2006/chart'
formats={'Complete exact answers / 32':'0','Readable complete answers / 32':'0',
         'Book tiles above clean ink threshold':'0.0%'}
with ZipFile(path) as z:entries=[(info,z.read(info.filename)) for info in z.infolist()]
found=set()
for i,(info,raw) in enumerate(entries):
 if '/charts/' not in info.filename or not info.filename.endswith('.xml'):continue
 root=ET.fromstring(raw);node=root.find('.//{'+ns+'}title');title=''.join(node.itertext()) if node is not None else ''
 if title not in formats or title in found:raise ValueError('Unexpected/repeated chart: '+title)
 groups=root.findall('.//{'+ns+'}dLbls')
 if not groups:raise ValueError('Native labels absent')
 for group in groups:
  fmt=group.find('{'+ns+'}numFmt')
  if fmt is None:
   fmt=ET.Element('{'+ns+'}numFmt');at=0
   while at<len(group) and ET.QName(group[at]).localname in ['dLbl','delete']:at+=1
   group.insert(at,fmt)
  fmt.set('formatCode',formats[title]);fmt.set('sourceLinked','0')
 entries[i]=(info,ET.tostring(root,encoding='utf-8',xml_declaration=True));found.add(title)
if found!=set(formats):raise ValueError('All three native charts required')
temporary=path.with_suffix('.formatting.pptx')
with ZipFile(temporary,'w') as z:
 for info,raw in entries:z.writestr(info,raw)
temporary.replace(path)
