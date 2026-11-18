"""Render the complete blinded contextual sample from original physical output bytes."""
from io import BytesIO

import numpy as np
from PIL import Image
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from scripts.evaluate_conditional_ink import WORK, inputs, load, require, sha
from scripts.report_conditional_ink import EVIDENCE, sites_independent


def lines(pdf, tiles, top):
    for offset in range(0, len(tiles), 35):
        tiles_row = tiles[offset:offset+35]
        ink = np.unpackbits(tiles_row,axis=1).reshape(-1,32,32)
        image = Image.fromarray((255 * (1 - np.concatenate(ink,axis=1))).astype('uint8'))
        buf = BytesIO()
        image.save(buf,format='PNG')
        buf.seek(0)
        pdf.drawImage(ImageReader(buf), 36,top-14,width=len(tiles_row)*14,height=14)
        top -= 18
    return top


def render():
    path = EVIDENCE/'reading_gallery.pdf'
    if path.exists():
        raise FileExistsError(path)
    rows,bank = inputs('confirmation')
    locations = list(sites_independent(rows))
    reading = WORK/'reading'
    result, binding = load(reading/'result.json'),load(reading/'binding.json')
    require(binding==result['binding'] and sha(reading/'mapping.json')==binding['mapping']
            and sha(reading/'ratings.txt')==result['ratings_sha256'],'Reading record changed')
    report = load(WORK/'results/confirmation/report.json')
    streams = {}
    for name in ('interaction','marginal','local'):
        source = WORK/'results/confirmation'/(name+'.npz')
        require(sha(source)==report['arms'][name]['pixels_sha256'],'Source stream changed')
        with np.load(source) as z:
            streams[name]=(np.split(z['pixels'],z['offsets'][1:-1]),z['stopped'])
    pdf=canvas.Canvas(str(path),pagesize=A4,pageCompression=1,invariant=1)
    pdf.setTitle('Chinese phrase insertions in context | 48 blinded cases')
    width,height=A4
    top=0
    page=0
    for number,(mapping,scores) in enumerate(zip(load(reading/'mapping.json'),result['rows'],strict=True),1):
        site=mapping['site']
        i,c,j,k=locations[site]
        r=rows[i]
        groups=[('Classical sentence',bank[r['x']]),('Modern prefix (last 35)',bank[r['y'][max(0,j-35):j]]),
                ('Following translation (first 35)',bank[r['y'][k:k+35]])]
        for key,label in [('interaction','Adjacent-pattern writer'),('marginal','Earlier writer'),('local','Whole-phrase counts'),('reference','Recorded insertion')]:
            grade={'A':'acceptable','I':'incompatible','U':'uncertain'}[scores[key]]
            tiles=bank[r['y'][j:k]] if key=='reference' else streams[key][0][site]
            ending='' if key=='reference' else (' / stopped' if streams[key][1][site] else ' / reached cap')
            groups.append((label+'  /  '+grade+ending,tiles))
        needed=29+sum(18+18*((len(a)+34)//35) for _,a in groups)+10
        if top-needed<34:
            if page:
                pdf.showPage()
            page+=1
            pdf.setFillColor(HexColor('#193849'))
            pdf.setFont('Helvetica-Bold',16)
            pdf.drawString(36,height-38,'Chinese phrase insertions in context')
            pdf.setFont('Helvetica',9)
            pdf.drawString(36,height-54,'All 48 sampled cases in blinded reading order. Gray labels give the frozen contextual judgments.')
            pdf.drawRightString(width-30,height-38,str(page))
            top=height-79
        pdf.setFillColor(HexColor('#116A68'))
        pdf.setFont('Helvetica-Bold',12)
        pdf.drawString(36,top,f'Case {number:02d}    /    source boundary after cell {c}    /    insertion site {site}')
        top-=19
        for label,tiles in groups:
            pdf.setFillColor(HexColor('#48555C'))
            pdf.setFont('Helvetica',9)
            pdf.drawString(36,top,label if len(tiles) else label+'  /  empty')
            top=lines(pdf,tiles,top-3)-15
        pdf.setStrokeColor(HexColor('#C5D7D6'));pdf.line(36,top+5,width-36,top+5)
        top-=16
    pdf.save()
    print(page,'pages',sha(path))


if __name__=='__main__':
    render()
