#!/usr/bin/env python3
"""Preserve fourteen evidence slides and append native free-writing results."""

import json

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE,XL_LEGEND_POSITION
from pptx.util import Pt
import torch

import generate_reading_editable_figures
from generate_evidence import HERE,ROOT
from generate_native_pixel_evidence import RUN
from generate_v55_editable_figures import px,text
from generate_v57_editable_figures import picture
from generate_v57_evidence import pad_answer


def native_chart(slide,categories,series,x,width,maximum,colors=("28638B","AE593F")):
    data=CategoryChartData();data.categories=categories
    for label,values in series:data.add_series(label,values)
    chart=slide.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED,px(x),px(155),px(width),px(330),data).chart
    chart.has_title=False;chart.has_legend=len(series)>1
    if chart.has_legend:
        chart.legend.position=XL_LEGEND_POSITION.BOTTOM
        chart.legend.include_in_layout=False;chart.legend.font.name="Arial";chart.legend.font.size=Pt(13)
    chart.value_axis.minimum_scale=0;chart.value_axis.maximum_scale=maximum
    chart.value_axis.tick_labels.font.size=Pt(13);chart.category_axis.tick_labels.font.size=Pt(13)
    for series,color in zip(chart.series,colors):
        series.format.fill.solid();series.format.fill.fore_color.rgb=RGBColor.from_string(color)
        series.format.line.fill.background()


def main():
    generate_reading_editable_figures.main()
    report=json.loads((RUN/"development/report.json").read_text())
    prs=Presentation(HERE/"ILM_Figures_editable.pptx")
    if len(prs.slides)!=14:raise ValueError("unexpected previous deck")
    slide=prs.slides.add_slide(prs.slide_layouts[0])
    text(slide,"Native free writing: improved pixel fit, failed language generation",45,25,1510,75,size=30,bold=True)
    text(slide,"Held-out answer-pixel NLL (nats)",75,110,640,45,size=21,bold=True)
    native_chart(slide,["Released base","Adapted"],[("Nats / pixel",[report["likelihood"][r] for r in ("base","trained")])],65,620,.8)
    text(slide,"Generated image scores (%)",800,110,720,45,size=21,bold=True)
    routes=("trained","trained_blank","base")
    native_chart(slide,["Adapted","Blank question","Base"],[(label,[100*report["measurements"][r][key] for r in routes])
        for label,key in (("Exact-raster F1","raster_f1"),("Foreground F1","pixel_foreground_f1"))],770,760,20,colors=("AE593F","28638B"))
    text(slide,"0/256 exact answers and 0/393 pairs both correct. Five of ten registered gates pass.\n477.15M externally pretrained parameters, fully adapted for 1,000 updates on GPU 0.",
        65,510,1470,85,size=21)
    from ilm.visual_lm.visual_qa_data import VisualQARenderer,read_questions
    records={r["id"]:r for r in read_questions(ROOT/"data/visual_qa/cmrc2018_short/development.jsonl")}
    renderer=VisualQARenderer()
    text(slide,"Retained reference",250,615,500,35,size=18,bold=True)
    text(slide,"Actual free-pixel answer",900,615,500,35,size=18,bold=True)
    for i,identifier in enumerate(report["primary_ids"][:2]):
        raw=torch.load(RUN/"development/predictions/trained"/(identifier+".pt"),map_location="cpu",weights_only=False)["images"]
        reference=renderer(records[identifier]["references"][0]["text"])
        y=660+i*65;text(slide,f"Audit row {i+1}",50,y,180,40,size=18)
        picture(slide,pad_answer(reference,16),250,y,480,30)
        picture(slide,pad_answer(raw,16),900,y,480,30)
    text(slide,"Training: 1,501.1 s and 9.912 GiB. Offline native-PNG replay passes. External pretraining and reused development disclosed.",
        45,825,1510,55,size=15)
    slide.notes_slide.notes_text_frame.text="Sources: evidence/v61_development_report.json; evidence/v61_training_report.json; evidence/v61_standalone_verification.json. Raw first two fixed primary answers; no output repair. See Supplementary Note 20."
    prs.save(HERE/"ILM_Figures_editable.pptx");print("Saved 15 editable evidence slides")


if __name__=="__main__":main()
