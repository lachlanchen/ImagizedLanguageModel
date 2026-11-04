#!/usr/bin/env python3
"""Append native density mechanism and evidence to the preserved V61 deck."""

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR,MSO_SHAPE

from generate_correlated_density_evidence import RUN,completed_report
from generate_evidence import HERE,ROOT
from generate_native_pixel_editable_figures import native_chart
from generate_v55_editable_figures import px,text
from generate_v57_editable_figures import picture
from generate_v57_evidence import pad_answer
import torch


def box(slide,label,x,y,w,h):
    shape=slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,px(x),px(y),px(w),px(h))
    shape.fill.solid();shape.fill.fore_color.rgb=RGBColor.from_string("F2F5F6")
    shape.line.color.rgb=RGBColor.from_string("677B87")
    text(slide,label,x+8,y+12,w-16,h-15,size=21,center=True)


def arrow(slide,x1,y1,x2,y2):
    line=slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,px(x1),px(y1),px(x2),px(y2))
    line.line.color.rgb=RGBColor.from_string("536674")
    # OOXML arrowhead keeps this a native, editable connector.
    from pptx.oxml.xmlchemy import OxmlElement
    end=OxmlElement("a:tailEnd");end.set("type","triangle");line.line._get_or_add_ln().append(end)


def main():
    report=completed_report()
    prs=Presentation(HERE/"baselines/v61/ILM_Figures_editable.pptx")
    if len(prs.slides)!=15:raise ValueError("verified V61 deck required")
    slide=prs.slides.add_slide(prs.slide_layouts[0])
    text(slide,"Conditional whole-image density",45,25,1510,75,size=32,bold=True)
    text(slide,"Training: prompt and observed answer images",65,125,1430,45,size=23,bold=True)
    boxes=[(65,210,300,100,"Visible writing\n32 × 64 blocks"),(450,210,300,100,"Causal visual\ntransformer"),(835,210,300,100,"Conditional experts\nand gate weights"),(1220,210,300,100,"Exact joint\nimage likelihood")]
    for x,y,w,h,label in boxes:box(slide,label,x,y,w,h)
    for left,right in zip(boxes,boxes[1:]):arrow(slide,left[0]+left[2],260,right[0],260)
    text(slide,"p(y | h) = Σk πk(h) Πi Bernoulli(yi ; sigmoid(aki(h)))",150,360,1300,70,size=29,center=True)
    text(slide,"One shared expert choice couples all pixels in a bitmap. Each expert is a neural function of context.\nK = 1: 477.15M parameters. K = 8: 490.00M. Same examples and update budget; different head size.",
         80,450,1450,100,size=21)
    text(slide,"Inference: image prompt → candidate bitmaps → score under full density → physical pixels",65,610,1450,70,size=24,bold=True)
    text(slide,"Feed generated pixels back into the visual context; stop at an all-white image block.\nComponent-mode search is approximate MAP. No character bank, OCR, Unicode IDs or external LM in the actor.",
         80,705,1450,100,size=21)
    text(slide,"Externally pretrained MIXAR foundation. The mixture law is a tested hypothesis, not a guarantee of readable language.",45,840,1510,45,size=15)
    slide.notes_slide.notes_text_frame.text="Source: references/correlated_native_density_v62_protocol.md; Supplementary Note 21. Native editable mechanism; no generated artwork."
    slide=prs.slides.add_slide(prs.slide_layouts[0])
    text(slide,"Conditional density: held-out fit and generated writing",45,25,1510,75,size=30,bold=True)
    text(slide,"Held-out answer-pixel NLL (nats)",75,110,640,45,size=21,bold=True)
    native_chart(slide,["K = 1","K = 8"],[("Nats / pixel",[report["likelihood"][r] for r in ("k1","k8")])],65,620,max(report["likelihood"].values())*1.3)
    text(slide,"Generated image scores (%)",800,110,720,45,size=21,bold=True)
    routes=("k1","k1_blank","k8","k8_blank")
    maximum=max(100*report["measurements"][r][key] for r in routes for key in ("raster_f1","pixel_foreground_f1"))*1.3+1
    native_chart(slide,["K1","K1 blank Q","K8","K8 blank Q"],[(label,[100*report["measurements"][r][key] for r in routes]) for label,key in (("Exact-raster F1","raster_f1"),("Foreground F1","pixel_foreground_f1"))],770,760,maximum,colors=("AE593F","28638B"))
    exact=[round(256*report["measurements"][r]["exact_answer"]) for r in ("k1","k8")]
    paired=[round(393*report["paired_both_exact"][r]) for r in ("k1","k8")]
    text(slide,f"Exact answers K1 / K8: {exact[0]} / {exact[1]} of 256. Both-correct pairs: {paired[0]} / {paired[1]} of 393.\nRegistered decision: {report['decision'].replace('_',' ')}. General image-language quality remains unqualified.",65,510,1470,85,size=21)
    from ilm.visual_lm.visual_qa_data import VisualQARenderer,read_questions
    records={r["id"]:r for r in read_questions(ROOT/"data/visual_qa/cmrc2018_short/development.jsonl")};renderer=VisualQARenderer()
    for label,x in (("Retained reference",240),("K = 1",710),("K = 8",1160)):text(slide,label,x,620,400,35,size=18,bold=True)
    for i,identifier in enumerate(report["primary_ids"][:2]):
        y=675+i*60;text(slide,f"Audit row {i+1}",45,y,170,40,size=18)
        images=[renderer(records[identifier]["references"][0]["text"])]+[torch.load(RUN/f"development/predictions/k{k}"/(identifier+".pt"),map_location="cpu",weights_only=False)["images"] for k in (1,8)]
        for x,cells in zip((240,710,1160),images):picture(slide,pad_answer(cells,20),x,y,390,19.5)
    text(slide,"First two fixed primary answers, unrepaired; white display padding. One seed, reused development, external pretraining disclosed.",45,840,1510,45,size=15)
    slide.notes_slide.notes_text_frame.text="Sources: evidence/v62_development_report.json, v62_k1_training_report.json, v62_k8_training_report.json; Figure 7 and Supplementary Note 21."
    prs.save(HERE/"ILM_Figures_editable.pptx");print("Saved 17 native editable evidence slides")


if __name__=="__main__":main()
