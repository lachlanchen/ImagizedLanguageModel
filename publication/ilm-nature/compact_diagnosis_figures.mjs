import fs from "node:fs/promises";
import path from "node:path";
import {createHash} from "node:crypto";
import {pathToFileURL} from "node:url";
import sharp from "sharp";
import {Presentation, PresentationFile} from "@oai/artifact-tool";

const [root, stage, skill, python] = process.argv.slice(2);
const {applyPresentationChartFont, finalizePresentation} = await import(pathToFileURL(
  path.join(skill, "container_tools/artifact_tool_utils.mjs")).href);
const hash = x => createHash("sha256").update(x).digest("hex");
const evidence = path.join(root, "references/evidence");
const sources = {}, bytes = {};
for (const [key, name] of Object.entries({
  v1: "compact_visual_span_learning_20261111.json",
  wider: "compact_visual_span_full_rank_learning_20261111.json",
  reference: "compact_visual_span_qwen_uncertainty_20261111.json",
  diagnosis: "compact_visual_span_diagnosis_20261111.json",
})) {
  bytes[key] = await fs.readFile(path.join(evidence, name));
  sources[key] = JSON.parse(bytes[key]);
}
const D = sources.diagnosis;
if (D.records.length !== 64 || !D.original_pixel_metrics_independently_recomputed
    || !D.all_reference_supplied_diagnostics) throw Error("Completed diagnostic required");
const steps = [0, 1956, 4049, 6152, 8172];
const font = "Arial", ink = "#192936", muted = "#43535C", blue = "#386985",
  brown = "#9B7457", grey = "#7E8990", grid = "#DDE4E8";
const deck = Presentation.create({slideSize: {width: 1600, height: 1600}});
const slides = [], chartData = [], pictures = [];
function label(slide, text, x, y, w, h, size=27, bold=false) {
  const s = slide.shapes.add({geometry: "textbox", position: {left:x, top:y, width:w, height:h},
    fill:"none", line:{fill:"none", width:0}});
  s.text = text;
  s.text.style = {typeface:font, fontSize:size, bold, color:bold?ink:muted, autoFit:"none"};
}
function slide(title) {
  const s=deck.slides.add(); s.background.fill="#FFFFFF"; slides.push(s);
  label(s,title,48,28,1504,70,46,true); return s;
}
const axes = {textStyle:{typeface:font,fontSize:25,fill:muted},line:{fill:muted,width:1}};
function chart(s,type,config) {
  const original=config.series.map(x=>({name:x.name,values:[...x.values]}));
  // Excel stores at most 15 significant digits. Bind the full source values
  // separately and author a declared 12-significant-digit chart snapshot.
  config.series=config.series.map(x=>({...x,values:x.values.map(v=>Number(v.toPrecision(12)))}));
  const c=s.charts.add(type,config); applyPresentationChartFont(c,{fontFamily:font});
  chartData.push({slide:slides.indexOf(s)+1,type,categories:config.categories,
    original_source_series:original,chart_significant_digits:12,
    series:config.series.map(x=>({name:x.name,values:x.values}))}); return c;
}
const titleStyle={typeface:font,fontSize:30,bold:true,fill:ink};
const legend={position:"bottom",overlay:false,textStyle:{fontSize:25,fill:muted}};
const unitY={...axes,min:0,max:1,majorUnit:.25,numberFormatCode:"0.00",majorGridlines:{fill:grid,width:1}};

const s1=slide("Wider writer: complete-answer evaluation");
for (const [i,[metric,title,n]] of [
  ["qa_complete_exact","Complete exact QA (32 questions)",32],
  ["qa_both_correct","Both questions correct (16 pairs)",16],
].entries()) {
  const series = ["v1","wider"].map((key,j)=>{
    const rows=steps.map(step=>sources[key].metrics.find(r=>r.step===step&&r.condition==="normal"&&r.metric===metric));
    if(rows.some(r=>!r||r.count!==0||r.cases!==n)) throw Error("Language curve changed");
    return {name:j?"Wider writer":"Compact V1",values:rows.map(r=>r.mean),
      line:{fill:j?brown:blue,width:j?2:3},marker:{symbol:j?"square":"circle",size:j?7:12}};
  });
  const ref=sources.reference.metrics.find(r=>r.metric===metric);
  series.push({name:"Text reference (CPU)",values:steps.map(()=>ref.mean),
    line:{fill:grey,width:2},marker:{symbol:"none"}});
  label(s1,i?"b":"a",40+i*780,130,40,50,34,true);
  chart(s1,"line",{position:{left:70+i*780,top:135,width:705,height:540},
    title,titlePlacement:"aboveChart",titleTextStyle:titleStyle,categories:steps.map(String),series,
    hasLegend:true,legend:{...legend,textStyle:{fontSize:22,fill:muted}},lineOptions:{smooth:false},
    xAxis:{...axes,title:{text:"Optimizer update",textStyle:{fontSize:24,fill:muted}}},
    yAxis:{...unitY,numberFormatCode:"0%"}});
  label(s1,`Both visual models: 0/${n} at all five checkpoints\nText reference: ${ref.count}/${n}`,80+i*780,690,690,90,27);
}
label(s1,"The two zero-valued visual curves overlap. Pointwise group intervals are [0, 0].\nConstant observations do not imply zero population success.",80,795,1440,80);
const rows=[...sources.v1.normal_output_counts.map(x=>({...x,model:"V1"})),
  ...sources.wider.normal_output_counts.map(x=>({...x,model:"Wide"}))];
if(rows.length!==10||rows.some(x=>x.stopped+x.capped!==44)) throw Error("Termination counts");
label(s1,"c",40,925,40,50,34,true);
chart(s1,"bar",{position:{left:75,top:920,width:1450,height:475},
  title:"Normal-prompt termination (44 inputs per checkpoint)",titlePlacement:"aboveChart",titleTextStyle:titleStyle,
  categories:rows.map(x=>`${x.model}\n${x.step}`),series:[
    {name:"Predicted stop",values:rows.map(x=>x.stopped),fill:blue},
    {name:"Generation budget reached",values:rows.map(x=>x.capped),fill:brown,
      dataLabelOverrides:rows.map((x,idx)=>({idx,showValue:x.capped>0,showSeriesName:false,
        showCategoryName:false,position:"center",textStyle:{typeface:font,fontSize:25,fill:"#FFFFFF"}}))}],
  barOptions:{direction:"column",grouping:"stacked",gapWidth:70},hasLegend:true,legend,
  xAxis:axes,yAxis:{...axes,min:0,max:44,majorUnit:11,numberFormatCode:"0",majorGridlines:{fill:grid,width:1}},
  dataLabels:{showValue:true,position:"center",textStyle:{fontSize:25,fill:"#FFFFFF"}}});
label(s1,"Blank outputs remain in the denominators. Wider final: 4 blank, 13 stops and 31 capped.\nEach model retains all 372 outputs and a single model-based opaque review.",80,1410,1440,90);
label(s1,"No independent human adjudication. The CPU reference gives task quality, without a hardware-matched efficiency comparison.",80,1520,1440,72,25);
s1.speakerNotes.textFrame.setText("Sources: references/evidence/compact_visual_span_learning_20261111.json; compact_visual_span_full_rank_learning_20261111.json; compact_visual_span_qwen_uncertainty_20261111.json. The zero visual curves coincide. Text-reference QA 95% group interval is 9.375–43.75%, paired interval 0–31.25%. All visual intervals are [0,0]. Different writer widths, same registered training exposure and evaluation. No independent seeds or human review.");

const arms=["codec","visual_reference_8","visual_reference_64","native_reference_8",
  "shuffled_reference_8","emitted_fixed_counts_8","learned_prompt_emitted_8"];
const names=["Codec","Correct history, 8 steps","Correct history, 64 steps","Native history, 8 steps",
  "Wrong context, 8 steps","Emitted history, fixed counts","Learned grouping + emitted history"];
const s2=slide("Frozen diagnostic: conditional writing");
for(const [i,split] of ["train","development"].entries()) {
  label(s2,i?"b":"a",40,130+i*645,40,50,34,true);
  chart(s2,"bar",{position:{left:70,top:130+i*645,width:1450,height:550},
    title:`${i?"Development":"Training"} targets (${i?131:122} glyphs in 16 records)`,titlePlacement:"aboveChart",titleTextStyle:titleStyle,
    categories:names,series:["v1","wider"].map((model,j)=>({name:j?"Wider writer":"Compact V1",fill:j?brown:blue,
      values:arms.map(arm=>{
        const r=D.aggregate_pixel_metrics.find(x=>x.model===model&&x.split===split&&x.arm===arm);
        if(!r||r.glyphs!==(i?131:122))throw Error("Diagnostic denominator");
        return r.foreground_f1;
      })})),barOptions:{direction:"bar",grouping:"clustered",gapWidth:40},hasLegend:true,legend,
    xAxis:axes,yAxis:{...unitY,title:{text:"Aggregate foreground F1",textStyle:{fontSize:25,fill:muted}}}});
}
label(s2,"Reference lengths are supplied to every writer arm. History/count supplies vary as labeled.\nThese measurements diagnose writing and are separate from autonomous language accuracy.",80,1395,1440,90);
label(s2,"The first span has 0/20 exact training glyphs and 0/26 exact development glyphs in both models.\nDirect codec reconstruction exceeds 0.9998 foreground F1. More sampling steps do not repair writing.",80,1500,1440,90,26);
s2.speakerNotes.textFrame.setText("Source: references/evidence/compact_visual_span_diagnosis_20261111.json, aggregate_pixel_metrics. All pixel counts recomputed from persisted gold and generated arrays. Sixteen training and sixteen development records, eight QA and eight book records in each split, up to eight following spans per record. All model arms receive reference output lengths, fixed initial noise and frozen weights. Native-history arm uses training-only source vectors as an explicit diagnostic bypass. No confidence intervals or autonomous language scores. No output repair.");

const s3=slide("Reader alignment and original diagnostic glyphs");
for (const [i,split] of ["train","development"].entries()) {
  const rows=["v1","wider"].map(model=>D.reader_metrics.filter(r=>r.model===model&&r.split===split));
  const weighted=(rs,key)=>rs.reduce((a,r)=>a+r[key]*r.native_positions,0)/rs.reduce((a,r)=>a+r.native_positions,0);
  label(s3,i?"b":"a",40+i*780,130,40,50,34,true);
  chart(s3,"bar",{position:{left:70+i*780,top:135,width:705,height:500},
    title:i?"Development positions":"Training positions",titlePlacement:"aboveChart",titleTextStyle:titleStyle,
    categories:["Compact V1","Wider writer"],series:[
      {name:"Achieved error",values:rows.map(rs=>weighted(rs,"achieved_feedback_relative_mse")),fill:blue},
      {name:"Best affine error floor",values:rows.map(rs=>weighted(rs,"affine_relative_mse_floor")),fill:brown}],
    barOptions:{direction:"column",grouping:"clustered",gapWidth:90},hasLegend:true,legend,
    xAxis:axes,yAxis:{...unitY,title:{text:"Relative squared error",textStyle:{fontSize:24,fill:muted}}}});
}
label(s3,"The terminal-normalized 384-channel head has a 383-dimensional affine family\nfor 2,048-dimensional targets. The floor is a target-alignment restriction, not a language bound.",80,680,1440,90);
label(s3,"c   First registered development QA example: original two-glyph outputs",48,825,1504,60,32,true);
const rawRoot=path.join(evidence,"compact_visual_span_diagnosis_20261111");
const caseName="development_qa_16";
for(const [j,model] of ["v1","wider"].entries()) {
  label(s3,j?"Wider writer":"Compact V1",805*j+50,895,700,45,30,true);
  for(const [i,[file,title]] of [["reference.png","Reference"],["codec.png","Direct codec"],
    ["visual_reference_8.png","Correct history, 8 steps"],["visual_reference_64.png","Correct history, 64 steps"],
    ["emitted_fixed_counts_8.png","Emitted history, fixed counts"]].entries()) {
    const relative=`raw_examples/${model}/${caseName}/${file}`;
    const binding=D.raw_examples.find(x=>x.file===relative); const raw=await fs.readFile(path.join(rawRoot,relative));
    if(!binding||hash(raw)!==binding.sha256)throw Error("Original diagnostic image changed");
    const meta=await sharp(raw).metadata();
    if(meta.width!==64||meta.height!==32)throw Error("Expected two untouched glyphs");
    label(s3,title,805*j+50,966+i*94,430,50,27);
    const zoom=await sharp(raw).resize(128,64,{kernel:"nearest"}).png().toBuffer();
    s3.images.add({blob:zoom,contentType:"image/png",fit:"contain",alt:`${model} ${caseName} ${file}, original pixels at 2x nearest scale`,
      position:{left:805*j+510,top:950+i*94,width:128,height:64}});
    pictures.push({slide:3,source:path.relative(root,path.join(rawRoot,relative)),sha256:hash(raw),display_scale:2,
      interpolation:"nearest",display_png_sha256:hash(zoom)});
  }
}
label(s3,"Display rule: first registered case of this split/kind, selected during analysis, not preregistered.\nAll answer pixels are original. Supplies and wrong forms are retained, without OCR replacement.",50,1480,1500,100,26);
s3.speakerNotes.textFrame.setText("Source: references/evidence/compact_visual_span_diagnosis_20261111.json, reader_metrics and raw_examples. Native-position-weighted relative MSE and best affine floors use the selected cases, not the complete training distribution. Reader affine rank 383 was measured for each final head. Raw case development_qa_16 is the deterministic first case in that split/kind, with an analysis-time display rule. Original reference, codec and model images are embedded with exact nearest-neighbour 2x expansion and are not repaired. These are supplied-reference diagnostics, not autonomous answers.");

await fs.mkdir(path.join(stage,"output"),{recursive:true});
const candidate=path.join(stage,"candidate.pptx");
await(await PresentationFile.exportPptx(deck)).save(candidate);
for(const [i,s] of slides.entries()){
  const b=await s.export({format:"png",scale:1});
  await fs.writeFile(path.join(stage,`artifact-slide-${i+1}.png`),new Uint8Array(await b.arrayBuffer()));
}
await fs.writeFile(path.join(stage,"source_bindings.json"),JSON.stringify({
  source_sha256:Object.fromEntries(Object.entries(bytes).map(([k,v])=>[k,hash(v)])),
  charts:chartData,pictures,native_charts:7,original_answer_images:10,
  display_rule_added_during_analysis:true,reference_supplied_diagnostics:true,
},null,2)+"\n");
await finalizePresentation({workspaceDir:stage,candidatePath:candidate,
  finalPath:path.join(stage,"output/Compact_diagnosis_figures.pptx"),explicitTotalSlideCount:3,
  requiredNativeChartOwnerSlides:[1,2,3],materializeLiteralChartWorkbooks:true,
  fontPolicy:{basis:"design",families:[font]},pythonExecutable:python,
  integrityValidatorPath:path.join(skill,"container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath:path.join(skill,"container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs:["--expected-slide-size-emu","15240000,15240000","--validate-heading-fit"],
  verifyArtifactToolImport:true,receiptPath:path.join(stage,"presentation_validation.json")});
