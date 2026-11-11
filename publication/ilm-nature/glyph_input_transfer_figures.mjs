import fs from "node:fs/promises";
import path from "node:path";
import {createHash} from "node:crypto";
import {pathToFileURL} from "node:url";
import sharp from "sharp";
import {Presentation, PresentationFile} from "@oai/artifact-tool";

const [root,stage,skill,python]=process.argv.slice(2);
const {applyPresentationChartFont,finalizePresentation}=await import(pathToFileURL(
  path.join(skill,"container_tools/artifact_tool_utils.mjs")).href);
const hash=x=>createHash("sha256").update(x).digest("hex");
const evidence=path.join(root,"references/evidence"),bytes={},data={};
for(const [key,name] of Object.entries({
  fixed:"glyph_tail_compact_learning_20261111.json",
  proportional:"glyph_tail_input_presentation_learning_20261111.json",
})){
  bytes[key]=await fs.readFile(path.join(evidence,name));data[key]=JSON.parse(bytes[key]);
}
const qa=k=>data[k].all_condition_counts.find(r=>r.kind==="qa"&&r.condition==="normal");
for(const [key,values] of [["fixed",[2,1,3,8]],["proportional",[6,0,11,19]]]){
  const r=qa(key);
  if(r.cases!==32||JSON.stringify([r.complete_exact,r.paired_both_correct,r.readable,r.stopped])!==JSON.stringify(values))
    throw Error("Complete frozen QA measurement changed");
}
if(data.proportional.retain_adapter_development_rule_passed||data.proportional.new_optimizer_updates!==0)
  throw Error("This is a failed-retention frozen presentation comparison");
const font="Arial",ink="#192936",muted="#43535C",blue="#386985",brown="#9B7457",grid="#DDE4E8";
const deck=Presentation.create({slideSize:{width:1600,height:1600}});
const s=deck.slides.add();s.background.fill="#FFFFFF";
const charts=[],pictures=[];
function label(text,x,y,w,h,size=27,bold=false){
  const z=s.shapes.add({geometry:"textbox",position:{left:x,top:y,width:w,height:h},
    fill:"none",line:{fill:"none",width:0}});
  z.text=text;z.text.style={typeface:font,fontSize:size,bold,color:bold?ink:muted,autoFit:"none"};
}
const axes={textStyle:{typeface:font,fontSize:24,fill:muted},line:{fill:muted,width:1}};
const legend={position:"bottom",overlay:false,textStyle:{fontSize:23,fill:muted}};
const titleTextStyle={typeface:font,fontSize:29,bold:true,fill:ink};
function chart(config){
  const c=s.charts.add("bar",config);applyPresentationChartFont(c,{fontFamily:font});
  charts.push({type:"bar",categories:config.categories,
    series:config.series.map(x=>({name:x.name,values:x.values})),
    position:config.position});
}
label("Input presentation and image-answer transfer",48,28,1504,70,46,true);
for(const [i,[field,title,n]] of [
  ["complete_exact","Exact answer (32 QA)",32],
  ["paired_both_correct","Both correct (16 pairs)",16],
  ["readable","Whole readable (32 QA)",32],
].entries()){
  label("abc"[i],40+515*i,130,40,50,34,true);
  chart({position:{left:70+515*i,top:135,width:465,height:455},title,
    titlePlacement:"aboveChart",titleTextStyle,categories:["Normal QA"],
    series:[{name:"Fixed cells",values:[qa("fixed")[field]],fill:blue},
            {name:"Proportional",values:[qa("proportional")[field]],fill:brown}],
    barOptions:{direction:"column",grouping:"clustered",gapWidth:90},hasLegend:true,legend,
    xAxis:axes,yAxis:{...axes,min:0,max:n,majorUnit:n/4,numberFormatCode:"0",majorGridlines:{fill:grid,width:1}},
    dataLabels:{showValue:true,position:"outEnd",textStyle:{typeface:font,fontSize:30,fill:ink}}});
}
label("Exact-answer gain: 12.5 percentage points (95% interval 0 to 25).\nBoth-correct change: -6.25 points (-18.75 to 0). Both intervals include zero.",80,625,1440,94,28);
label("d",40,750,40,50,34,true);
chart({position:{left:75,top:750,width:720,height:300},
  title:"Termination (32 normal QA)",titlePlacement:"aboveChart",titleTextStyle,
  categories:["Fixed cells","Proportional"],
  series:[{name:"Predicted stop",values:[qa("fixed").stopped,qa("proportional").stopped],fill:blue},
    {name:"Budget reached",values:[32-qa("fixed").stopped,32-qa("proportional").stopped],fill:brown}],
  barOptions:{direction:"bar",grouping:"stacked",gapWidth:85},hasLegend:true,legend,
  xAxis:axes,yAxis:{...axes,min:0,max:32,majorUnit:8,numberFormatCode:"0",majorGridlines:{fill:grid,width:1}},
  dataLabels:{showValue:true,position:"center",textStyle:{typeface:font,fontSize:28,fill:"#FFFFFF"}}});
label("Same full passages and weights; zero new updates\nBoth erased controls: 0/32 exact in each layout\nInherited visual/contextual body: 7.522B parameters\nRegistered answer-retention requirement failed",850,800,700,210,28);
label("e   Complete original outputs",48,1080,750,55,32,true);
label("Fixed cells",360,1140,570,45,27,true);
label("Proportional",970,1140,570,45,27,true);
const rawRoot=path.join(evidence,"glyph_tail_input_presentation_learning_20261111");
for(const [j,[key,title]] of [
  ["qa_09_a_normal","Area: lost readable answer"],
  ["qa_09_b_normal","Location: exact retained"],
  ["qa_04_a_normal","Quantity: newly exact"],
].entries()){
  const binding=data.proportional.raw_examples.find(x=>x.case===key);
  if(!binding)throw Error("Original matched example missing");
  const names=["fixed_cell_answer.png","answer.png"];
  const originals=await Promise.all(names.map(name=>fs.readFile(path.join(rawRoot,"raw_examples",key,name))));
  const metadata=await Promise.all(originals.map(raw=>sharp(raw).metadata()));
  const scale=Math.min(2,570/Math.max(...metadata.map(x=>x.width)));
  label(title,50,1200+j*107,300,75,25);
  for(const [i,raw] of originals.entries()){
    const expected=i?binding.image_sha256:binding.matched_fixed_cell_image_sha256;
    if(hash(raw)!==expected)throw Error("Unrepaired answer PNG changed");
    const meta=metadata[i],position={left:360+i*610,top:1195+j*107,width:meta.width*scale,height:meta.height*scale};
    s.images.add({blob:raw,contentType:"image/png",fit:"contain",position,
      alt:`${key} ${i?"proportional":"fixed-cell"} original complete answer; no crop or repair`});
    pictures.push({source:path.relative(root,path.join(rawRoot,"raw_examples",key,names[i])),
      sha256:hash(raw),source_width:meta.width,source_height:meta.height,
      source_gray_pixels_sha256:hash(await sharp(raw).removeAlpha().greyscale().raw().toBuffer()),
      common_scale_for_matched_pair:scale,position,original_bytes_supplied:true,raster_resampled_before_embedding:false});
  }
}
label("Examples selected retrospectively after complete scoring. All original pixels retained.\nOne model reviewer with prior cohort familiarity; no independent human adjudication.",50,1515,1500,75,25);
s.speakerNotes.textFrame.setText("Sources: references/evidence/glyph_tail_compact_learning_20261111.json and glyph_tail_input_presentation_learning_20261111.json. Same 32 questions, sixteen paired passage groups, fixed weights and actual emitted-pixel feedback, 128-glyph cap. Both complete cohorts and erased controls retained. Strict complete canonical matching requires a natural stop; noncanonical wording is not automatically semantically false. Pointwise 95% intervals use 10,000 paired whole-passage-group bootstrap draws, seed 202611154. Reused development cohort, no independent seed or untouched confirmation. The single successful fixed-cell pair is lost under proportional presentation; the area output loses whole readability. The displayed location remains exact, and the displayed quantity is a new exact answer. These three examples were chosen after scoring to expose that tradeoff; original complete PNGs are embedded without reconstruction, cropping or raster resampling, with a common physical scale per matched row. No new training or language-learning curve. The retained larger inherited body does not establish compact-model parity or a matched-quality efficiency advantage.");

await fs.mkdir(path.join(stage,"output"),{recursive:true});
const candidate=path.join(stage,"candidate.pptx");
await(await PresentationFile.exportPptx(deck)).save(candidate);
const preview=await s.export({format:"png",scale:1});
await fs.writeFile(path.join(stage,"artifact-slide-1.png"),new Uint8Array(await preview.arrayBuffer()));
await fs.writeFile(path.join(stage,"source_bindings.json"),JSON.stringify({
  source_sha256:Object.fromEntries(Object.entries(bytes).map(([k,v])=>[k,hash(v)])),
  charts,pictures,native_charts:4,original_answer_images:6,
  display_selection_retrospective:true,zero_new_optimizer_updates:true,
  failed_registered_retention_rule:true,all_denominators_preserved:true,
},null,2)+"\n");
await finalizePresentation({workspaceDir:stage,candidatePath:candidate,
  finalPath:path.join(stage,"output/Glyph_input_transfer_figures.pptx"),explicitTotalSlideCount:1,
  requiredNativeChartOwnerSlides:[1],materializeLiteralChartWorkbooks:true,
  fontPolicy:{basis:"design",families:[font]},pythonExecutable:python,
  integrityValidatorPath:path.join(skill,"container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath:path.join(skill,"container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs:["--expected-slide-size-emu","15240000,15240000","--validate-heading-fit"],
  verifyArtifactToolImport:true,receiptPath:path.join(stage,"presentation_validation.json")});
