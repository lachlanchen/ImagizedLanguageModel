import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation,PresentationFile} from '@oai/artifact-tool';
const [root,stage,skill,python,study="field"]=process.argv.slice(2);
const settings={
  field:{assets:"field_transfer",evidence:"field_to_glyph_v1",file:"Independent_field_writer.pptx",
    chartTitle:"Does the independently written answer preserve the event?",
    chartNote:"All four ILM conditions produce neural pixels. The source-path interventions measure the effect of exchanging equal-length pictured passages with target history supplied.",
    plateTitle:"Complete translations from the two field-conditioned writers"},
  target:{assets:"target_glyph_teaching",evidence:"target_glyph_teaching_v1",file:"Teaching_the_next_glyph.pptx",
    chartTitle:"Can teaching the next glyph improve a whole translation?",
    chartNote:"The chart on the left counts free translations on held passages. The chart on the right measures the probability deficit on training examples with the human prefix supplied.",
    plateTitle:"Original answers after target-side teaching"}
}[study];
if(!settings) throw new Error("Unknown figure study");
const {finalizePresentation,applyPresentationChartFont}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')));
const deck=Presentation.create({slideSize:{width:1060,height:652}});
const paper=path.join(root,'publication/ilm-nature');
const read=async file=>JSON.parse(await fs.readFile(file));
function text(slide,value,x,y,w,size,bold=false,color='#25413e'){
  const shape=slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:size*1.5},fill:'none',line:{fill:'none',width:0}});
  shape.text=value;shape.text.style={typeface:'Arial',fontSize:size,bold,color,autoFit:'none',insets:{left:0,right:0,top:0,bottom:0}};
}
async function plate(file,title,notes){
  const scene=await read(file),s=deck.slides.add();s.background.fill='#ffffff';
  text(s,title,24,10,1010,29,true);
  const scale=Math.min(1008/scene.width,567/scene.height);
  for(const i of scene.items){
    const x=24+scale*i.x,y=65+scale*i.y;
    if(i.type==='text') text(s,i.text,x,y,1030-x,i.size*scale,i.bold,i.color);
    else s.images.add({blob:await fs.readFile(path.join(paper,'figures',settings.assets,i.name)),contentType:'image/png',alt:i.name,fit:'contain',position:{left:x,top:y,width:i.w*scale,height:i.h*scale}});
  }
  s.speakerNotes.textFrame.setText(notes);
}
const report=await read(path.join(root,'references/evidence',settings.evidence,'chart.json'));
const s=deck.slides.add();s.background.fill='#ffffff';
text(s,settings.chartTitle,24,20,1012,30,true);
report.panels.forEach((p,i)=>{
  const x=18+i*526;text(s,p.title,x,107,510,20,true);text(s,p.unit,x,149,510,16);
  const chart=s.charts.add('bar',{position:{left:x,top:193,width:513,height:358},categories:p.labels,
    series:Object.entries(p.series).map(([name,values])=>({name,values,fill:report.colors[name]})),
    ...(study==='target'?{dataLabels:{showValue:true,position:'outEnd',textStyle:{fontSize:12,typeface:'Arial'}}}:{}),
    hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:13}},barOptions:{direction:'bar',grouping:'clustered'}});
  applyPresentationChartFont(chart,{fontFamily:'Arial'});
});
s.speakerNotes.textFrame.setText(settings.chartNote);
await plate(path.join(root,'references/evidence',settings.evidence,'scene.json'),settings.plateTitle,'Every illustrated answer was replayed on CPU from the original source PNG. Reference transcripts were unavailable to the generating writer.');
for(const d of ['draft','output']) await fs.mkdir(path.join(stage,d),{recursive:true});
const candidatePath=path.join(stage,'draft/provenance.pptx'),finalPath=path.join(stage,'output',settings.file);
await(await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:2,
materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],verifyArtifactToolImport:true,
integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
layoutArgs:['--expected-slide-size-emu','10096500,6210300'],receiptPath:path.join(stage,'presentation_validation.json')});
