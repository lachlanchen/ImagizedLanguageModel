import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation,PresentationFile} from '@oai/artifact-tool';
const [root,stage,skill,python]=process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')));
const evidence=path.join(root,'references/evidence/book_breadth_v1');
const load=async name=>JSON.parse(await fs.readFile(path.join(evidence,name)));
const deck=Presentation.create({slideSize:{width:1060,height:652}});
function label(s,text,x,y,w,size,bold=false) {
  const b=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:size*1.6},fill:'none',line:{fill:'none',width:0}});
  b.text=text; b.text.style={typeface:'Arial',fontSize:size,bold,color:'#213634',autoFit:'none',insets:{left:0,right:0,top:0,bottom:0}};
}
const first=deck.slides.add();
first.background.fill='#FFFFFF';
label(first,'What changes when a writer reads more distinct passages?',26,22,1012,32,true);
const {panels,colors}=await load('charts.json');
panels.forEach((p,i)=>{
  const x=20+526*i;
  label(first,p.title,x,115,508,21,true);
  label(first,p.unit,x,162,508,17);
  const c=first.charts.add(i?'bar':'line',{position:{left:x,top:210,width:504,height:340},
    categories:p.labels,series:Object.entries(p.series).map(([name,values])=>({name,values,fill:colors[name],line:{fill:colors[name],width:2}})),
    hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:13}},xAxis:{textStyle:{fontSize:13}},
    yAxis:i?{min:0,max:48,majorUnit:12}:{min:1,max:3,majorUnit:.5},
    ...(i?{barOptions:{direction:'column',grouping:'clustered'}}:{})});
  applyPresentationChartFont(c,{fontFamily:'Arial'});
});
first.speakerNotes.textFrame.setText('The writers saw the same target positions in every batch. Each point averages teacher-forced loss across three development domains. Meaning is graded on independent free replies.');
for(const [sceneName,title] of [['rasters.json','Original literary inputs and freely written replies'],['credited.json','Every broad-writer reply credited by the grader']]){
const second=deck.slides.add();second.background.fill='#FFFFFF';
label(second,title,26,12,1010,32,true);
const scene=await load(sceneName);
const scale=Math.min(1008/scene.width,555/scene.height);
for(const it of scene.items){
  const x=26+it.x*scale,y=64+it.y*scale;
  if(it.type==='text') label(second,it.text,x,y,1020-x,it.size*scale,it.bold);
  else second.images.add({blob:await fs.readFile(path.join(root,'publication/ilm-nature/figures/book_breadth',it.name)),
    contentType:'image/png',alt:it.name,fit:'contain',position:{left:x,top:y,width:it.w*scale,height:it.h*scale}});
}
}
for(const d of ['draft','output']) await fs.mkdir(path.join(stage,d),{recursive:true});
const candidatePath=path.join(stage,'draft/book.pptx'),finalPath=path.join(stage,'output/Reading_bilingual_books.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:3,
  materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],verifyArtifactToolImport:true,
  integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','10096500,6210300'],receiptPath:path.join(stage,'presentation_validation.json')});
