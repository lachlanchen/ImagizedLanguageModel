import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation,PresentationFile} from '@oai/artifact-tool';
const [root,stage,skill,python]=process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')));
const evidence=path.join(root,'references/evidence/library_roles_v1');
const load=async name=>JSON.parse(await fs.readFile(path.join(evidence,name)));
const deck=Presentation.create({slideSize:{width:1060,height:652}});
function label(s,text,x,y,w,size,bold=false) {
  const b=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:size*1.6},fill:'none',line:{fill:'none',width:0}});
  b.text=text; b.text.style={typeface:'Arial',fontSize:size,bold,color:'#213634',autoFit:'none',insets:{left:0,right:0,top:0,bottom:0}};
}
const first=deck.slides.add();
first.background.fill='#FFFFFF';
label(first,'Does attending to a grammatical role preserve meaning?',26,22,1012,32,true);
const {panels,colors}=await load('charts.json');
panels.forEach((p,i)=>{
  const x=20+526*i;
  label(first,p.title,x,115,508,21,true);
  label(first,p.unit,x,162,508,17);
  const c=first.charts.add('bar',{position:{left:x,top:210,width:504,height:340},
    categories:p.labels,series:Object.entries(p.series).map(([name,values])=>({name,values,fill:colors[name],line:{fill:colors[name],width:2}})),
    hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:13}},xAxis:{textStyle:{fontSize:13}},
    yAxis:{min:0,max:p.upper},
    barOptions:{direction:'column',grouping:'clustered'}});
  applyPresentationChartFont(c,{fontFamily:'Arial'});
});
first.speakerNotes.textFrame.setText('Writing scores cover every one of the 384 new literary requests. Source attention is measured separately with the correct answer prefix supplied.');
for(const [sceneName,title] of [['rasters.json','What the role-trained writer drew']]){
const second=deck.slides.add();second.background.fill='#FFFFFF';
label(second,title,26,12,1010,32,true);
const scene=await load(sceneName);
const scale=Math.min(1008/scene.width,555/scene.height);
for(const it of scene.items){
  const x=26+it.x*scale,y=64+it.y*scale;
  if(it.type==='text') label(second,it.text,x,y,1020-x,it.size*scale,it.bold);
  else second.images.add({blob:await fs.readFile(path.join(root,'publication/ilm-nature/figures/library_roles',it.name)),
    contentType:'image/png',alt:it.name,fit:'contain',position:{left:x,top:y,width:it.w*scale,height:it.h*scale}});
}
}
for(const d of ['draft','output']) await fs.mkdir(path.join(stage,d),{recursive:true});
const candidatePath=path.join(stage,'draft/book.pptx'),finalPath=path.join(stage,'output/Writing_with_grammatical_roles.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:2,
  materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],verifyArtifactToolImport:true,
  integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','10096500,6210300'],receiptPath:path.join(stage,'presentation_validation.json')});
