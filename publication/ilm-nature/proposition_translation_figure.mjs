import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation, applyPresentationChartFont} = await import(pathToFileURL(path.join(skill, 'container_tools/artifact_tool_utils.mjs')));
const base = path.join(root, 'references/evidence');
const study = path.join(base, 'proposition_translation_v1');
const read = async p => JSON.parse(await fs.readFile(p));
const deck = Presentation.create({slideSize:{width:1088, height:538}});
const ink = '#182526';
function label(slide, text, x, y, w, h, size, bold, color=ink) {
  const box = slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  box.text = text;
  box.text.style = {typeface:'Arial',fontSize:size,bold,color,autoFit:'none',insets:{left:0,right:0,top:0,bottom:0}};
}

function slide(title) {
  const s=deck.slides.add(); s.background.fill='#FFFFFF';
  label(s,title,26,16,1034,44,29,true); return s;
}
function chart(s,type,area,categories,series,extra={}) {
  const c=s.charts.add(type,{position:area,categories,series,hasLegend:true,
    legend:{position:'bottom',textStyle:{fontSize:13}},
    xAxis:{textStyle:{fontSize:13}},yAxis:{textStyle:{fontSize:13}},...extra});
  applyPresentationChartFont(c,{fontFamily:'Arial'}); return c;
}
for(const plate of await read(path.join(study,'plates.json'))) {
  const slide = deck.slides.add();
  slide.background.fill = '#FFFFFF';
  for(const r of plate.items) {
    const x=2*r.x,y=2*r.y;
    if(r.type==='text') {
      label(slide,r.value,x,y,1088-x,Math.min(538-y,2.8*r.size+4),2*r.size,r.bold,r.color);
      continue;
    }
    const position={left:x,top:y,width:2*r.w,height:2*r.h};
    if(r.type==='image') {
      slide.images.add({blob:await fs.readFile(path.join(study,'plates',r.path)),contentType:'image/png',alt:r.path,fit:'contain',position});
    } else {
      slide.shapes.add({geometry:'rect',position,fill:r.color,line:{fill:'none',width:0}});
    }
  }
  slide.speakerNotes.textFrame.setText('The two passive requests exchange agent and patient. Every printed output matches the retained GPU generation and a separate CPU replay from its original PNG.');
}
const data=await read(path.join(study,'chart_data.json'));
for (const [name, panels] of Object.entries(data)) {
  const s=slide(name.startsWith('Figure_191') ? 'Finding a translation and writing it' : 'Does varied wording transfer to natural translations?');
  panels.forEach((p,i)=>{
    const left=20+540*i;
    label(s,p.title,left,81,516,40,21,true);
    chart(s,'bar',{left,top:132,width:502,height:340},p.labels.map(x=>x.replace('\n',' ')),
      Object.entries(p.series).map(([name,values],j)=>({name,values,fill:['#526F84','#00766B','#A96941'][j]})),
      {barOptions:{direction:'column',grouping:'clustered'},yAxis:{min:0,max:p.top,majorUnit:p.ticks[1]}});
  });
  s.speakerNotes.textFrame.setText('Counts from fixed checkpoints. Recall is over 128 candidate translations. Meaning requires a stopped answer accepted in two opposite blind grading orders. The Qwen comparator receives text.');
}
for(const folder of ['draft','output']) await fs.mkdir(path.join(stage,folder),{recursive:true});
const candidatePath=path.join(stage,'draft/translation.pptx');
const finalPath=path.join(stage,'output/Visual_propositions.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:3,materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[2,3],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','10363200,5124450'],receiptPath:path.join(stage,'presentation_validation.json')});
