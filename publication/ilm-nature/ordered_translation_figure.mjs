import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation, applyPresentationChartFont} = await import(pathToFileURL(path.join(skill, 'container_tools/artifact_tool_utils.mjs')));
const base = path.join(root, 'references/evidence');
const study = path.join(base, 'ordered_translation_v1');
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
const quality=slide('What shorter visual sequences preserve');
const colors=['#006B66','#9B6D46','#577189'];
const fits=await Promise.all([1,4].map(k=>read(path.join(root,'models/ordered_translation',String(k),'fit.json'))));
label(quality,'Development loss (nats per glyph)',25,85,495,35,21,true);
chart(quality,'line',{left:22,top:126,width:503,height:342},fits[0].epochs.map(r=>String(r.epoch)),
  fits.map((fit,i)=>({name:'Stride '+fit.stride,values:fit.epochs.map(r=>Number(r.dev.mean_nll.toFixed(7))),fill:colors[i]})));
label(quality,'Meaning preserved (/32)',568,85,480,35,21,true);
const grades=(await read(path.join(study,'judgments.json'))).groups;
chart(quality,'bar',{left:559,top:126,width:500,height:342},['To English','To Chinese'],
  ['stride1','stride4','qwen'].map((m,i)=>({name:['Stride 1','Stride 4','Qwen 8B Q4'][i],values:['en-sample','zh-sample'].map(k=>grades[k][m]),fill:colors[i]})),
  {barOptions:{direction:'column',grouping:'clustered'},yAxis:{min:0,max:32,majorUnit:8}});
quality.speakerNotes.textFrame.setText('Same parameters and data for strides 1 and 4. Twelve actual epochs on one RTX 4090 D. Qwen receives transcribed text. A separate Qwen3-VL judge assesses four systems under both candidate orders.');
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
  slide.speakerNotes.textFrame.setText('Original model output pixels, adapted from Tatoeba contributors Hybrid, GlossaMatik, CK and wzhd (CC BY 2.0 FR). Attested images from hanziyuan.net. The first failure loses dolphin trainer. The first exact reference omits the English adjective good.');
}
const transfer=slide('Does training on news improve held translations?');
const domains=(await read(path.join(study,'news-judgments.json'))).groups;
chart(transfer,'bar',{left:25,top:100,width:620,height:355},['Conversation','News'],
  ['parent','replay','news','qwen'].map((m,i)=>({name:['Base','Replay','News mix','Qwen 8B Q4'][i],values:['conversation','news'].map(k=>domains[k][m].meaning),fill:['#829087','#9B6D46','#006B66','#577189'][i]})),
  {barOptions:{direction:'column',grouping:'clustered'},yAxis:{min:0,max:32,majorUnit:8}});
label(transfer,'Model translations accepted in both orders',30,64,720,31,19,true);
const rows=[['Held set','Human','Foil'],...['conversation','news'].map(k=>[k,String(domains[k].reference.meaning),String(domains[k].foil.meaning)])];
const table=transfer.tables.add({rows:3,columns:3,left:682,top:149,width:377,height:147,columnWidths:[161,108,108],values:rows});
table.borders.assign({style:'solid',fill:'#D4DFD9',width:1});
for(let i=0;i<3;i++) for(let j=0;j<3;j++){let cell=table.getCell(i,j);cell.fill=i?'#FFFFFF':'#E9F1ED';cell.text.style={typeface:'Arial',fontSize:17,bold:i===0,color:ink};}
label(transfer,'Calibration per 32 requests',680,96,378,36,20,true);
transfer.speakerNotes.textFrame.setText('Sixteen families from each domain, both translation directions. The two arms continue the same stride-one visual checkpoint for four passes. News sentences are longer. Chart values count stopped outputs accepted in both judging orders.');
for(const folder of ['draft','output']) await fs.mkdir(path.join(stage,folder),{recursive:true});
const candidatePath=path.join(stage,'draft/translation.pptx');
const finalPath=path.join(stage,'output/Ordered_visual_translation.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:3,materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1,3],requiredNativeTableOwnerSlides:[3],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','10363200,5124450','--require-native-table-slide','3'],receiptPath:path.join(stage,'presentation_validation.json')});
