import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile} from '@oai/artifact-tool';
const [root,stage,skill,python]=process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')));
const deck=Presentation.create({slideSize:{width:1280,height:720}});
const pub=path.join(root,'references/evidence/optical_span_rollout_v1');
const read=async x=>JSON.parse(await fs.readFile(path.join(pub,x)));
function label(s,text,x,y,w,size,bold=false,color='#19332f'){
 const shape=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:size*1.48},fill:'none',line:{fill:'none',width:0}});
 shape.text=text;
 shape.text.style={typeface:'Arial',fontSize:size,bold,color,autoFit:'none',insets:{left:0,right:0,top:0,bottom:0}};
}
function slide(heading){
 const s=deck.slides.add();s.background.fill='#ffffff';label(s,heading,36,22,1208,32,true);return s;
}
async function plate(name,title,notes){
 const v=await read(name),s=slide(title);
 const sc=Math.min(1170/v.width,586/v.height);
 const x0=42,y0=95;
 for(const i of v.items){
  const y=y0+i.y*sc;
  if(i.type==='heading'){
   label(s,i.text,x0,y,sc*300,11*sc,true);
   label(s,i.meta,x0+sc*300,y+sc,sc*192,7.8*sc,false,'#49615b');
  }else{
   label(s,i.label,x0+8*sc,y+sc,sc*78,8.5*sc);
   s.images.add({blob:await fs.readFile(path.join(root,'publication/ilm-nature/figures/span_rollout',i.name)),contentType:'image/png',alt:i.name,fit:'contain',position:{left:x0+i.x*sc,top:y,width:i.w*sc,height:i.h*sc}});
  }
 }
 s.speakerNotes.textFrame.setText(notes);
}
await plate('plate.json','What did the model actually write?','The first credited and first rejected span translations in the frozen case order. Every image is the raw input or model answer.');
{
 const s=slide('Which passages kept their meaning?');
 const v=await read('blocks.json');
 label(s,'Green cells earned credit in both source-only readings. Each column is one passage within the book.',36,75,1200,20);
 for(let b=0;b<v.blocks.length;b++){
  const r=v.blocks[b], gx=b%4, gy=Math.floor(b/4), x=104+gx*291,y=170+gy*238;
  label(s,r.work,x,y-36,287,16,true);
  if(gx===0) label(s,r.language==='en'?'To English':'To Chinese',12,y-62,140,16,true);
  if(gx===0) Object.values(v.methods).forEach((k,i)=>label(s,k,10,y+i*29+5,94,13));
  const t=s.tables.add({rows:5,columns:24,left:x,top:y,width:279,height:145,columnWidths:Array(24).fill(279/24),values:r.values.map(row=>row.map(()=>''))});
  t.cells.block({row:0,column:0,rowCount:5,columnCount:24}).assign({margins:{left:0,right:0,top:0,bottom:0},textStyle:{typeface:'Arial',fontSize:2}});
  t.borders.assign({fill:'#ffffff',width:0.6,style:'solid'});
  for(let i=0;i<5;i++) for(let j=0;j<24;j++) t.getCell(i,j).fill=r.values[i][j]?'#156c62':'#e3e8e6';
  label(s,'1',x,y+147,25,12);label(s,'12',x+127,y+147,25,12);label(s,'24',x+262,y+147,26,12);
 }
 s.speakerNotes.textFrame.setText('All 192 distinct directions. Each editable table has five rows and 24 columns; paired sources have matching column positions. Full unrounded outcomes and both vote orders are supplied.');
}
{
 const s=slide('The effect of reading and writing through images');
 const data=await read('interface-chart.json');
 const chart=s.charts.add('bar',{position:{left:70,top:110,width:1130,height:465},categories:data.labels,
   series:[{name:'To English',values:data.series.en},{name:'To Chinese',values:data.series.zh}],
   dataLabels:{showValue:true,position:'outEnd',textStyle:{typeface:'Arial',fontSize:20,fill:'#19332f'}},
   hasLegend:true,legend:{position:'bottom'},barOptions:{direction:'column',grouping:'clustered'}});
 applyPresentationChartFont(chart,{fontFamily:'Arial'});
 label(s,'Complete meanings in 96 requests per direction, assessed with the same two masked readings.',80,615,1140,19);
 s.speakerNotes.textFrame.setText('The contextual decoder is shared across the three arms within each language. The first two alternatives output text. Only the third draws and reads back its own glyphs.');
}
await plate('origin-plate.json','Can the writer answer word-origin questions?','The requested order contains actual Hanziyuan specimens. Both response strips are original model pixels.');
for(const name of ['draft','output']) await fs.mkdir(path.join(stage,name),{recursive:true});
const candidatePath=path.join(stage,'draft/figures.pptx'),finalPath=path.join(stage,'output/Autonomous_span_translation.pptx');
await(await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:4,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[3],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000'],receiptPath:path.join(stage,'presentation_validation.json')});
