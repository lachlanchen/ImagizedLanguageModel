import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile} from '@oai/artifact-tool';
const [root,stage,skill,python]=process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')));
const deck=Presentation.create({slideSize:{width:1280,height:720}});
const pub=path.join(root,'references/evidence/print_translation_v1');
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
 const sc=Math.min(1194/v.width,586/v.height);
 const x0=42,y0=95;
 for(const i of v.items){
  const y=y0+i.y*sc;
  if(i.type==='heading'){
   label(s,i.text,x0,y,sc*350,11*sc,true);
   label(s,i.meta,x0+sc*355,y+sc,sc*149,7.8*sc,false,'#49615b');
  }else{
   label(s,i.label,x0+8*sc,y+sc,sc*101,8.5*sc);
   s.images.add({blob:await fs.readFile(path.join(root,'publication/ilm-nature/figures/print_translation',i.name)),contentType:'image/png',alt:i.name,fit:'contain',position:{left:x0+i.x*sc,top:y,width:i.w*sc,height:i.h*sc}});
  }
 }
 s.speakerNotes.textFrame.setText(notes);
}
await plate('scene.json','How the same passage changes with the print face','All strips are the original PNGs. The normalized answer has identical pixels to the canonical answer. Case selection and full source are in the study archive.');
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
await plate('credits.json','Every credited canonical translation','All five original sources and full GPU answers in trial order. The judge accepted two replies that omit a detail. All five outputs reproduced on CPU.');
{
 const s=slide('Time to write the same 192 answers');
 const data=await read('inference.json');
 const categories=['Direct motor','Compiled motor'];
 const chart=s.charts.add('line',{position:{left:100,top:114,width:1060,height:460},categories,series:data.samples.map((r,i)=>({name:'Round '+(i+1),values:[r.direct,r.compiled].map(v=>Math.round(v*1e6)/1000)})),hasLegend:true,legend:{position:'bottom'},lineOptions:{smooth:false}});
 applyPresentationChartFont(chart,{fontFamily:'Arial'});
 label(s,'GPU wall time (ms), one RTX 4090 D. Timing excludes weight loading and table construction.',88,597,1130,18);
 s.speakerNotes.textFrame.setText('Five paired rounds, batch size 24, order alternated. Table compilation took '+data.compile_seconds+' seconds. Each arm reproduced the 192 canonical GPU images, lengths and stops.');
}
for(const name of ['draft','output']) await fs.mkdir(path.join(stage,name),{recursive:true});
const candidatePath=path.join(stage,'draft/figures.pptx'),finalPath=path.join(stage,'output/Glyph_stream_experiments.pptx');
await(await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:4,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[4],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000'],receiptPath:path.join(stage,'presentation_validation.json')});
