import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src=path.join(root,'references/evidence/grounded_ink_editor_20261124');
const stats = JSON.parse(await fs.readFile(path.join(src,'figure.json')));

const deck = Presentation.create({slideSize:{width:1500,height:1470}});
const slide = deck.slides.add();
slide.background.fill='#FFFFFF';
const ink='#193849', teal='#16686b';
function text(value,left,top,width,height,size,bold=false,color=ink){
  const box=slide.shapes.add({geometry:'textbox',position:{left,top,width,height},fill:'none',line:{fill:'none',width:0}});
  box.text=value;
  box.text.style={typeface:'Arial',fontSize:size,bold,color,autoFit:'none'};
}
async function image(name,left,top,width,height){
  const blob=await fs.readFile(path.join(src,name+'.png'));
  slide.images.add({blob,contentType:'image/png',alt:name,fit:'contain',position:{left,top,width,height}});
}
function table(rows,top,height,widths){
  const t=slide.tables.add({rows:rows.length,columns:widths.length,left:55,top,width:1385,height,columnWidths:widths,values:rows});
  t.styleOptions={headerRow:true,bandedRows:false};t.borders.assign({style:'solid',fill:'#D7E4E5',width:1});
  for(let i=0;i<rows.length;i++)for(let j=0;j<widths.length;j++){
    const cell=t.getCell(i,j);cell.fill=i===0?'#E7F0F0':'#FFFFFF';
    cell.text.style={typeface:'Arial',fontSize:22,bold:i===0,color:j===1?teal:ink};
  }
}



async function bitmap(name,x,y,width,height){
 const data=await fs.readFile(path.join(src,name+'.png'));
 const w=data.readUInt32BE(16),h=data.readUInt32BE(20),scale=Math.min(width/w,height/h);
 await image(name,x,y,w*scale,h*scale);
}
text('Writing two answers from the ink on a page',50,16,1420,64,43,true);
text('Pictured questions and page   →   visual reader   →   two source regions   →   recurrent editor   →   written reply',55,90,1410,46,23,false,teal);
text('a   Exact two-clause responses, including generated ink and stopping (%)',50,154,1420,45,29,true);
const colors=['#B97854','#28767A','#AAB4BE'];
const captions=['Randomized order','Aligned regions','Supplied regions'];
const chart=slide.charts.add('bar',{barOptions:{direction:'column',grouping:'clustered'},
 position:{left:62,top:207,width:936,height:305},categories:['Journey · zh','Journey · en','Dream · zh','Dream · en'],
 hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:17}},
 series:captions.map((name,i)=>({name,values:stats.series[i],fill:colors[i]})),
 xAxis:{textStyle:{fontSize:17}},yAxis:{min:0,max:100,majorUnit:25,textStyle:{fontSize:17},majorGridlines:{fill:'#DFE6E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('What does the editor write?',1020,220,408,65,25,true,teal);
text('At each step it copies any contiguous span, draws a 32 × 32 bitmap, or stops. The connector is absent from the source. Both printed instruction formats were seen in training.',1020,290,399,206,22);
text('b   Excerpts with all four Chinese replies correct in both request orders and both worlds',50,535,1420,67,28,true);
table(stats.table,610,230,[343,255,260,259,268]);
text('c   The same two pictured questions on an original and an edited page',50,861,1415,50,29,true);
text('Q1',58,921,90,38,20,true,teal);await bitmap('question0',122,921,440,31);
text('Q2',635,921,90,38,20,true,teal);await bitmap('question1',709,921,560,31);
text('Page',60,980,250,40,21,true,teal);text('Source ink',322,980,310,40,21,true,teal);text('Q1 then Q2',671,980,330,40,21,true,teal);text('Q2 then Q1',1060,980,370,40,21,true,teal);
for(let w=0;w<2;w++){
 const y=1037+63*w;text(w?'Actions exchanged':'As printed',60,y,255,44,20,true);
 await bitmap('page'+w,322,y,328,34);await bitmap('written'+w+'_0',671,y,370,34);await bitmap('written'+w+'_1',1060,y,366,34);
}
text('d   Writing attested oracle and bronze forms of 言 in the requested order',50,1184,1410,55,29,true);
text('Adapted reader',65,1255,318,48,22,true,teal);await bitmap('yan_retrieved_0',410,1252,265,47);
text('Frozen parent',757,1255,286,48,22,true,teal);await bitmap('yan_parent_0',1070,1252,280,47);
text('Both lookups used pictured questions on the same historical plate. The original optical parent found both forms. Its editor copied their ink, drew the connector and final punctuation, and stopped in both requested orders.',60,1340,1375,90,21);
slide.speakerNotes.textFrame.setText('Measured final checkpoints on twelve excerpts per book. Strict scoring checks every emitted pixel and voluntary stopping. The chapter examples and historical outputs are saved images from actual inference. Paired controls and source editions are recorded in references/evidence/grounded_ink_editor_20261124.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Grounded_ink_editor.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_118_grounded_ink_editor.png'),new Uint8Array(await png.arrayBuffer()));
