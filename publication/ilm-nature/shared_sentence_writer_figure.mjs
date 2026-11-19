import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/shared_sentence_writer_20261119');
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
text('Reading two answers from the same sentence',50,22,1450,65,41,true);
text('a   Both complete responses correct in 864 question pairs',50,108,1450,52,29,true);
const chart=slide.charts.add('bar',{barOptions:{direction:'bar',grouping:'clustered'},position:{left:48,top:176,width:828,height:306},
 categories:['Frozen writer','Randomized','Matched','Calibrated','Passage reader'],hasLegend:false,
 series:[{name:'Correct pairs',values:stats.series[0],fill:teal}],xAxis:{textStyle:{fontSize:21}},
 yAxis:{min:0,max:120,textStyle:{fontSize:19},majorGridlines:{fill:'#DBE6E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('Different questions, identical evidence',908,182,525,78,26,true,teal);
text('Each sentence contains both answer strings. It can still omit a fact needed to answer the question. Only one of 27 person-question pairs explicitly states both relations in the source review.',908,276,510,189,24);
text('b   Exact answers after learning when to copy a complete run',50,510,1450,50,29,true);
table(stats.table,566,223,[300,200,278,328,279]);
text('Who uses a sentence selected by glyph overlap. Roles uses 24 original situations with two questions and two name orders. Qwen received the same passages and questions as text.',62,804,1370,69,22);
text('c   Evidence and original responses from the calibrated visual writer',50,903,1450,52,29,true);
async function strip(name,x,y,w,h){
 const [iw,ih]=stats.dimensions[name],f=Math.min(w/iw,h/ih);await image(name,x,y,iw*f,ih*f);
}
for(const i of [0,1]){
 const y=982+i*244;
 text(i===0?'Earlier context needed':'Both roles stated',58,y+12,350,48,24,true,teal);
 text('Returned image',1128,y+83,310,35,20,true);
 await strip('pair'+i+'_passage',405,y-2,990,72);
 for (const j of [0,1]){
  await strip('pair'+i+'_q'+j,60,y+89+53*j,1000,45);await strip('pair'+i+'_a'+j,1130,y+119+47*j,283,43);
 }
 text(i===0?'The writer overshoots Wang Mang, then names a place.':'The donor and architect are both recovered exactly.',62,y+194,1030,42,21);
}
slide.speakerNotes.textFrame.setText('The first two person-question pairs in the fixed evaluation order. All Chinese strips show original evidence, questions and complete emitted glyphs.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Shared_sentence_writer.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_86_shared_sentence_writer.png'),new Uint8Array(await png.arrayBuffer()));
