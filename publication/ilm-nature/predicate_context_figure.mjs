import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/predicate_context_20261119');
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
text('Reading one glyph in two different senses',44,18,1410,64,40,true);
text('a   Complete source images and freely written answers',44,98,1410,51,29,true);
text('Source image',65,159,350,42,23,true,teal);
text('Answer image',484,159,440,42,23,true,teal);
text('What the answer says',965,159,480,42,23,true,teal);
for(const [i,[tag,description]] of [
 ['distance','Thirty li from Shouchun.'],
 ['leave','I cannot leave Zhongbao behind.'],
 ['wrong','Duan Jun “distances” instead of leaving.'],
 ['timber','The king of Wu sends men to fell trees.'],
 ['attack','Wu sends troops against Yue; wording is clipped.'],
 ['neural','The minister can therefore leave.']
].entries()){
 const y=215+i*55;
 for(const [suffix,x] of [['source',67],['answer',484]]){
  const name=tag+'_'+suffix;await image(name,x,y,stats.widths[name]*32,32);
 }
 text(description,965,y-1,478,52,21);
}
text('The phrase writer produced the first five answers. The neural contrast writer produced the last.',66,552,1380,50,21);
text('b   A correct predicate can appear in an incomplete translation',44,625,1410,48,29,true);
table(stats.table,696,248,[365,190,215,335,280]);
text('Twelve corpus clauses and eight authored clauses. Both clauses must be faithful for a pair to count. Broad edits use 1,024 separate corpus passages.',66,969,1376,68,21);
text('c   Neural choice between two glyphs after a supplied answer prefix',44,1062,1420,50,29,true);
const chart=slide.charts.add('bar',{position:{left:48,top:1139,width:785,height:275},categories:['Parent','Matched replay','First-glyph contrast'],hasLegend:true,
 barOptions:{direction:'bar',grouping:'clustered'},series:[{name:'Actual source',values:stats.series[0],fill:teal},{name:'Opposite-sense source',values:stats.series[1],fill:'#849FB1'}],
 xAxis:{textStyle:{typeface:'Arial',fontSize:17}},yAxis:{min:0,max:30,majorUnit:5,textStyle:{typeface:'Arial',fontSize:17},majorGridlines:{fill:'#DBE6E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('Does the source change the choice?',870,1141,545,53,24,true,teal);
text('Of 30 held-out clauses, the contrast writer favoured the intended first glyph on 16 with the correct source and on 15 after exchanging the source. The answer prefix was identical in each comparison.',870,1213,535,204,22);
slide.speakerNotes.textFrame.setText('Each depicted answer is a complete saved physical stream. The third answer uses the distance sense in an intransitive departure context. Panel c measures conditional probabilities, separately from autonomous text quality.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Predicate_context.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_64_predicate_context.png'),new Uint8Array(await png.arrayBuffer()));
