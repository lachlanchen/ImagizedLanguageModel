import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/span_choice_20261124');
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


text('Finding an answer among pictured spans',50,25,1410,65,43,true);
text('a   How often is the reference present?',50,111,1410,48,29,true);
const chart=slide.charts.add('bar',{barOptions:{direction:'bar',grouping:'clustered'},
 position:{left:48,top:168,width:860,height:333},categories:['1 span','2 spans','4 spans','8 spans'],
 hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:16}},series:[
 {name:'Emitted answer (fixed)',values:stats.series[0],fill:'#436778'},
 {name:'Reference available',values:stats.series[1],fill:'#74a69c'}],
 xAxis:{textStyle:{fontSize:18}},yAxis:{min:0,max:384,textStyle:{fontSize:16},majorGridlines:{fill:'#DFE6E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('384 human questions',963,185,469,58,28,true,teal);
text('Each of four readers proposes spans. Keeping their best eight raises the number of questions with an available reference from 300 to 346. The trained selector emitted 285 exact answers using one proposal per reader.',963,257,467,220,24);
text('b   Correct in both word orders and after every exchange',50,530,1420,48,29,true);
table(stats.table,593,212,[393,159,227,305,301]);
text('c   A recovered definition and an incomplete classification',50,848,1410,50,29,true);
text('Definition',365,922,540,44,24,true,teal);
text('Classification',908,922,525,44,24,true,teal);
async function pair(label,suffix,y){
 text(label,55,y+4,302,44,23,true);
 for(const [tag,x] of [['recovery',369],['miss',917]]){
   const name=tag+'_'+suffix;const blob=await fs.readFile(path.join(src,name+'.png'));
   const w=blob.readUInt32BE(16),h=blob.readUInt32BE(20);const size=Math.min(43,516*h/w);
   await image(name,x,y,w*size/h,size);
 }
}
await pair('Question','question',990);
await pair('Generic reader','generic',1064);
await pair('Pooled answer','answer',1138);
await pair('Human reference','reference',1212);
text('The definition is recovered exactly. The family name in the second reply is correct, but the annotated answer also gives the higher taxonomic ranks.',55,1322,1370,92,24);
slide.speakerNotes.textFrame.setText('Chinese strips preserve original pixels. Panel c uses the first recovered generic-reader error and the first absent reference in the fixed fresh order.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Span_choice.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_96_span_choice.png'),new Uint8Array(await png.arrayBuffer()));
