import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src=path.join(root,'references/evidence/verse_canvas_v1');
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
text('Writing the missing ink in Tang verse',48,18,1410,66,43,true);
text('First line in pixels   /   five or seven output cells   /   6,362 unseen line pairs',53,90,1390,48,25,false,teal);
text('a   Does learning to identify visible ink help the writer?',50,170,1400,50,31,true);
text('Correct hidden glyphs (%)',62,240,750,34,19,false,teal);
const chart=slide.charts.add('bar',{barOptions:{direction:'column',grouping:'clustered'},
 position:{left:45,top:279,width:820,height:289},categories:['Ink','Ink + identity','Bytes + identity','Aligned counts'],
 hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:17}},
 series:['Blank second line','Two cells hidden'].map((name,i)=>({name,values:stats.series[i],fill:['#16686B','#C39963'][i]})),
 xAxis:{textStyle:{fontSize:17}},yAxis:{min:0,majorUnit:2,textStyle:{fontSize:17},majorGridlines:{fill:'#DCE5E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('One classifier for reading and writing',923,277,514,82,25,true,teal);
text('The reader sees a first line and the surviving ink of a second. During fitting, the identity loss asks the same output layer to name the canonical bitmap of each visible cell.',923,374,508,185,23);
text('b   From a blank line and from a damaged one',50,618,1400,50,31,true);
text('First reserved pair',58,703,560,48,24,true,teal);
text('First source-sensitive two-cell repair',757,703,683,48,24,true,teal);
for (const [left,labels,names] of [
 [58,['First line','Reference','Blank-line output'],['first-source','first-reference','first-blank']],
 [757,['First line','Damaged line','Completed line'],['repair-source','repair-template','repair-restored']]]){
 for(let i=0;i<3;i++){
  text(labels[i],left,775+70*i,238,39,21,i===2);
  await bitmap(names[i],left+250,762+70*i,420,51);
 }
}
text('c   Writing changes when the source changes',50,1025,1400,50,31,true);
table(stats.table,1108,208,[378,358,330,319]);
text('Each line keeps its original metre. Switching the typeface changes only the first line. Exchanging sources preserves length and keeps the same poet where possible.',55,1357,1390,83,23);
slide.speakerNotes.textFrame.setText('All image strips are actual input and output pixels. Blank-line generation starts without any target glyphs. The damaged-line task preserves all but two adjacent cells. Input identity is an auxiliary training signal. The independent image writer receives pixels and performs no OCR at inference.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Verse_canvas.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_136_verse_canvas.png'),new Uint8Array(await png.arrayBuffer()));
