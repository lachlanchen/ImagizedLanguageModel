import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/inverse_role_writer_20261119');
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
text('Which participant does the pictured question ask for?',50,22,1430,65,40,true);
text('a   Human-written Chinese questions about unnamed participants',50,105,1430,52,29,true);
const chart=slide.charts.add('bar',{barOptions:{direction:'bar',grouping:'clustered'},position:{left:42,top:165,width:848,height:340},
 categories:['Randomized','Matched','Passage reader'],hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:18}},
 series:[{name:'Complete passage (%)',values:stats.series[0],fill:teal},{name:'Selected sentence (%)',values:stats.series[1],fill:'#A3B6C2'}],
 xAxis:{textStyle:{fontSize:20}},yAxis:{min:0,max:40,textStyle:{fontSize:18},majorGridlines:{fill:'#DBE6E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('Where the answer goes wrong',920,173,510,74,26,true,teal);
text('The matched writer missed 325 of 387 questions. In 301 errors the first glyph already differed from every accepted answer. In 314 it copied a contiguous passage fragment.',920,251,478,200,25);
text('b   Complete answers on historical and unfamiliar writing tasks',50,527,1440,52,29,true);
table(stats.table,593,195,[282,265,285,285,268]);
text('The four-name task changes who holds each of four objects. Neither fitted writer followed any object through all four rotations.',60,808,1380,71,23);
text('c   Asking opposite questions about one pictured passage',50,916,1440,55,29,true);
async function strip(name,x,y,space,height) {
 const [w,h]=stats.dimensions[name], scale=Math.min(space/w,height/h);
 await image(name,x,y,w*scale,h*scale);
}
text('Passage',63,989,185,45,24,true);await strip('bronze0_premise',235,979,565,52);
text('Returned name',914,990,435,44,24,true);
for(const i of [0,1]){
 const y=1059+68*i;
 await strip('bronze_query'+i,63,y,810,50);
 await strip('bronze_answer'+i,924,y,75,50);
 text(i===0?'Qin  (correct)':'Qin  (expected Chu)',1022,y+2,400,48,23);
}
text('d   A wrong name in ordinary text',50,1236,1430,48,29,true);
text('Human question',62,1302,300,45,22,true);text('Writer',792,1302,243,45,22,true);text('Reference',1120,1302,290,45,22,true);
await strip('human_query',62,1360,670,58);await strip('human_answer',792,1360,251,58);await strip('human_reference',1120,1360,288,58);
slide.speakerNotes.textFrame.setText('The raw answer to the human question names Matsu instead of Qi Jiguang. The upper historical example uses attested Qin and Chu images in a counterfactual passage.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Inverse_role_writer.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_85_inverse_role_writer.png'),new Uint8Array(await png.arrayBuffer()));
