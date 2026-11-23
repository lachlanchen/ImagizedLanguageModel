import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src=path.join(root,'references/evidence/book_span_distillation_20261124');
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
text('Distilling a visual reader into 64,666 parameters',50,20,1420,65,41,true);
text('a   Exact answers found anywhere on the pictured passage (%)',50,100,1420,52,30,true);
const colors=['#A3AEB8','#BB734A','#347C80'];
const captions=['Earlier small reader','Exchanged teacher','Matched teacher'];
const chart=slide.charts.add('bar',{barOptions:{direction:'column',grouping:'clustered'},
 position:{left:62,top:159,width:943,height:330},categories:['Harry Potter · zh','Harry Potter · en','Three-Body · zh','Three-Body · en'],
 hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:17}},
 series:captions.map((name,i)=>({name,values:stats.series[i],fill:colors[i]})),
 xAxis:{textStyle:{fontSize:17}},yAxis:{min:0,max:40,majorUnit:10,textStyle:{fontSize:17},majorGridlines:{fill:'#DFE6E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('What was exchanged?',1040,190,360,60,28,true,teal);
text('Both students saw the same pages and human answers. The control received the other question’s teacher probabilities for the same page.',1040,250,360,217,23);
text('b   Exact source replies without supplied candidates',50,505,1410,52,29,true);
table(stats.table,576,300,[368,248,252,249,268]);
text('c   Question, reference and emitted ink',50,905,1400,54,29,true);
text('Passage',60,975,245,43,21,true,teal);
text('Pictured question',325,975,560,43,21,true,teal);
text('Source answer',931,975,232,43,21,true,teal);
text('Reader output',1190,975,260,43,21,true,teal);
for(let i=0;i<stats.strips.length;i++){
 const r=stats.strips[i],y=1030+54*i;
 text(r.work,60,y,255,49,21,true);
 await bitmap(r.base+'_q',325,y,580,36);
 await bitmap(r.base+'_gold',931,y,230,36);
 await bitmap(r.base+'_out',1190,y,250,36);
}
text('First reserved passage per work, followed by the first complete Chinese success when available. The matched student can select any strip up to eight glyphs wide.',60,1382,1370,67,20);
slide.speakerNotes.textFrame.setText('Each question can select any contiguous strip up to eight glyphs. The chart separates the two novels; the table compares all freely selected replies. Neither student uses the teacher at inference.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Book_span_distillation.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_113_book_span_distillation.png'),new Uint8Array(await png.arrayBuffer()));
