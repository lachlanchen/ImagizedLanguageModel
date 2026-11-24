import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src=path.join(root,'references/evidence/optical_passage_qa_v1');
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
text('Extracting answers from pictured passages',48,15,1420,63,42,true);
text('One English passage, two pictured questions and two intervals of original ink.',54,89,1400,52,25,false,teal);
text('a   Does the answer follow the question?',49,163,1400,50,30,true);
const chart=slide.charts.add('bar',{barOptions:{direction:'column',grouping:'clustered'},
 position:{left:48,top:223,width:878,height:320},categories:['Reserved articles','Development articles'],
 hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:18}},
 series:['Original questions','Exchanged questions'].map((name,i)=>({name,values:stats.series[i],fill:['#16686B','#AB896B'][i]})),
 xAxis:{textStyle:{fontSize:18}},yAxis:{min:0,max:30,majorUnit:10,textStyle:{fontSize:17},majorGridlines:{fill:'#DCE5E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('Both answers exact (%)',968,240,475,54,25,true,teal);
text('The passage stays fixed. On 484 reserved pairs the reader answered both questions on 108. Exchanging their assignments gave zero.',968,310,460,197,23);
text('b   What the reader returned',49,577,1410,50,30,true);
text('Pictured question (transcription)',56,647,915,45,23,true,teal);
text('Answer pixels',1005,647,430,45,23,true,teal);
text('In Portugal, which city was destroyed in the 1755 earthquake?',56,719,917,63,23);
await bitmap('examples/reserved-correct-1-answer',1009,717,403,52);
text('In which year did Brazel become independent?',56,797,916,61,23);
await bitmap('examples/reserved-correct-2-answer',1009,797,403,52);
text('Which Super Bowl did Roger Goodell speak about?',56,891,914,63,23);
await bitmap('examples/dev-miss-2-answer',1009,887,405,48);
text('The third answer is wrong. The source calls the 50th Super Bowl “spectacular”.',56,965,1390,68,22,false,teal);
text('c   Whole-answer scores on eligible passages',49,1062,1412,50,30,true);
table(stats.table,1132,175,[365,210,260,236,314]);
text('Exact answer uses SQuAD normalization. Exact ink preserves case and punctuation.',55,1354,1390,70,22);
slide.speakerNotes.textFrame.setText('First paired success in the reserved split and first failed pair in the development split, in source order. The source spelling Brazel is preserved. All output images were reproduced from the PNG inputs on CPU. These are contiguous copied intervals. Teacher supervision was excluded following the registered test.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Optical_passage_qa.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_127_optical_passage_qa.png'),new Uint8Array(await png.arrayBuffer()));
