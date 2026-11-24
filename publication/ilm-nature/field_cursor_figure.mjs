import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src=path.join(root,'references/evidence/field_cursor_v1');
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
text('Writing two facts under the right names',50,20,1400,62,43,true);
text('Chinese questions and answers as images   /   333,850 parameters trained from scratch',54,90,1395,42,25,false,teal);
text('a   The price and address of the same hotel',52,151,1400,48,31,true);
text('Question',55,214,355,40,23,true,teal);
await bitmap('exchange-query',330,210,1095,83);
text('Earlier image writer',55,329,360,40,23,true,teal);
await bitmap('exchange-before',330,317,1080,78);
text('Current-field writer',55,436,360,40,23,true,teal);
await bitmap('exchange-answer',330,427,1080,78);
text('The first writer returned both values but called 1,354 yuan an address. The smaller writer attaches each value to its property.',55,537,1390,63,22);
text('b   Held replies with both catalogue values',52,626,1370,46,31,true);
const chart=slide.charts.add('bar',{barOptions:{direction:'column',grouping:'clustered'},
 position:{left:42,top:694,width:800,height:268},categories:['Joint writer','Aligned + copy','Current field','Qwen3-8B'],
 hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:17}},
 series:['Clean values','Explicitly attached'].map((name,i)=>({name,values:stats.series[i],fill:['#9AB5BA','#126B70'][i]})),
 xAxis:{textStyle:{fontSize:16}},yAxis:{min:0,max:40,majorUnit:10,textStyle:{fontSize:17},majorGridlines:{fill:'#DCE5E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('A third fact in the question',889,687,520,42,23,true,teal);
await bitmap('failure-query',884,744,548,83);
await bitmap('failure-answer',884,844,548,73);
text('The user also asked for spending per person. The two-field writer omitted it.',889,927,531,72,20);
text('c   Does the answer follow the requested order?',52,1040,1400,46,31,true);
table(stats.table,1120,213,[785,300,300]);
text('Both orders of each field pair must pass. The last construction asks for the reverse of the order in which the fields appear.',55,1376,1390,66,22);
slide.speakerNotes.textFrame.setText('All example strips retain actual predicted pixels. The comparator read ordinary text. Its two literal-score misses include a correct municipal address and a requested third fact omitted from the two-field annotations.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Two_fact_figure.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_141_two_fact_figure.png'),new Uint8Array(await png.arrayBuffer()));
