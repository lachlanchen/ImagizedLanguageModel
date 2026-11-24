import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src=path.join(root,'references/evidence/word_memory_v1');
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
text('Can the writer recall a missing word image?',48,20,1420,62,43,true);
text('A frozen visual reader gains a learned distribution over 2,048 word images from earlier answers.',54,95,1390,68,24,false,teal);
text('a   Answer quality on 259 questions requiring new phrasing',49,170,1420,52,30,true);
const chart=slide.charts.add('bar',{barOptions:{direction:'column',grouping:'clustered'},
 position:{left:48,top:245,width:815,height:297},categories:['Original writer','Retrieval off','Retrieval on'],
 hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:18}},
 series:['Pictured input','Byte input'].map((name,i)=>({name,values:stats.series[i],fill:['#176F6E','#8499A8'][i]})),
 xAxis:{textStyle:{fontSize:18}},yAxis:{min:0,max:20,majorUnit:5,textStyle:{fontSize:17},majorGridlines:{fill:'#DCE5E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('Mean word F1 (%)',62,219,325,31,17,false,teal);
text('What did retrieval recognize?',904,246,520,56,26,true,teal);
text('For 108 hidden words, the correct image was absent from the inputs and available in memory. It ranked first in memory on 3 occasions and in the top eight on 17.',904,314,515,170,24);
text('b   Answers written from pictured passages and questions',49,567,1410,50,30,true);
text('Where did Mr. Preece work?',55,643,725,46,24,true);
await bitmap('grounded',780,640,660,39);
text('What was their deal?',55,730,725,46,24,true);
await bitmap('miss',780,726,660,39);
text('Reference answer: would toss a coin',785,778,628,44,20,false,teal);
text('Was guy familiar with Mr. Edmonstone writing style?',55,870,1110,52,24,true);
await bitmap('retrieval-change',1200,865,220,39);
text('Retrieval changed this reply from “No” to “yes”. The reply stayed “yes” after the question was exchanged.',55,937,1375,65,22,false,teal);
text('c   Whole answers and concealed source-absent words',49,1049,1410,55,30,true);
table(stats.table,1128,225,[463,319,317,286]);
text('Missing-word columns give literal image matches with every other reference word and the answer length supplied.',54,1395,1400,53,19);
slide.speakerNotes.textFrame.setText('Panels use matched single-seed image and byte writers on 147 unseen stories. The pictured examples were regenerated using the released model from PNG inputs. Stored and other refer to membership in the 2,048-image fitting inventory. The decoder-only control combines physical probabilities with the retrieval gate set to zero.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Word_memory.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_133_word_memory.png'),new Uint8Array(await png.arrayBuffer()));
