import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/english_request_writer_20261119');
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
text('Writing a dictionary answer from an English question',50,22,1450,65,39,true);
text('a   Complete answers to independently worded requests',50,113,1450,54,29,true);
const chart=slide.charts.add('bar',{barOptions:{direction:'bar',grouping:'clustered'},position:{left:55,top:172,width:825,height:248},
 categories:['Parent','Shuffled','Matched'],hasLegend:false,series:[{name:'Correct',values:stats.series[0],fill:teal}],
 xAxis:{textStyle:{fontSize:20}},yAxis:{min:0,max:96,textStyle:{fontSize:19},majorGridlines:{fill:'#DBE6E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('96 questions on 24 held-out cards',935,181,512,81,26,true,teal);
text('The local author received the intended word, fact and script. All questions count, including 26 with ambiguous or changed requests.',935,269,503,140,23);
text('b   Exact stopped answers',50,464,1450,55,29,true);table(stats.table,536,201,[170,273,263,205,205,269]);
text('c   First gained and first missed answer among clearer requests',50,794,1450,52,29,true);
for(const [key,top] of [['gained',873],['missed',1147]]){
 text(stats.examples[key].question,60,top,1385,89,24,true,teal);
 for(const [suffix,label,y] of [['_output','Output',top+96],['_reference','Reference',top+170]]){
  const name=key+suffix;const [w,h]=stats.dimensions[name],scale=Math.min(1.3,1140/w);
  text(label,62,y,204,47,23,true);await image(name,277,y,w*scale,h*scale);
 }
}
slide.speakerNotes.textFrame.setText('The change trains only the question encoder, its key projection and pooling. Source selection, historical rasters, writer and stop parameters remain fixed. The old English standard frame and equal-multiset crossed question each have 1,136 examples. Chinese uses the independent set. DuReader measures natural composition.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/English_request_writer.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_80_english_request_writer.png'),new Uint8Array(await png.arrayBuffer()));
