import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/participant_margin_20261123');
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


text('Following the person when two credits change',50,23,1410,68,43,true);
text('a   Complete exchange tests with a glyph-by-glyph writer',50,105,1410,54,29,true);
const chart=slide.charts.add('bar',{barOptions:{direction:'bar',grouping:'clustered'},
 position:{left:48,top:161,width:895,height:322},categories:['Four questions per passage','Eight questions per credit'],
 hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:15}},series:[
 {name:'Previous',values:stats.series[0],fill:'#91A4AA'},
 {name:'Random labels',values:stats.series[1],fill:'#D8B588'},
 {name:'With margin',values:stats.series[2],fill:'#548F89'},
 {name:'Likelihood only',values:stats.series[3],fill:'#2C516D'}],
 xAxis:{textStyle:{fontSize:17}},yAxis:{min:0,max:100,textStyle:{fontSize:16},majorGridlines:{fill:'#DFE6E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('All answers exact (%)',974,184,454,46,25,true,teal);
text('Each question requires two stopped answers in different word orders. Likelihood alone passes 79 of 128 passage tests and 37 of 48 tests that also reverse clause order.',974,245,448,202,23);
text('b   Returning both word orders on the earlier questions',50,496,1410,48,28,true);
table(stats.table,549,165,[280,132,210,252,238,273]);
text('c   Following an exchange of composer and lyricist',50,752,1400,55,29,true);
async function row(label,name,y){
 text(label,57,y+3,395,40,22,true,teal);
 const blob=await fs.readFile(path.join(src,name+'.png'));
 const w=blob.readUInt32BE(16), h=blob.readUInt32BE(20);
 const height=Math.min(49,943*h/w);
 await image(name,479,y,w*height/h,height);
}
await row('Exchanged passage','song_page',819);
await row('Question','song_question',879);
await row('Previous answer','song_previous',939);
await row('With margin','song_matched',999);
text('d   A previously correct answer lost on a human question',50,1099,1400,55,29,true);
await row('Question','human_question',1164);
await row('Previous answer','human_previous',1225);
await row('With margin','human_matched',1286);
text('The question asks for the author of Jiuming qiyuan. The earlier answer correctly names Wu Woyao.',57,1382,1380,62,23);
slide.speakerNotes.textFrame.setText('All Chinese examples show the original physical input or emitted pixels. Panel d is the first loss in the original order of the human questions.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Participant_margin.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_95_participant_margin.png'),new Uint8Array(await png.arrayBuffer()));
