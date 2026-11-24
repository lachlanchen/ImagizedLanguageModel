import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src=path.join(root,'references/evidence/everyday_gloss_v1');
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
text('Can the reader follow a classroom question?',50,16,1420,64,43,true);
text('An unchanged writer draws one English word for the requested part of a classical Chinese clause.',55,89,1400,54,23,false,teal);
text('a   Ordinary questions on 539 previously held passages',50,154,1410,52,29,true);
const colors=['#8393A1','#276E73','#AD694B'];
const chart=slide.charts.add('bar',{barOptions:{direction:'column',grouping:'clustered'},
 position:{left:50,top:218,width:882,height:288},categories:['English pair','Chinese pair','All four questions'],
 hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:18}},
 series:['Original reader','Broader questions','Exchanged answers'].map((name,i)=>({name,values:stats.series[i],fill:colors[i]})),
 xAxis:{textStyle:{fontSize:17}},yAxis:{min:0,max:300,majorUnit:100,textStyle:{fontSize:17},majorGridlines:{fill:'#DCE5E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('What counts as a pass?',986,231,465,48,25,true,teal);
text('Both questions must select the intended Chinese characters and produce accepted English images. The four-question test requires the same passage to pass in both languages.',986,292,443,194,22);
text('b   Locating the word and writing its meaning',50,530,1400,51,29,true);
table(stats.table,599,212,[340,200,215,329,301]);
text('The ceiling supplies both source positions to the frozen writer. The two word-order rows partition the new books.',57,838,1390,65,21);
text('c   English answers drawn from two previously unseen passages',50,943,1400,50,29,true);
for(const [label,x] of [['Passage',58],['Action',420],['Object',770],['Reading',1080]])text(label,x,1032,340,45,22,true,teal);
for(const [row,id,note] of [[0,'26','Both answers correct'],[1,'25','Object misspelled']]) {
 const y=1110+row*110;
 for(const [suffix,x] of [['',54],['-q0',415],['-q1',765]]) await bitmap('samples/page'+id+suffix,x,y,285,58);
 text(note,1080,y,362,65,22);
}
text('The second object denotes a box or casket. Its actual output reads “capper”.',57,1375,1370,55,22);
slide.speakerNotes.textFrame.setText('Panel a shows absolute counts on the same 539 classical passages. Panel b uses four classroom questions. The writer ceiling is measured at the supplied locations. Panel c contains unedited image output from the English classroom questions. Each answer was replayed independently on CPU.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Everyday_gloss.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_121_everyday_gloss.png'),new Uint8Array(await png.arrayBuffer()));
