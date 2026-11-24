import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src=path.join(root,'references/evidence/story_mentions_v1');
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
text('Reading who “the latter” refers to',50,18,1410,70,46,true);
text('a   Two venues in one paragraph',55,112,1380,50,29,true);
text('Passage',55,184,205,41,24,true,teal);
await bitmap('paragraph',260,174,1170,69);
text('Question',55,268,205,41,24,true,teal);
await bitmap('question',260,260,670,52);
text('Name in prompt',55,365,345,40,23,true,teal);
text('Original writer',502,365,388,40,23,true,teal);
text('With reference reading',975,365,470,40,23,true,teal);
await bitmap('venue-1',55,428,300,49);
await bitmap('old-fee',502,428,398,49);
await bitmap('fee-1',975,428,413,49);
await bitmap('venue-2',55,509,300,49);
await bitmap('old-fee',502,509,398,49);
await bitmap('fee-2',975,509,413,49);
text('The garden costs 12 yuan. The tower costs 35 yuan. Only the pictured target name changes.',55,601,1400,48,24);
text('b   Which earlier name does the marked expression refer to?',55,695,1400,50,29,true);
const chart=slide.charts.add('bar',{barOptions:{direction:'column',grouping:'clustered'},
 position:{left:60,top:786,width:847,height:270},categories:['English (11)','Traditional Chinese (15)','Simplified Chinese (15)'],
 hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:15}},
 series:['Base','Visual order','Cue erased','Byte order'].map((name,i)=>({name,values:stats.series[i],fill:['#87969B','#157478','#C1A281','#485C79'][i]})),
 xAxis:{textStyle:{fontSize:15}},yAxis:{min:0,max:15,majorUnit:5,textStyle:{fontSize:16},majorGridlines:{fill:'#DCE5E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('Correct passages (count)',67,754,827,33,18,false,teal);
text('38 of 41',960,793,440,70,50,true,teal);
text('The visual reader and a fixed ordering rule each scored 38. The byte model scored 39. A passage could also refer to neither name.',960,876,463,166,23);
text('c   What transferred to other tests?',55,1084,1400,51,29,true);
table(stats.table,1150,205,[651,320,414]);
text('Changed-context Winograd pairs: the reader answered both variants on just 1 / 120. On Chinese CLUE, always rejecting the proposed link gives 193 / 304.',55,1388,1387,65,20);
slide.speakerNotes.textFrame.setText('Input and output use archived bitmap cells. The example is authored. Chart order cues and truth were annotated before fitting. Data and scorable denominators differ between the three rows. Joined utterances can recur.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Story_mentions.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_143_story_mentions.png'),new Uint8Array(await png.arrayBuffer()));
