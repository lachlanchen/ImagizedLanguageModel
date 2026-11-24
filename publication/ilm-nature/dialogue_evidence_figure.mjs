import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src=path.join(root,'references/evidence/dialogue_evidence_v1');
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
text('Learning where to find an answer',48,15,1420,63,42,true);
text('Human readers mark the source words that support their answer. A visual writer can learn from those marks.',54,87,1400,65,25,false,teal);
text('a   Does evidence supervision improve reading across training seeds?',49,171,1400,50,30,true);
const chart=slide.charts.add('bar',{barOptions:{direction:'column',grouping:'clustered'},
 position:{left:48,top:237,width:850,height:320},categories:['Seed 1','Seed 2','Seed 3','Seed 4'],
 hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:18}},
 series:['321–448 words (47 stories)','Above 448 words (8 stories)'].map((name,i)=>({name,values:stats.series[i],fill:['#18766F','#577B91'][i]})),
 xAxis:{textStyle:{fontSize:19}},yAxis:{min:-0.5,max:1,majorUnit:0.5,textStyle:{fontSize:17},majorGridlines:{fill:'#DCE5E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('F1 change (percentage points)',946,247,485,58,25,true,teal);
text('Mean changes were +0.39 and +0.27 points. Both 95% intervals included zero after resampling training seeds and stories.',946,326,474,205,23);
text('b   Actual answers from the two visual writers',49,605,1400,52,30,true);
text('Pictured question (transcription)',54,683,590,46,23,true,teal);
text('Continued fit',708,683,280,46,23,true,teal);
text('Evidence loss',1080,683,348,46,23,true,teal);
text('Who took the tea first?',55,779,620,65,25);
await bitmap('gain-plain',713,781,263,41);
await bitmap('gain-human',1085,781,315,41);
text('who used a lot of Cockney vulgarities?',55,883,623,76,24);
await bitmap('loss-plain',713,887,303,41);
await bitmap('loss-human',1085,887,302,41);
text('Accepted answers include Shen Nong for the first question and Jane Humphreys for the second.',55,983,1390,70,22,false,teal);
text('c   A 4B text teacher on 83 questions requiring new wording',49,1081,1410,52,30,true);
table(stats.table,1151,165,[425,280,306,374]);
text('Changing question order improved the teacher. Both prompts fell below the 35% F1 required to use teacher answers in the student fits.',54,1362,1388,84,22);
slide.speakerNotes.textFrame.setText('Seed-paired gains compare human-evidence loss with ordinary continuation on 758 and 154 eligible turns. All models share the original visual parent. The two example rows are the first gained and lost any-reference matches in source order; answer strips are original predictions reproduced on CPU. Teacher exact counts accept any human answer. Valid quote means a contiguous supporting string found in the passage; relevance is a separate property.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Dialogue_evidence.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_130_dialogue_evidence.png'),new Uint8Array(await png.arrayBuffer()));
