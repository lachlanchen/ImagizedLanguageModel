import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src=path.join(root,'references/evidence/questioned_events_v1');
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
text('Two actions within one Chinese line',48,15,1420,63,42,true);
text('The question names an action. The reader locates its verb and object, then draws the object in English.',54,89,1415,59,23,false,teal);
text('a   Reading every event on 971 held lines',49,164,1410,50,30,true);
const chart=slide.charts.add('bar',{barOptions:{direction:'column',grouping:'clustered'},
 position:{left:48,top:221,width:920,height:306},categories:['All verb and object positions','All positions and written words'],
 hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:16}},
 series:['Frozen','Exchanged objects','Next cell','Learned'].map((name,i)=>({name,values:stats.series[i],fill:['#9DAAB3','#BA9472','#577D9A','#16686B'][i]})),
 xAxis:{textStyle:{fontSize:16}},yAxis:{min:0,max:350,majorUnit:100,textStyle:{fontSize:16},majorGridlines:{fill:'#DCE5E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('Why score positions?',985,244,450,56,24,true,teal);
text('A familiar word alone earns no credit. Every action must select both annotated cells and produce a stopped answer in two question frames.',985,307,443,190,22);
text('b   The same pixels support two answers',49,557,1425,50,30,true);
for(const [label,x] of [['Source line',53],['Requested action',550],['Actual output',1010]])text(label,x,632,460,46,22,true,teal);
await bitmap('moon-page',54,729,445,96);
text('raise',554,705,375,55,27);text('invite',554,800,375,55,27);
await bitmap('moon-raise',1014,705,367,56);await bitmap('moon-invite',1014,800,367,56);
text('The speaker raises a cup and invites the moon. This previously seen line tests whether changing the question can move the pointer within one line.',55,902,1380,82,22);
text('c   Complete answers under changes of wording and source',49,1012,1415,54,29,true);
table(stats.table,1081,255,[383,129,260,277,336]);
text('The unseen-question subset excludes all previously rendered full prompts for every action on that page.',56,1371,1390,67,21);
slide.speakerNotes.textFrame.setText('The chart compares complete verb-object localization and complete localization plus English word images on the pinned Kyoto benchmark. Next cell uses the learned predicate and selects the immediately following cell. It scores better in the held-work subset. The poem is a known mechanistic contrast, not a held-source success. Natural-question gains include prior question overlap. The registered transfer criterion fails.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Questioned_events.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_126_questioned_events.png'),new Uint8Array(await png.arrayBuffer()));
