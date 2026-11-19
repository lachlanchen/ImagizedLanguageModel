import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/grounded_proposition_20261119');
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
text('Which name belongs to which fact?',50,23,1410,68,43,true);
text('a   Writing both requested facts in the specified order',50,108,1410,52,29,true);
const chart=slide.charts.add('bar',{barOptions:{direction:'bar',grouping:'clustered'},
 position:{left:47,top:159,width:879,height:278},
 categories:['Practised combinations','Withheld combinations'],hasLegend:true,
 legend:{position:'bottom',textStyle:{fontSize:15}}, series:[
 {name:'Ordered fit',values:stats.series[0],fill:'#80ABAC'},
 {name:'Disrupted order',values:stats.series[1],fill:'#BBC8CA'},
 {name:'Varied wording',values:stats.series[2],fill:'#D39955'},
 {name:'Clause planner',values:stats.series[3],fill:teal}],
 xAxis:{textStyle:{fontSize:16}},yAxis:{min:0,max:100,textStyle:{fontSize:15},majorGridlines:{fill:'#DFE6E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('Complete statements (%)',950,185,475,65,25,true,teal);
text('Both names and both relations must be correct. The writer never practised composing lyricist with composer, or director with screenwriter, in the same answer.',950,267,477,198,23);
text('b   Transfer to questions in unfamiliar wording',50,467,1400,47,29,true);
table(stats.table,525,155,[290,256,303,292,244]);
text('The first three cohorts require a complete statement in an accepted wording. The open cohort requires the original reference answer.',57,698,1382,54,22);
text('c   A successful exchange in the first held-out passage',50,776,1400,48,28,true);
text('Original',55,850,180,41,21,true,teal);await image('first_board_0',260,832,1155,48);
text('Exchanged',55,911,190,40,21,true,teal);await image('first_board_2',260,894,1155,48);
text('d   A role error in a separately written film programme',50,982,1400,49,28,true);
text('Request',55,1052,210,40,21,true,teal);await image('compound_0_question',267,1036,1140,38);
text('Original note',55,1112,206,40,21,true,teal);await image('compound_0_passage',267,1096,1140,37);
text('Answer',55,1167,211,40,21,true,teal);await image('compound_0_answer',267,1150,1140,39);
text('Names exchanged',55,1230,215,40,21,true,teal);await image('compound_2_passage',267,1214,1140,37);
text('Answer',55,1287,211,40,21,true,teal);await image('compound_2_answer',267,1270,1140,39);
text('The original film note credits different people with the screenplay and direction. The writer assigns both roles to the same person in each answer.',57,1372,1380,68,22);
slide.speakerNotes.textFrame.setText('Percentages count accepted complete two-sentence answers for 312 practised and 136 withheld relation combinations. The notes are authored fictional passages. The lower panel shows the first fixed compound request and the same request after exchanging names. Consult the caption for exact counts and free-response limitations. Native chart and table are editable.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Grounded_proposition.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_92_grounded_proposition.png'),new Uint8Array(await png.arrayBuffer()));
