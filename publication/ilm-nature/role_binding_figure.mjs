import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/role_binding_20261123');
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
text('Reading a Chinese fact from glyph images',50,23,1410,68,43,true);
text('a   Which parts of the answer were recovered?',50,108,1410,52,29,true);
const chart=slide.charts.add('bar',{barOptions:{direction:'bar',grouping:'clustered'},
 position:{left:48,top:157,width:895,height:325},
 categories:['Human questions','Programme notes','DuIE passages'],hasLegend:true,
 legend:{position:'bottom',textStyle:{fontSize:16}},series:[
 {name:'Subject',values:stats.series[0],fill:'#B6C9C9'},
 {name:'Relation',values:stats.series[1],fill:'#77A9A8'},
 {name:'Object',values:stats.series[2],fill:'#D19B59'},
 {name:'Complete statement',values:stats.series[3],fill:teal}],
 xAxis:{textStyle:{fontSize:16}},yAxis:{min:0,max:100,textStyle:{fontSize:15},majorGridlines:{fill:'#DFE6E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('Correct answers (%)',960,190,465,50,25,true,teal);
text('The reader predicts two spans of ink and the relation between them. A complete statement requires the correct subject, relation and object.',960,255,457,173,23);
text('b   Complete answers from two ways of writing the same fact',50,497,1400,45,28,true);
table(stats.table,555,162,[310,160,252,326,337]);
text('The last column requires the autoregressive writer to produce both a subject-first and an object-first answer for the same input.',58,737,1380,55,22);
text('c   Film credits before and after exchanging two names',50,809,1400,47,28,true);
text('Original note',55,873,229,38,21,true,teal);await image('film_0_passage',279,860,1140,38);
text('Names exchanged',55,927,229,38,21,true,teal);await image('film_2_passage',279,914,1140,38);
text('Director question',55,984,230,38,21,true,teal);await image('film_0_question',281,971,1130,38);
text('Screenwriter question',55,1037,232,38,20,true,teal);await image('film_1_question',281,1024,1130,38);
text('Original credits',55,1102,235,35,21,true,teal);
await image('film_0_answer',291,1086,547,41);await image('film_1_answer',865,1086,550,41);
text('Exchanged credits',55,1161,231,35,21,true,teal);
await image('film_2_answer',278,1145,710,42);await image('film_3_answer',995,1145,424,42);
text('Object-first replies on the original note',55,1234,1255,38,23,true,teal);
await image('free_film_0',85,1282,623,51);await image('free_film_1',758,1282,622,51);
text('The exchanged director span extends across part of the passage. All answer strips show the emitted pixels.',57,1381,1380,60,23);
slide.speakerNotes.textFrame.setText('The chart shows accuracy by field, scored on all inputs. The native table reports counts. Film credits are shown left to right as director and screenwriter. The two outputs in the bottom row were written one glyph at a time.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Role_binding.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_93_role_binding.png'),new Uint8Array(await png.arrayBuffer()));
