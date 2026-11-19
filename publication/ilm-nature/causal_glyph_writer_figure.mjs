import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/causal_glyph_writer_20261119');
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
text('When an answer needs a glyph missing from the page',50,26,1410,65,41,true);
text('a   Prediction with the correct answer so far',50,111,1375,56,29,true);
const chart=slide.charts.add('bar',{barOptions:{direction:'bar',grouping:'clustered'},
 position:{left:47,top:178,width:897,height:310},
 categories:['In the prompt','Absent from the prompt','First absent glyph'],hasLegend:true,
 legend:{position:'bottom',textStyle:{fontSize:16}},
 series:[{name:'Shuffled pairs',values:stats.series[0],fill:'#A9BABD'},
         {name:'Matched pairs',values:stats.series[1],fill:teal}],
 xAxis:{textStyle:{fontSize:17}},yAxis:{min:0,max:100,textStyle:{fontSize:16},majorGridlines:{fill:'#DFE6E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('Correct next glyph (%)',970,186,462,59,26,true,teal);
text(stats.note,970,264,449,200,25);
text('b   Complete answers from the model’s own image history',50,510,1400,54,29,true);
table(stats.table,570,244,[380,245,238,226,296]);
text('Every score requires the exact reference image sequence and an explicit stop. Open questions share the same 571 development cases.',57,835,1384,65,22);
text('c   Two questions about the same map',50,937,1380,52,29,true);
text('Passage',57,1016,150,41,23,true,teal);
await image('map_passage',243,994,1070,58);
text('Pictured question',55,1103,547,45,23,true,teal);
text('Returned glyphs',660,1103,772,45,23,true,teal);
await image('map_q0',57,1162,513,54);
await image('map_a0',653,1162,792,54);
await image('map_q1',57,1252,513,54);
await image('map_a1',653,1252,792,54);
text('The first reply identifies the mapmaker. The second repeats that fact when asked who needed directions.',57,1380,1400,62,22);
slide.speakerNotes.textFrame.setText('The example uses the first two role questions in their original order. Refer to the caption for the English translation and exact denominators. Chart data and the counts table are editable.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Causal_glyph_writer.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_90_causal_glyph_writer.png'),new Uint8Array(await png.arrayBuffer()));
