import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/pointing_causal_writer_20261119');
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
text('Reading a fact and composing a historical caption',50,26,1410,65,41,true);
text('a   Who did what in a short pictured passage?',50,103,1375,53,29,true);
const chart=slide.charts.add('bar',{barOptions:{direction:'bar',grouping:'clustered'},
 position:{left:47,top:168,width:858,height:275},
 categories:['First 16 scenes','New 16 scenes'],hasLegend:true,
 legend:{position:'bottom',textStyle:{fontSize:15}},
 series:[{name:'Original fit',values:stats.series[0],fill:'#9FAFB1'},
         {name:'Random excerpts',values:stats.series[1],fill:'#D59950'},
         {name:'Answer sentences',values:stats.series[2],fill:teal}],
 xAxis:{textStyle:{fontSize:17}},yAxis:{min:0,max:64,textStyle:{fontSize:16},majorGridlines:{fill:'#DFE6E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('Exact names out of 64',932,168,499,52,24,true,teal);
text('Who drew the map?',931,235,344,45,22);
await image('role_0_question',928,275,330,43); await image('role_0_expected',1314,275,85,43);
text('Who needed directions?',931,326,452,42,22);
await image('role_1_question',928,365,330,43); await image('role_1_expected',1314,365,85,43);
text('b   Two questions about the same source',50,492,1400,51,29,true);
table(stats.table,560,169,[285,210,210,205,245,230]);
text('CMRC and DRCD count passages with both facts correct. Original and reworded count both requested historical orders. New ink counts individual exact answers.',57,760,1384,75,22);
text('c   The oracle and bronze forms of 言 in two requested orders',50,878,1400,54,29,true);
text('Source plate',57,967,185,43,23,true,teal);
await image('yan_2_passage',260,945,1150,58);
text('Pictured instruction',55,1055,840,44,23,true,teal);
text('Returned caption',905,1055,531,44,23,true,teal);
await image('yan_2_question',57,1124,793,48);await image('yan_2_expected',903,1124,530,48);
await image('yan_3_question',57,1222,793,48);await image('yan_3_expected',903,1222,530,48);
text('The same plate supplies both historical images. Each answer names the requested stages and reproduces their source shapes in that order.',57,1345,1380,79,23);
slide.speakerNotes.textFrame.setText('Bars show counts for three selection fits on the same held role questions and sixteen newly written scenarios. The two map queries and both orders for 言 use original output pixels. The 言 card belongs to the separate demonstration from a training family. The captions table reports held-out families. No historical exemplar was drawn by an illustrator.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Pointing_causal_writer.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_91_pointing_causal_writer.png'),new Uint8Array(await png.arrayBuffer()));
