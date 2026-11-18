import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/grounded_glyph_writer_20261119');
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
text('Does the pictured question change the answer?',50,20,1450,65,40,true);
text('a   Both questions answered correctly with two supplied choices',50,110,1450,54,29,true);
const chart=slide.charts.add('bar',{barOptions:{direction:'bar',grouping:'clustered'},
 position:{left:53,top:194,width:830,height:267},categories:['Matched writer','Pairs broken','Direct overlap','Via passages'],hasLegend:false,
 series:[{name:'Both correct in both orders',values:stats.series[0],fill:teal}],
 xAxis:{textStyle:{fontSize:20}},yAxis:{min:0,textStyle:{fontSize:19},majorGridlines:{fill:'#DBE6E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('285 held-out pairs',925,194,509,47,26,true,teal);
text('Both references are supplied. The overlap controls compare sets of glyphs. Each pair must pass with both questions in both passage orders.',925,267,494,185,24);
text('b   Generation from an empty answer',50,499,1450,55,29,true);
table(stats.table,573,218,[375,230,230,230,320]);
text('c   The matched writer copied both pictured premises',50,849,1450,50,29,true);
text('Which script appears on the right?',62,928,1350,45,24,true,teal);
text('How many bronze seals remain?',62,1191,1350,48,24,true,teal);
for(const [name,label,y] of [['script_output','Output',1003],['script_reference','Reference',1101],
 ['seals_output','Output',1258],['seals_reference','Reference',1338]]){
 text(label,62,y,222,52,24,true);
 const [w,h]=stats.dimensions[name],scale=Math.min(1.08,1110/w);
 await image(name,293,y,w*scale,h*scale);
}
slide.speakerNotes.textFrame.setText('In panel b, the first three accuracy columns count cases out of 571. The final column has 300 questions requiring content ink absent from the input. Both response strips contain the complete supplied passage. The references request small-seal script and three bronze seals.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Grounded_glyph_writer.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_77_grounded_glyph_writer.png'),new Uint8Array(await png.arrayBuffer()));
