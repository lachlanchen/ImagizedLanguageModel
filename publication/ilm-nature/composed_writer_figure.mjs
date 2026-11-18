import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/composed_writer_20261119');
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
text('Writing a Chinese answer from a pictured passage',50,19,1430,65,39,true);
text('a   Which script appears in the right column?',50,92,1440,48,29,true);
text('b   How many bronze seals remain?',50,503,1440,48,29,true);
for (const [group,locations] of [['script',[151,207,288,367,451]],['seals',[565,615,665,715,765]]]) {
  for (const [i,tag] of ['question','copy','finite','raster','reference'].entries()) {
    const name=group+'_'+tag,y=locations[i];
    text(['Question','Copy only','Training shapes','Pixel writer','Reference'][i],62,y,282,44,22,true,teal);
    const [w,h]=stats.dimensions[name];
    const scale=Math.min(1.01,1090/w);
    await image(name,351,y,w*scale,h*scale);
  }
}
text('c   Complete answers matching the reference image',50,822,1440,50,29,true);
table(stats.table,884,279,[326,290,278,274,217]);
text('d   Which missing glyph would the writer choose?',50,1191,1410,52,29,true);
const chart=slide.charts.add('bar',{barOptions:{direction:'bar',grouping:'clustered'},
 position:{left:50,top:1253,width:785,height:185},categories:['Training shapes','Pixel writer'],hasLegend:false,
 series:[{name:'Correct glyphs',values:stats.series[0],fill:teal}],xAxis:{textStyle:{fontSize:18}},
 yAxis:{min:0,textStyle:{fontSize:17},majorGridlines:{fill:'#DBE6E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('Correct preceding ink supplied',868,1255,552,48,23,true,teal);
text('The reference requires 1,525 missing cells. Selecting a familiar shape recovers 162. Adjusting its pixels recovers 57.',868,1310,540,130,22);
slide.speakerNotes.textFrame.setText('The first reference is small-seal script. The second answer is three seals. Panels a and b use two independently authored examples fixed before evaluating the DuReader answers. Panel c shows exact stopped counts. Panel d supplies the reference history and the instruction to write a glyph.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Composed_writer.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_75_composed_writer.png'),new Uint8Array(await png.arrayBuffer()));
