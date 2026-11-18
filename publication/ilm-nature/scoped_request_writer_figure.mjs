import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/scoped_request_writer_20261119');
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
text('Reading which clause requests a historical form',50,22,1450,65,39,true);
text('a   All four answers correct for a headword and layout',50,105,1450,55,29,true);
const chart=slide.charts.add('bar',{barOptions:{direction:'bar',grouping:'clustered'},position:{left:41,top:175,width:808,height:265},
 categories:['Local','Global, no order','Global, ordered'],hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:16}},
 series:[{name:'Familiar cues',values:stats.series[0],fill:teal},{name:'New cues',values:stats.series[1],fill:'#A26040'}],
 xAxis:{textStyle:{fontSize:20}},yAxis:{min:0,textStyle:{fontSize:19},majorGridlines:{fill:'#DBE6E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('Same nine-cell neighborhoods',907,175,530,73,26,true,teal);
text('Each group of four questions contains the same set of local glyph windows, with identical counts. Which clause governs a field changes the answer.',907,255,510,166,24);
text('b   Complete groups and independent questions',50,475,1450,55,29,true);
table(stats.table,544,200,[239,248,208,212,211,267]);
text('¹ Complete groups out of 1,136 per construction. DRCD has 1,042 pairs.',60,758,1400,45,20);
text('c   Same card and local windows, different requests',50,826,1450,53,29,true);
for(const [key,top] of [['0',897],['3',1175]]){
 const m=stats.examples[key],field=m.field==='本义'?'original meaning':'Mandarin pronunciation';
 text('Requested: '+field+' and '+m.stage+' form of 耳',60,top,1392,49,25,true,teal);
 for(const [model,label,offset] of [['local','Local',65],['ordered','Global, ordered',127],['reference','Reference',189]]){
  const name=model+key,[w,h]=stats.dimensions[name],scale=Math.min(1.22,1055/w);
  text(label,64,top+offset,270,45,22,true);await image(name,351,top+offset,w*scale,h*scale);
 }
}
slide.speakerNotes.textFrame.setText('The plotted quantity requires all four field and stage switches to produce the proper whole stopped image. Four different orders of clauses and fields are included for every held-out headword. Panel c uses fixed rows 0 and 3 in the first layout, including any errors.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Scoped_request_writer.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_81_scoped_request_writer.png'),new Uint8Array(await png.arrayBuffer()));
