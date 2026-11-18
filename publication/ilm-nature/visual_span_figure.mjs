import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/visual_span_20261118');
const stats = JSON.parse(await fs.readFile(path.join(src,'figure.json')));

const deck = Presentation.create({slideSize:{width:1500,height:1300}});
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
text('Choosing a phrase while reading pictured Chinese',44,12,1410,65,42,true);
text('a   Original source, generated writing and modern reference',44,86,1410,48,30,true);
text('A changed answer',380,142,475,38,23,true,teal);
text('A runaway insertion',885,142,540,38,23,true,teal);
for(const [i,[field,label]] of [['source','Classical source'],['marginal','Single edit path'],['marginal_prefix','Next image'],['risk','Minimum edit risk'],['reference','Modern reference']].entries()){
  const y=196+i*49;
  text(label,64,y,305,38,23,true);
  for(const [name,x] of [['changed',380],['capped',885]]){
    const key=name+'_'+field;
    await image(key,x,y,stats.widths[key]*29,29);
  }
}
text('For the runaway output, the single-path row shows the first 16 of 192 glyphs.',62,447,1390,34,20);
text('b   Complete translations and glyph edit rate',44,504,1420,45,30,true);
text('Each column contains 1,024 requests. Each entry gives the exact count and edits per 100 reference glyphs.',61,557,1410,42,21);
table(stats.table,617,285,[620,380,385]);
text('c   Assigning exchanged phrases to their recorded contexts',44,940,1420,46,30,true);
const chart=slide.charts.add('bar',{position:{left:50,top:1012,width:835,height:258},
 categories:['Discovery','Replication'],hasLegend:true,legend:{position:'bottom'},
 barOptions:{direction:'column',grouping:'clustered'},series:[
 {name:'Original source order',values:stats.span_accuracy_percent[0],fill:teal},
 {name:'Reversed source',values:stats.span_accuracy_percent[1],fill:'#ADC9C8'}],
 xAxis:{textStyle:{typeface:'Arial',fontSize:18}},yAxis:{min:0,max:100,majorUnit:25,
 textStyle:{typeface:'Arial',fontSize:18},majorGridlines:{fill:'#DCE4E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('Correct assignments (%)',915,1018,540,53,28,true,teal);
text('Paired phrases share length and both neighboring modern glyphs. All exchanged glyphs are absent from both classical sources.',915,1085,535,155,22);
slide.speakerNotes.textFrame.setText('All images are original saved output pixels. The left example is the first exact longer mixed discovery answer from summed-path training with single-path decoding. The right example is the first capped discovery answer with source and modern reference of at most fourteen glyphs. The 192-glyph row is cropped to its first sixteen cells. The paired test uses 154 and 139 nonoverlapping sentence pairs.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Visual_span.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,12382500','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_51_visual_span.png'),new Uint8Array(await png.arrayBuffer()));
