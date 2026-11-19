import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/font_invariant_reader_20261119');
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
text('Reading from pixels and composing in glyphs',50,22,1450,65,41,true);
text('a   Complete answers to the same 64 pictured questions',50,105,1450,50,29,true);
const chart=slide.charts.add('bar',{barOptions:{direction:'bar',grouping:'clustered'},position:{left:48,top:168,width:850,height:312},
 categories:['Regular','Bold'],hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:18}},
 series:[{name:'Raw pixel map',values:stats.series[0],fill:'#92ADAF'},{name:'Learned glyph metric',values:stats.series[1],fill:teal}],
 xAxis:{textStyle:{fontSize:20}},yAxis:{min:0,max:64,textStyle:{fontSize:18},majorGridlines:{fill:'#DFE6E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('Typeface changes the result',920,184,505,70,26,true,teal);
text('The metric learns to match different renderings of the same glyph. Bold improved from 1 to 41 correct answers. A separate, unseen Kai typeface still yielded only 1 of 64.',921,267,499,175,24);
text('b   A learned decision selects reading or composition',50,505,1450,50,29,true);
table(stats.table,564,211,[277,208,316,296,288]);
text('Person QA uses a sentence selected by glyph overlap. Bronze prose requires writing the requested sentence with a historical participant glyph.',60,796,1380,67,22);
text('c   Images produced by the combined model',50,898,1450,50,29,true);
async function strip(name,x,y,w,h){
 const [iw,ih]=stats.dimensions[name],f=Math.min(w/iw,h/ih);await image(name,x,y,iw*f,ih*f);
}
text('Read a name from bold print',55,982,430,53,24,true,teal);
await strip('fresh_bold_0_passage',492,970,928,62);
text('Question',60,1043,500,34,19,true);text('Returned image',1034,1043,390,34,19,true);
for(const i of [0,1]){
 await strip('fresh_bold_'+i+'_question',60,1090+i*55,933,43);
 await strip('fresh_bold_'+i+'_answer',1036,1090+i*55,364,43);
}
text('Compose with a bronze name',55,1240,435,53,24,true,teal);
await strip('historic_describe_0_passage',504,1226,892,59);
text('Question',60,1318,535,34,19,true);text('Returned image',961,1318,460,34,19,true);
await strip('historic_describe_0_question',59,1370,864,51);
await strip('historic_describe_0_answer',960,1370,456,51);
slide.speakerNotes.textFrame.setText('The reading and composition examples are the first cases of the bold fresh-scene and historical-description tests. CKIP BERT Base Chinese QA contextual weights are inherited; the dense visual interface supplies all lexical input. Source for inherited weights: https://huggingface.co/ckiplab/bert-base-chinese-qa');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Font_invariant_reader.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_88_font_invariant_reader.png'),new Uint8Array(await png.arrayBuffer()));
