import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/paired_stance_writer_20261119');
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
text('Writing opposite positions from pictured Chinese evidence',50,22,1450,65,39,true);
text('a   Both positions correct for the same question (%)',50,104,1450,50,29,true);
const chart=slide.charts.add('bar',{barOptions:{direction:'bar',grouping:'clustered'},position:{left:40,top:174,width:790,height:302},
 categories:['Lexical marker','Reassigned','Matched'],hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:17}},
 series:[{name:'Validation (144 pairs)',values:stats.series[0],fill:'#83ABA9'},{name:'Official (18 pairs)',values:stats.series[1],fill:teal}],
 xAxis:{textStyle:{fontSize:20}},yAxis:{min:0,max:100,textStyle:{fontSize:19},majorGridlines:{fill:'#DBE6E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('One question, two human answers',883,171,560,77,27,true,teal);
text('Each passage supports a different position. Both writers learn to quote the same human evidence. The control changes which position accompanies it.',883,250,525,167,24);
text('b   What the answer image contains',50,505,1450,52,29,true);
table(stats.table,580,205,[210,208,185,210,252,320]);
text('Quotes require an exact, stopped human quotation after either valid position. Complete answers require the correct position as well.',60,802,1370,77,22);
text('c   Emitted position prefixes on the first official question',50,915,1450,52,29,true);
text('Human annotation',70,1000,355,50,24,true);
text('Reassigned writer',522,1000,355,50,24,true);
text('Matched writer',1020,1000,365,50,24,true);
for (const [i,label] of [[0,'Affirmative passage'],[1,'Negative passage']]) {
 const y=1084+i*136;
 text(label,70,y+7,418,60,26,true,teal);
 await image('reassigned'+i,525,y,192,64);
 await image('matched'+i,1020,y,192,64);
}
text('Three original 32 × 32 cells per prefix. The corresponding human quotations are scored in panel b.',70,1375,1360,58,22);
slide.speakerNotes.textFrame.setText('Every test question has independently annotated affirmative and negative passages. This figure separates the learned decision from the evidence quotation. The source passages were cropped using the references.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Paired_stance_writer.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_82_paired_stance_writer.png'),new Uint8Array(await png.arrayBuffer()));
