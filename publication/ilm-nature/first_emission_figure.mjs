import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/first_emission_20261118');
const stats = JSON.parse(await fs.readFile(path.join(src,'figure.json')));

const deck = Presentation.create({slideSize:{width:1500,height:1610}});
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
text('Learning from the first glyph a writer gets wrong',44,12,1410,65,42,true);
text('a   A corrected first glyph and a persistent error',44,86,1430,48,30,true);
text('A complete multi-glyph gain',355,149,540,40,23,true,teal);
text('An incorrect opening glyph',908,149,550,40,23,true,teal);
for(const [i,[field,label]] of [['source','Classical sentence'],['prefix','Modern prefix, last 14'],['frozen','Frozen writer'],['prediction','Preference'],['reference','Recorded insertion']].entries()){
  const y=210+i*49;
  text(label,64,y,279,38,23,true);
  for(const [name,x] of [['gained',355],['missed',908]]){
    const key=name+'_'+field;
    await image(key,x,y,stats.widths[key]*24,24);
  }
}
text('The writer sees the classical sentence, the insertion boundary and the modern prefix. It draws the entire insertion and decides when to stop.',62,464,1390,62,22);
text('b   Exact stopped insertions across five cohorts',44,548,1420,45,30,true);
table(stats.table,608,284,[335,210,210,205,205,220]);
text('c   What changed in the first generated image?',44,922,1420,46,30,true);
const chart=slide.charts.add('bar',{position:{left:50,top:1001,width:845,height:270},
 categories:['Wrong to correct','Correct to wrong','Wrong to another error'],hasLegend:true,legend:{position:'bottom'},
 barOptions:{direction:'column',grouping:'clustered'},series:[

 {name:'Likelihood',values:stats.exchange_counts[0],fill:'#D9A55B'},
 {name:'Preference',values:stats.exchange_counts[1],fill:teal}],
 xAxis:{textStyle:{typeface:'Arial',fontSize:17}},yAxis:{min:0,
 textStyle:{typeface:'Arial',fontSize:18},majorGridlines:{fill:'#DCE4E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('The old error as a training example',920,1014,505,48,26,true,teal);
text('Only the first drawing state receives the correction. The writer feeds the new image back into its frozen continuation and ending mechanism. Counts cover all 24,048 insertions.',920,1082,514,150,22);
const sentence = JSON.parse(await fs.readFile(path.join(src,'sentences.json')));
text('d   What happens when the model chooses its own boundaries?',44,1308,1435,54,30,true);
for(const [i,[key,label]] of [['source','Classical source'],['frozen','Frozen writer'],['preference','Preference'],['reference','Recorded modern gloss']].entries()){
  const y=1380+i*44;
  text(label,60,y,290,40,23,true);
  await image('sentence_'+key,355,y,sentence.sample.widths['sentence_'+key]*25,25);
}
text('Complete translation',920,1382,480,43,26,true,teal);
text('Both writers recovered 0 of 84 recorded sentences. The preference writer stopped on 22 and reached its phrase limit on 62. The complete output at left uses its own preceding images.',920,1433,508,160,22);
slide.speakerNotes.textFrame.setText('Original physical answer strips, two fixed-order examples from Holdout 2. The table covers all 24,048 sites. The chart recounts changes in the first physical bitmap relative to the frozen writer. The 84 newly reserved passages include all eligible rows from the registered archive.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/First_emission.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,15335250','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_56_first_emission.png'),new Uint8Array(await png.arrayBuffer()));
