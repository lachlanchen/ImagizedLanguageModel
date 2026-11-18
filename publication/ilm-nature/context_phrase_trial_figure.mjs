import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/context_phrase_trial_20261119');
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
text('Can the pictured source determine the next phrase?',48,20,1420,75,39,true);
text('a   Three independent translations',48,115,1400,50,30,true);
text('Classical input',335,176,450,42,25,true,teal);
text('Independent visual answer',815,176,615,42,25,true,teal);
for(const [i,label] of ['Father answers','Son answers','Teacher learns'].entries()){
  const top=221+i*70;
  text(label,52,top,277,52,24,true);
  for(const [method,x,width] of [['source',333,454],['crossed',813,620]]){
    const key=method+i, n=stats.strips[key].cells, scale=Math.min(40,width/n);
    await image(key,x,top,n*scale,scale);
  }
}
text('The writer loses who answered when the father asks the son. The final row shows a credited answer from a different role pair. The teacher learns ritual from a pupil.',55,430,1380,85,23);
text('b   Choosing a phrase and writing an answer',48,551,1410,50,30,true);
table(stats.table,626,354,[280,332,315,195,263]);
text('Both choices require the right answer for each source given a prefix and alternatives. Whole translations and paired details were graded from independently generated answers with method identities masked.',55,1003,1390,78,21);
text('c   Training on source and phrase assignments',48,1114,1410,50,28,true);
const chart=slide.charts.add('line',{position:{left:46,top:1170,width:840,height:270},categories:['97','194','291','388','485','582','679','776','873'],hasLegend:true,legend:{position:'bottom'},
 series:['Rehearsal','Crossed phrases'].map((name,i)=>({name,values:stats.series[i],line:{fill:['#6483A2',teal][i],width:4}})),
 xAxis:{textStyle:{typeface:'Arial',fontSize:17}},yAxis:{min:0,textStyle:{typeface:'Arial',fontSize:16},majorGridlines:{fill:'#DBE6E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('Mean pairwise logistic loss',925,1177,485,58,24,true,teal);
text('Each point averages 97 updates. Both writers saw the same images. Only the crossed writer included this loss in its training objective.',925,1250,475,173,23);
slide.speakerNotes.textFrame.setText('The father and son are the first registered role swap. The last example was selected for a credited reading. Curves use the same contrast for both writers. Dashes indicate an inapplicable measurement.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Context_phrases.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_69_context_phrases.png'),new Uint8Array(await png.arrayBuffer()));
