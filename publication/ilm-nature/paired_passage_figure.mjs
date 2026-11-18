import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/paired_passage_20261119');
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
text('Reading two questions from one image',48,18,1420,62,40,true);
text('a   Reversing the questions exchanges the answers',48,90,1410,53,29,true);
for (const [tag,label,y,w,h] of [
 ['forward_prompt','Question pair',154,1150,28],['forward_answer','Output',207,924,28],
 ['reverse_prompt','Reversed pair',277,1150,28],['reverse_answer','Output',330,924,28],
 ['missed_prompt','Question pair',495,1150,52],['missed_answer','Output',570,1150,52],
 ['missed_reference','Reference',648,336,28]]) {
 text(label,60,y-4,225,48,24,true,tag==='missed_answer'?'#924629':teal);
 await image(tag,300,y,w,h);
}
text('b   The reader chooses the wrong description and biography',48,421,1420,58,29,true);
text('c   Complete answer images on 425 held-out training-release passages',48,737,1420,54,28,true);
table(stats.table,801,263,[490,312,288,295]);
text('d   Transfer to official development passages',48,1090,1420,55,28,true);
const chart=slide.charts.add('bar',{barOptions:{direction:'column',grouping:'clustered'},position:{left:49,top:1153,width:812,height:290},
 categories:['Familiar','Short Chinese','New Chinese','English'],hasLegend:true,legend:{position:'bottom'},
 series:['Fixed training','Varied training','Varied + ordered decoder'].map((name,i)=>({name,values:stats.series[i],fill:['#7897AA','#b17f52',teal][i]})),
 xAxis:{textStyle:{typeface:'Arial',fontSize:17}},yAxis:{min:0,textStyle:{typeface:'Arial',fontSize:17},majorGridlines:{fill:'#DBE6E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('Both question orders correct (%)',895,1158,541,83,23,true,teal);
text(stats.externalNote,895,1246,535,178,23);
slide.speakerNotes.textFrame.setText('Images are the first correct exchanged pair and first failed request in corpus order from the refined reader. The chart uses released official development passages. The third series uses the same varied planner with constrained action decoding.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Paired_passage.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_73_paired_passage.png'),new Uint8Array(await png.arrayBuffer()));
