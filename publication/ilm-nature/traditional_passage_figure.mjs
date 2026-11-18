import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/traditional_passage_20261119');
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
text('Learning to select evidence in traditional Chinese',50,20,1420,65,40,true);
text('a   A question about the census year',50,100,1420,52,29,true);
text('b   A question about a region still elicits a person',50,453,1420,55,29,true);
for (const [tag,label,y] of [
 ['gained_question','Pictured question',171],['gained_replay','CMRC replay',229],['gained_mixed','CMRC + DRCD',286],['gained_reference','Reference',343],
 ['missed_question','Pictured question',526],['missed_replay','CMRC replay',582],['missed_mixed','CMRC + DRCD',638],['missed_reference','Reference',694]]) {
  text(label,62,y-2,264,42,23,true,teal);
  const [w,h]=stats.dimensions[tag];
  const ratio=Math.min(1.06,1060/w);
  await image(tag,341,y,w*ratio,h*ratio);
}
text('c   Exact answer images and learned termination',50,786,1420,57,29,true);
table(stats.table,859,243,[510,165,220,235,255]);
text('d   What changed in the 3,505 individual DRCD answers?',50,1144,1420,54,28,true);
const chart=slide.charts.add('bar',{barOptions:{direction:'column',grouping:'stacked'},position:{left:50,top:1205,width:787,height:245},
 categories:['Released','CMRC replay','CMRC + DRCD'],hasLegend:true,legend:{position:'bottom'},
 series:['Exact answer','Includes answer with extra ink'].map((name,i)=>({name,values:stats.series[i],fill:[teal,'#c99d72'][i]})),
 xAxis:{textStyle:{typeface:'Arial',fontSize:17}},yAxis:{min:0,textStyle:{typeface:'Arial',fontSize:17},majorGridlines:{fill:'#DBE6E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('Percentage of questions',874,1207,540,60,24,true,teal);
text('Of 947 newly exact answers over replay, 331 previously included the reference with extra writing. The other 616 previously lacked a complete accepted answer.',874,1270,545,159,22);
slide.speakerNotes.textFrame.setText('The first gain and the first remaining failure are shown in corpus order. All comparisons share the initial weights. Both fits have the same scheduled examples and optimizer steps.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Traditional_passage.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_74_traditional_passage.png'),new Uint8Array(await png.arrayBuffer()));
