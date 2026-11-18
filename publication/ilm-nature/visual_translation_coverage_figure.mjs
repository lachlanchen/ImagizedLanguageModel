import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/visual_translation_coverage_20261119');
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
text('How far does learning a reviewed translation transfer?',48,20,1420,75,39,true);
text('a   A learned biography and an error of meaning',48,109,1400,50,30,true);
text('Classical input',335,172,450,42,25,true,teal);
text('Broad writer output',815,172,560,42,25,true,teal);
for(const [i,[tag,label]] of [['supervised','Reviewed biography'],['natural','New book passage'],['modified','Changed quantity']].entries()){
  const top=232+i*80;
  text(label,52,top,275,55,23,true);
  for(const [suffix,x,width] of [['input',333,454],['answer',813,620]]){
    const key=tag+'_'+suffix, n=stats.strips[key].cells, scale=Math.min(37,width/n);
    await image(key,x,top,n*scale,scale);
  }
}
text('The biography gives Lü Yi’s courtesy name and birthplace. In the two new inputs, 360 and 560 represent a celestial count. Both answers add the glyph for years of age.',55,486,1380,81,23);
text('b   Complete answers to unfamiliar passages',48,606,1410,50,30,true);
table(stats.table,676,296,[325,225,210,295,330]);
text('Edits compare with a single book gloss. Paired columns require both the original answer and its response to a changed quantity. Literal copying preserves all eight quantities.',55,987,1390,75,21);
text('c   Training objective across the matched fits',48,1094,1410,50,28,true);
const chart=slide.charts.add('line',{position:{left:46,top:1160,width:840,height:278},categories:['194','388','582','776','970','1,164'],hasLegend:true,legend:{position:'bottom'},
 series:['108 sources','841 sources'].map((name,i)=>({name,values:stats.series[i],line:{fill:['#6483A2',teal][i],width:4}})),
 xAxis:{textStyle:{typeface:'Arial',fontSize:17}},yAxis:{min:2.4,max:3.2,majorUnit:0.2,textStyle:{typeface:'Arial',fontSize:16},majorGridlines:{fill:'#DBE6E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('What repetition bought',925,1165,480,51,25,true,teal);
text('On the 108 shared supervised passages, the narrow writer reproduced 49 exact answers. The broad writer reproduced two. Each fit used 18,624 reviewed exposures.',925,1231,475,190,23);
slide.speakerNotes.textFrame.setText('Source and emitted images come directly from the archived packed cells. Semantic judgments were made with method identities masked by one model reader. Curves average the training objective over each interval of 194 updates.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Translation_coverage.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_68_translation_coverage.png'),new Uint8Array(await png.arrayBuffer()));
