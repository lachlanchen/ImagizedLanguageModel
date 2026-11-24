import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src=path.join(root,'references/evidence/verse_contrast_v1');
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



async function bitmap(name,x,y,width,height){
 const data=await fs.readFile(path.join(src,name+'.png'));
 const w=data.readUInt32BE(16),h=data.readUInt32BE(20),scale=Math.min(width/w,height/h);
 await image(name,x,y,w*scale,h*scale);
}
text('Can a writer recover the missing phrase?',48,18,1410,66,43,true);
text('Fitted on Tang verse   /   tested on 3,854 Song lines   /   192 poet attributions',53,90,1390,48,25,false,teal);
text('a   Complete repairs from the original pictured line',50,170,1400,50,31,true);
text('Both missing glyphs correct (pages)',62,235,760,34,19,false,teal);
const chart=slide.charts.add('bar',{barOptions:{direction:'column',grouping:'clustered'},
 position:{left:45,top:274,width:830,height:275},categories:['Frozen image','Paired image','Contrastive image','Contrastive byte'],
 hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:17}},
 series:['Left-first greedy','Eight first glyphs'].map((name,i)=>({name,values:stats.series[i],fill:['#B99B76','#16686B'][i]})),
 xAxis:{textStyle:{fontSize:15}},yAxis:{min:0,textStyle:{fontSize:17},majorGridlines:{fill:'#DCE5E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('Reading before writing',928,270,479,60,26,true,teal);
text('Each writer sees a first line and a damaged second line. The contrastive loss asks whether two authentic endings fit their own sources better than the exchanged ones.',928,337,488,216,23);
text('b   What the contrastive image writer put on the page',50,572,1400,54,31,true);
text('First held page',58,645,620,44,24,true,teal);
text('First exact repair lost after source exchange',749,645,690,52,24,true,teal);
for(const [x,id] of [[58,'first'],[748,'repair']]){
 for(const [i,[label,suffix]] of [['Source','source'],['Damaged line','damaged'],['Written line','output'],['Recorded line','reference']].entries()){
  const y=724+66*i;
  text(label,x,y,215,40,21,i===2);
  await bitmap(id+'-'+suffix,x+219,y-8,419,49);
 }
}
text('c   Choosing the correct source and writing the missing cells',50,1002,1400,56,31,true);
table(stats.table,1080,224,[342,394,306,343]);
text('The assignment test supplies both real endings. The repair test withholds the two adjacent glyphs. The last column swaps the first line while keeping all surviving second-line ink.',55,1344,1390,96,23);
slide.speakerNotes.textFrame.setText('The native chart and table use measured counts from the frozen Song evaluation. Image strips retain the actual input, damaged lower line, upper-conditioned prediction, and recorded lower line. Exact repair credits both cells together.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Verse_contrast.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_137_verse_contrast.png'),new Uint8Array(await png.arrayBuffer()));
