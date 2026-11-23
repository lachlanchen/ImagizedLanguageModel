import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/pictured_relation_20261123');
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

text('Reading the role named in a pictured question',50,25,1410,65,43,true);
text('a   Generating the same fact in both requested word orders',50,111,1410,50,29,true);
const chart=slide.charts.add('bar',{barOptions:{direction:'bar',grouping:'clustered'},
 position:{left:45,top:163,width:895,height:310},
 categories:['Human questions','Programme notes','DuIE passages'],hasLegend:true,
 legend:{position:'bottom',textStyle:{fontSize:18}},series:[
 {name:'Previous reader',values:stats.series[0],fill:'#9DABB2'},
 {name:'With question reader',values:stats.series[1],fill:teal}],
 xAxis:{textStyle:{fontSize:17}},yAxis:{min:0,max:100,textStyle:{fontSize:16},majorGridlines:{fill:'#DFE6E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('Both replies exact (%)',971,185,461,50,25,true,teal);
text('Each reply must stop and match the reference pixels. The updated reader passed 34 of 48 human questions, 18 of 32 programme questions and 892 of 1,200 DuIE passages.',971,248,451,204,23);
text('b   Complete answers written with the learned sentence pattern',50,488,1410,48,28,true);
table(stats.table,550,164,[345,235,215,205,205,180]);
text('c   A composer previously credited as a producer',50,752,1400,55,29,true);
async function row(label,name,y){
 text(label,57,y+3,395,40,22,true,teal);
 const blob=await fs.readFile(path.join(src,name+'.png'));
 const w=blob.readUInt32BE(16), h=blob.readUInt32BE(20);
 const height=Math.min(49,938*h/w);
 await image(name,481,y,w*height/h,height);
}
await row('Passage','song_0_passage',819);
await row('Question','song_0_question',879);
await row('Previous answer','song_0_baseline_object',939);
await row('Updated answer','song_0_updated_object',999);
text('d   After exchanging the names on a different song credit',50,1099,1400,55,29,true);
await row('Revised passage','song_3_passage',1164);
await row('Question','song_3_question',1225);
await row('Updated answer','song_3_updated_object',1286);
text('The page credits Su Tian with the music. The model still names Zhou Zhi, who wrote the lyrics.',57,1382,1380,62,24);
slide.speakerNotes.textFrame.setText('Panels c and d contain original physical image strips. Panel c was the diagnostic that motivated the comparison. Panel d is a separate song, tested with both people in both roles.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Pictured_relation.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_94_pictured_relation.png'),new Uint8Array(await png.arrayBuffer()));
