import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/independent_entry_20261119');
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
text('Reading Chinese questions written by an independent model',48,20,1420,65,38,true);
text('a   Original prompt and answer images',48,100,1420,55,29,true);
for(const [tag,label,y] of [['gained_question','Question',165],['gained_answer','Output, exact',218],['missed_question','Question',289],['missed_answer','Output, wrong headword',382],['missed_reference','Reference',438]]){
  text(label,54,y,365,46,23,true,tag==='missed_answer'?'#934827':teal);
  const v=stats.strips[tag],scale=Math.min(32,1000/v.cols);
  await image(tag,430,y,v.cols*scale,Math.ceil(v.cells/v.cols)*scale);
}
text('The first answer gives “bright” and a bronze form. The second request asks about 编. Its greeting contains 老, the other headword on the card; the writer answers for 老.',55,509,1387,90,23);
text('b   Entire answer image correct, including stopping',48,623,1420,55,28,true);
table(stats.table,685,307,[360,180,165,170,160,350]);
text('The independent set includes all 96 local Qwen requests. The corrective construction has 1,136 distinct requests, each repeated four times in the registered evaluation.',55,1011,1390,77,21);
text('c   Mean loss per supervised writing action',48,1115,1410,50,28,true);
const chart=slide.charts.add('bar',{barOptions:{direction:'column',grouping:'clustered'},position:{left:48,top:1176,width:828,height:270},categories:['1','2','3','4','5','6','7','8'],hasLegend:true,legend:{position:'bottom'},
 series:['Shared','Candidate-specific','Replay'].map((name,i)=>({name,values:stats.series[i],fill:[teal,'#67839b','#ba8955'][i]})),
 xAxis:{textStyle:{typeface:'Arial',fontSize:16}},yAxis:{min:0,textStyle:{typeface:'Arial',fontSize:16},majorGridlines:{fill:'#DBE6E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('Eight passes over 89,200 requests',908,1175,515,70,23,true,teal);
text('Repeating the original questions gave 18 exact independent answers. Broader wording gave 66 with the shared update and 52 with the candidate-specific update.',908,1260,516,170,22);
slide.speakerNotes.textFrame.setText('The shared writer supplies both archived outputs. The first is the first clear independently authored request gained over its predecessor. The second is its first clear failure.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Independent_questions.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_72_independent_questions.png'),new Uint8Array(await png.arrayBuffer()));
