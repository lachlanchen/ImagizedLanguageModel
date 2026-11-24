import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src=path.join(root,'references/evidence/pictured_action_lookup_v1');
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
text('Choosing a Chinese line from a pictured English question',48,15,1420,64,40,true);
text('Each question names an action. The model draws the English meaning of its object.',54,88,1395,57,23,false,teal);
text('a   Complete contrasts on 1,691 held two-line pages',49,156,1410,54,29,true);
const colors=['#A5B1B9','#BD865D','#5E819C','#16686B'];
const chart=slide.charts.add('bar',{barOptions:{direction:'column',grouping:'clustered'},
 position:{left:45,top:221,width:920,height:315},categories:['Selected both lines','Located and wrote both objects'],
 hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:16}},
 series:['Exchanged','Glyph counts','Learned','Combined'].map((name,i)=>({name,values:stats.series[i],fill:colors[i]})),
 xAxis:{textStyle:{fontSize:17}},yAxis:{min:0,max:1500,majorUnit:500,textStyle:{fontSize:17},majorGridlines:{fill:'#DCE5E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('Eight requests per page',986,242,460,48,24,true,teal);
text('Both actions, two held question frames and both line orders must succeed. The count control receives the English action name.',986,306,445,207,22);
text('b   The same two-line page gives two different written answers',49,568,1405,53,29,true);
for(const [label,x] of [['Page image',52],['Action in the question',644],['Generated reply',1020]])text(label,x,639,410,48,22,true,teal);
await bitmap('samples/example-page',52,710,490,164);
for(const [i,verb] of [[0,'get'],[1,'whip']]){
 const y=724+88*i;text(verb,645,y,340,54,28);await bitmap('samples/example-literal-answer-'+i,1022,y,414,51);
}
text('The question reads “What is it that they … here?” Answers shown are the emitted glyph images.',55,947,1380,65,22);
text('c   Changing the action name on the same 188 held pages',49,1047,1410,54,29,true);
table(stats.table,1122,190,[485,161,360,379]);
text('The model and source images are fixed across wordings. The matched subset includes only pairs with eligible alternate names.',56,1364,1390,70,21);
slide.speakerNotes.textFrame.setText('The 703 written pairs include 165 of 487 from reserved works and 538 of 1204 from reserved sections. The neural-only model gives 667. The pure optical substring control gives 546. In the example, the passages concern obtaining benevolence and whipping a horse. This example was chosen for legibility. Across the alternate-name set, both drawn meanings remain correct on seven of 188 pairs. Exact word gloss and source pointer are required.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Pictured_action_lookup.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_124_pictured_action_lookup.png'),new Uint8Array(await png.arrayBuffer()));
