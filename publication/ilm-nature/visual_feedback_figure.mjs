import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/visual_feedback_20261118');
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
text('What happens after the writer makes a mistake?',44,17,1412,69,43,true);
text('a   Predicting the next picture after a wrong glyph',44,107,1420,52,31,true);
const chart=slide.charts.add('bar',{position:{left:36,top:174,width:851,height:238},categories:['Correct history','One wrong image'],hasLegend:false,
 barOptions:{direction:'bar',grouping:'clustered'},series:[{name:'Correct next images',values:stats.series[0],fill:teal}],
 xAxis:{textStyle:{typeface:'Arial',fontSize:20}},yAxis:{min:0,max:500,majorUnit:100,textStyle:{typeface:'Arial',fontSize:18},majorGridlines:{fill:'#DBE6E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('442 versus 133 correct next images',916,186,504,65,26,true,teal);
text('At the first error, substitute the image the model actually wrote. Score the following image against the same reference in 1,019 paired passages.',916,256,510,152,24);
text('b   Complete translations after four more passes through the books',44,451,1412,56,31,true);
table(stats.table,528,312,[360,210,223,290,302]);
text('1,024 new passages, 724 chapters. Error-trained and clean control received the same 2,368 updates. Autonomous stops fell from 989 to 674. Extra inserted glyphs can raise error above 100%.',62,864,1380,82,23);
text('c   One accurate translation and one that loses the subject',44,978,1412,55,31,true);
text('Military defeat',337,1057,525,44,25,true,teal);text('A prediction about a man’s death',874,1057,585,44,25,true,teal);
for(const [i,[arm,label]] of [['copy','Classical source'],['clean','Clean control'],['recovery','Error-trained'],['reference','Corpus gloss']].entries()){
 const y=1125+i*66;text(label,60,y,265,43,23,true);
 for(const [kind,x] of [['accepted',337],['failed',874]]){
  const key=kind+'_'+arm;await image(key,x,y,stats.widths[key]*28,28);
 }
}
slide.speakerNotes.textFrame.setText('Panel a uses the initial checkpoint. Panels b and c use the independent matched-training cohort. Both examples are original complete output images. In the left reference, one personal name is printed as two component glyphs. Grades were frozen before model assignments were scored.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Visual_feedback.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_59_visual_feedback.png'),new Uint8Array(await png.arrayBuffer()));
