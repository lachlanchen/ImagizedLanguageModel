import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/pictured_revision_20261118');
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
text('Does rereading the source improve the translation?',44,17,1412,69,43,true);
text('a   Original writing from two complete passages',44,106,1420,52,31,true);
text('An exact new gloss',348,177,467,43,26,true,teal);text('A damaged calendar date',871,177,540,43,26,true,teal);
for(const [i,[arm,label]] of [['source','Classical source'],['draft','First draft'],['revised','Revised answer'],['reference','Corpus gloss']].entries()){
 const y=247+i*61;text(label,63,y,264,42,23,true);
 for(const [kind,x] of [['gain',348],['loss',871]]){
  const key=kind+'_'+arm;await image(key,x,y,stats.widths[key]*32,32);
 }
}
text('b   Complete translation with the same first draft',44,525,1430,56,31,true);
table(stats.table,600,258,[375,235,215,260,300]);
text('1,024 passages from 623 new chapters. The reviser changed 189 drafts. Of these, 82 moved closer to the recorded gloss, 40 moved further away and 67 had the same distance.',64,882,1370,80,24);
text('c   Faithful modern translations in two blinded readings',44,1000,1420,54,31,true);
const chart=slide.charts.add('bar',{position:{left:44,top:1074,width:792,height:344},categories:['First draft','Revised','Qwen3 text','Corpus gloss'],hasLegend:true,
 barOptions:{direction:'bar',grouping:'clustered'},series:[{name:'All-comers (18)',values:stats.series[0],fill:teal},{name:'Changed drafts (18)',values:stats.series[1],fill:'#7297AE'}],
 xAxis:{textStyle:{typeface:'Arial',fontSize:18}},yAxis:{min:0,max:18,majorUnit:3,textStyle:{typeface:'Arial',fontSize:18},majorGridlines:{fill:'#DBE6E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('What changes when the teacher reads pixels?',860,1060,574,80,25,true,teal);
text('On the same 18 all-comers, Qwen3-VL gave '+stats.vision.typed+' faithful translations from typed sources and '+stats.vision.image+' from pictures. It transcribed '+stats.ocr+' complete pictured sources exactly. Both translation arms used its pretrained text decoder.',860,1170,566,224,24);
slide.speakerNotes.textFrame.setText('The first revised exact gain and first frozen faithful-draft loss are shown. The chart uses the actual complete blind readings. The two packets overlap in four cases. The pretrained teacher was evaluated separately from the two-stage pictured writer.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Pictured_revision.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_60_pictured_revision.png'),new Uint8Array(await png.arrayBuffer()));
