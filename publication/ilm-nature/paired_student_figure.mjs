import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/paired_student_20261119');
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
text('Can a pictured writer learn from a second translation?',44,17,1420,68,41,true);
text('a   Complete output images on two unseen passages',44,104,1415,50,30,true);
text('Fewer archive edits',355,174,482,44,26,true,teal);text('More archive edits',885,174,520,44,26,true,teal);
for(const [i,[arm,label]] of [['source','Classical source'],['archive','Archive replay'],['teacher','Teacher + archive'],['pooled','Pooled decoder'],['reference','Archive gloss']].entries()){
 const y=222+i*52;text(label,62,y,275,43,23,true);
 for(const [kind,x] of [['gain',355],['loss',885]]){
  const key=kind+'_'+arm;await image(key,x,y,stats.widths[key]*32,32);
 }
}
text('b   Glyph edits across complete emitted answers',44,502,1410,50,30,true);
table(stats.table,567,316,[407,315,315,348]);
text('Archive scoring uses 1,024 passages from '+stats.chapters+' chapters. Teacher scoring uses '+stats.quantitative+' complete, renderable translations of the same sources. Both students received 192 updates.',62,904,1370,78,24);
text('c   Faithful translations in a masked reading of 18 passages',44,1011,1410,52,30,true);
const chart=slide.charts.add('bar',{position:{left:45,top:1086,width:805,height:341},categories:['Parent','Archive replay','Teacher + archive','Pooled decoder','Exchanged source','Composer','Qwen3-VL text','Archive gloss'],hasLegend:false,
 barOptions:{direction:'bar',grouping:'clustered'},series:[{name:'Faithful complete translations',values:stats.series[0],fill:teal}],
 xAxis:{textStyle:{typeface:'Arial',fontSize:18}},yAxis:{min:0,max:18,majorUnit:3,textStyle:{typeface:'Arial',fontSize:18},majorGridlines:{fill:'#DBE6E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('Does pooling improve writing?',907,1080,488,65,25,true,teal);
text('Pooling raised correct first glyphs from 393 to 451 against the archive. Across complete passages it saved 2,545 glyph edits. Both decoders produced zero faithful answers in the 18-passage reading.',907,1164,492,222,24);
slide.speakerNotes.textFrame.setText('First compact gain and first compact loss against archive replay. Edit counts compare pictures. The independent masked reading evaluates the entire modern meaning.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Paired_student.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_61_paired_student.png'),new Uint8Array(await png.arrayBuffer()));
