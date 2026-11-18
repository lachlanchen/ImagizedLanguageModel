import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/block_translation_20261118');
const stats = JSON.parse(await fs.readFile(path.join(src,'figure.json')));

const deck = Presentation.create({slideSize:{width:1500,height:1760}});
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
text('Translating a pictured passage without a supplied answer',44,16,1410,68,42,true);
text('a   What the neural writer drew',44,103,1430,50,31,true);
for (const [i,[field,label]] of [['source','Classical source'],['answer','Neural output'],['gloss','Recorded gloss']].entries()){
  const key='neural_'+field, y=174+i*52;
  text(label,61,y,300,42,24,true);
  await image(key,372,y,stats.widths[key]*30,30);
}
text('The first enrolled passage asks whether an elderly servant could deceive his ruler. The writer repeats a common particle and obscures who might deceive whom.',62,340,1378,78,24);
text('b   Source order helps the writer, but copying has lower error',44,437,1430,53,31,true);
table(stats.table,510,320,[352,243,236,269,285]);
text('c   Composing observed phrases on 2,048 additional passages',44,866,1430,56,31,true);
const chart=slide.charts.add('bar',{position:{left:39,top:942,width:736,height:262},
 categories:['Copy source','Local majority','Phrase composer'],hasLegend:false,
 barOptions:{direction:'bar',grouping:'clustered'},series:[{name:'Bitmap edit distance',values:stats.exchange_counts[0],fill:teal}],
 xAxis:{textStyle:{typeface:'Arial',fontSize:21}},yAxis:{min:0,max:40000,majorUnit:10000,textStyle:{typeface:'Arial',fontSize:18},majorGridlines:{fill:'#DBE6E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('Total bitmap edits over 61,342 target glyphs',57,1213,730,38,20);
text('What changed in the sentence?',825,948,615,46,26,true,teal);
text('The composer reduced the error rate from 56.62% to 53.65%. It reproduced five changed complete glosses, all absent from training as complete answers. In a blinded reading, four of 24 outputs were acceptable, including one unchanged date.',825,1010,606,184,24);
text('d   A complete lexical change and a misrendered verb',44,1271,1430,52,31,true);
text('Exact modern gloss',375,1340,490,40,23,true,teal);
text('An unsuccessful modernization',885,1340,570,40,23,true,teal);
for(const [i,[field,label]] of [['source','Classical source'],['answer','Composer output'],['gloss','Recorded gloss']].entries()){
  const y=1401+i*53;
  text(label,61,y,298,40,24,true);
  for(const [id,x] of [['gain',375],['error',885]]){
    const key=id+'_'+field;
    await image(key,x,y,stats.widths[key]*32,32);
  }
}
const sizeStudy=JSON.parse(await fs.readFile(path.join(src,'scale_summary.json')));
text('e   Does increasing the training set improve translation?',44,1590,1430,54,31,true);
text('On '+sizeStudy.rows.toLocaleString('en-US')+' further passages, increasing training from 32,768 to 55,872 pairs reduced error from '+(100*sizeStudy.arms.small.cer).toFixed(2)+'% to '+(100*sizeStudy.arms.large.cer).toFixed(2)+'%. The 95% chapter interval for the difference spanned zero. Accepted answers changed from three to four of 24.',60,1661,1378,84,23);
slide.speakerNotes.textFrame.setText('Chinese strips reproduce full physical images directly. Cohorts comprise distinct passages from partially overlapping held-out chapters. Full counts and chapter-cluster uncertainty are in the text.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Block_translation.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,16764000','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_57_block_translation.png'),new Uint8Array(await png.arrayBuffer()));
