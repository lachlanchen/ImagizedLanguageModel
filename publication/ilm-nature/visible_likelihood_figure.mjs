import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/visible_likelihood_20261119');
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
text('What does training on the visible glyph teach the writer?',44,17,1420,69,39,true);
text('a   An exact utterance and a mistaken diplomatic mission',44,107,1410,49,30,true);
text('A short refusal',286,170,365,44,24,true,teal);text('Who was sent where?',665,170,750,44,24,true,teal);
for(const [i,[name,label]] of [['source','Source'],['marginal','Neural writer'],['composer','Phrase composer'],['reference','Archive gloss']].entries()){
 const y=231+i*57;text(label,61,y,214,43,22,true);
 for(const [tag,x,scale] of [['exact',286,30],['mission',665,30]]){
  const key=tag+'_'+name;await image(key,x,y,stats.widths[key]*scale,scale);
 }
}
text('b   Translation of 1,024 passages from 423 unseen chapters',44,489,1415,53,30,true);
table(stats.table,557,292,[408,305,315,357]);
text('The two neural training arms saw 55,872 identical passages in the same order for four passes. Scoring used 25,610 archive glyphs. The interval for the 336-edit advantage was 0.03 to 2.56 percentage points.',62,875,1380,80,23);
text('c   Changing who spoke and whether permission was granted',44,982,1420,59,30,true);
const chart=slide.charts.add('bar',{position:{left:47,top:1070,width:715,height:345},categories:['Speaker / 8','Permission / 8','Whole utterance / 8','Both signs / 4'],hasLegend:true,
 barOptions:{direction:'bar',grouping:'clustered'},series:[{name:'Neural writer',values:stats.series[0],fill:teal},{name:'Phrase composer',values:stats.series[1],fill:'#7699AD'}],
 xAxis:{textStyle:{typeface:'Arial',fontSize:18}},yAxis:{min:0,max:8,majorUnit:2,textStyle:{typeface:'Arial',fontSize:18},majorGridlines:{fill:'#DBE6E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('Pictured prompt',791,1093,255,47,23,true,teal);text('Written answer',1090,1093,334,47,23,true,teal);
for(const [r,i] of [2,3].entries()){
 const y=1177+r*67;await image('speech_'+i+'_source',791,y,(i===2?6:7)*30,30);
 await image('speech_'+i+'_marginal',1090,y,(i===2?7:8)*30,30);
}
text('Eight prompts crossed four speakers with permission or refusal. All eight complete source images were absent from training.',790,1317,636,118,23);
slide.speakerNotes.textFrame.setText('The first exact answer with two changed lexical characters and a mission with an incorrect diplomatic role are shown. The targeted speech test includes its motivating example. Both systems got both signs for three of four speakers. Faithfulness on diverse prose remains poor.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Visible_likelihood.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_62_visible_likelihood.png'),new Uint8Array(await png.arrayBuffer()));
