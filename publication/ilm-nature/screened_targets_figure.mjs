import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/screened_targets_20261119');
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
text('Changing one pictured numeral changes a complete answer',48,20,1420,70,39,true);
text('a   What the visual writer actually emits',48,110,1400,50,30,true);
text('Classical input',320,171,400,42,24,true,teal);
text('Complete generated answer',820,171,610,42,24,true,teal);
for (const [i,[tag,label]] of [['familiar','Training sentence'],['changed','Unseen numeral edit'],['unfamiliar','Unseen chapter']].entries()){
  const top=234+i*85;
  text(label,52,top,264,55,22,true);
  for(const [suffix,x,width] of [['source',320,475],['screened',820,610]]){
    const key=tag+'_'+suffix;
    const scale=Math.min(35,width/stats.strips[key].cells);
    await image(key,x,top,stats.strips[key].cells*scale,scale);
  }
}
text('Rows 1–2 describe famine relief in Shaanxi. Replacing the third month with the fifth changes the written date. Row 3 gives two times of day. The generated sentence loses that meaning.',55,501,1380,80,23);
text('b   Transfer to new passages and paired prompts',48,626,1410,50,30,true);
table(stats.table,693,282,[337,219,228,302,299]);
text('Edits count glyph insertions, deletions and substitutions against a single book translation. The two rightmost columns require success on both sides of a change in role, number or negation.',55,992,1390,65,21);
text('c   Exact complete translations on 113 supervised sentences',48,1096,1410,50,28,true);
const chart=slide.charts.add('bar',{position:{left:46,top:1153,width:825,height:287},categories:['Starting writer','Book targets','Reviewed targets'],hasLegend:true,legend:{position:'bottom'},
 barOptions:{direction:'column',grouping:'clustered'},series:['Book wording','Reviewed wording'].map((name,i)=>({name,values:stats.series[i],fill:['#6483A2',teal][i]})),
 xAxis:{textStyle:{typeface:'Arial',fontSize:16}},yAxis:{min:0,max:60,majorUnit:15,textStyle:{typeface:'Arial',fontSize:15},majorGridlines:{fill:'#DBE6E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('Change one image in a familiar frame',910,1160,530,68,24,true,teal);
text('13 of 18 changed quantities yield the exact revised answer. All 13 answers were new. A rule that substitutes a single glyph in a stored translation solves 18 of 18 and abstains on all 118 new passages.',910,1240,520,194,23);
slide.speakerNotes.textFrame.setText('Unaltered saved source and answer cells. The two matched fits replayed the same 55,872 pictured sources and oversampled the same 113 sources. The post-fit Qwen comparison used text. Masked scores came from a model reader. The local number probes use familiar sentence frames.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Screened_targets.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_67_screened_targets.png'),new Uint8Array(await png.arrayBuffer()));
