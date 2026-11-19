import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/ordered_page_writer_20261119');
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
text('Writing two separated forms in the requested order',50,22,1450,65,41,true);
text('a   Both orderings correct on a historical page',50,101,1450,51,29,true);
const chart=slide.charts.add('bar',{barOptions:{direction:'bar',grouping:'clustered'},position:{left:46,top:164,width:870,height:320},
 categories:['Original wording','Rephrased','New cards and wording'],hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:16}},
 series:[{name:'Ordered',values:stats.series[0],fill:'#7F969F'},{name:'More replay',values:stats.series[1],fill:'#BDD2D1'},
 {name:'Varied requests',values:stats.series[2],fill:teal}],
 xAxis:{textStyle:{fontSize:17}},yAxis:{min:0,max:96,textStyle:{fontSize:16},majorGridlines:{fill:'#DFE6E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('96 contrasts per wording',949,181,473,60,26,true,teal);
text('Two prompts name the same historical forms in opposite orders. Each contrast earns one point only if both returned image sequences are complete and correct.',949,252,467,200,24);
text('b   Complete answers on ordinary Chinese questions',50,505,1450,49,29,true);
table(stats.table,567,228,[265,287,315,243,275]);
text('Each compound question joins two human-written questions about one passage. Reversing the requested order exchanges the two answers.',59,811,1372,70,22);
text('c   Historical forms of 言 returned by each writer',50,909,1450,53,29,true);
await image('yan_source',62,973,1380,87);
text('Requested order',58,1097,404,45,22,true,teal);
text('Prior responder',494,1097,307,45,22,true,teal);
text('Varied requests',815,1097,340,45,22,true,teal);
text('Reference',1195,1097,246,45,22,true,teal);
for (const [i,key] of ['early_first','late_first'].entries()) {
 const y=1166+i*112;
 text(i===0?'Oracle, then bronze':'Bronze, then oracle',58,y+5,400,46,25);
 await image(key+'_previous',499,y,79,64);
 await image(key+'_varied',822,y,145,64);
 await image(key+'_target',1194,y,145,64);
}
text('All returned and reference glyphs are original pixels from the recorded evaluation.',60,1400,1390,43,21);
slide.speakerNotes.textFrame.setText('The source is a Hanziyuan plate for 言 and 征. The historical chart is scored only across both requested orders. The earlier single-span responder cannot return the two separated forms in one response.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Ordered_page_writer.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_89_ordered_page_writer.png'),new Uint8Array(await png.arrayBuffer()));
