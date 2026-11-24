import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src=path.join(root,'references/evidence/visual_word_senses_20261124');
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
text('One printed character, two written meanings',50,16,1420,64,43,true);
text('A marked cell in classical Chinese   →   an ordered window of ink   →   recurrent drawing of an English gloss',55,90,1410,46,23,false,teal);
text('a   Exact English glosses on unseen works (%)',50,154,1420,45,29,true);
const colors=['#B97854','#28767A','#AAB4BE'];
const captions=['Shuffled glosses','Aligned writer','Local counts'];
const chart=slide.charts.add('bar',{barOptions:{direction:'column',grouping:'clustered'},
 position:{left:62,top:207,width:936,height:305},categories:['Analects · all','Analects · pairs','Chu ci · all','Chu ci · pairs'],
 hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:17}},
 series:captions.map((name,i)=>({name,values:stats.series[i],fill:colors[i]})),
 xAxis:{textStyle:{fontSize:17}},yAxis:{min:0,max:100,majorUnit:25,textStyle:{fontSize:17},majorGridlines:{fill:'#DFE6E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('What earns pair credit?',1020,219,403,63,25,true,teal);
text('Two different passages share the same central printed character. Both complete gloss images must match their distinct annotated readings and stop.',1020,282,406,220,22);
text('b   Complete pairs after changing only the source typeface',50,525,1410,62,28,true);
table(stats.table,598,202,[291,279,278,269,268]);
text('Ink matching uses only regular and bold shapes from training. The output typeface stays fixed.',60,814,1370,48,21);
text('c   The marked character 言 in two passages of the Analects',50,887,1400,52,28,true);
text('Pictured context',65,947,605,35,21,true,teal);text('Model output',715,947,425,35,21,true,teal);
for(let i=0;i<2;i++){await bitmap('yan_'+i+'_page',55,991+i*54,590,42);await bitmap('yan_'+i+'_answer',715,991+i*54,425,40);}
text('d   The same writer misses the second reading of 一 in Chu ci',50,1120,1400,57,28,true);
text('Pictured context',65,1189,597,38,21,true,teal);text('Output',712,1189,280,38,21,true,teal);text('Reference',1070,1189,290,38,21,true,teal);
for(let i=0;i<2;i++){await bitmap('error_'+i+'_page',55,1236+i*57,590,41);await bitmap('error_'+i+'_answer',712,1236+i*57,230,40);await bitmap('error_'+i+'_reference',1070,1236+i*57,270,40);}
text('The underline identifies the queried cell. The displayed answers are the original 32 × 32 output bitmaps.',60,1380,1370,64,21);
slide.speakerNotes.textFrame.setText('Independent eight-pass fits used 247,456 marked passages. Both held works were reserved in full within the Kyoto training release. The visual writer learned its own contextual features and output pixels. Panel d is the first incomplete pair in reserved order. Source-given gloss identity can differ from philological correctness.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Visual_word_senses.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_119_visual_word_senses.png'),new Uint8Array(await png.arrayBuffer()));
