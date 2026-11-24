import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src=path.join(root,'references/evidence/word_canvas_v1');
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
text('Writing from a blank answer strip',48,18,1420,60,43,true);
text('A pictured passage and question determine the strip length. The writer fills word images in four passes.',54,88,1390,66,23,false,teal);
text('a   Answer quality on 200 held conversational passages',49,167,1410,52,30,true);
const chart=slide.charts.add('bar',{barOptions:{direction:'column',grouping:'clustered'},
 position:{left:48,top:234,width:845,height:315},categories:['List-trained parent','Causal continuation','Masked continuation'],
 hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:17}},
 series:['All turns (3,155)','Absent wording (443)'].map((name,i)=>({name,values:stats.series[i],fill:['#577B91','#18766F'][i]})),
 xAxis:{textStyle:{fontSize:18}},yAxis:{min:0,max:40,majorUnit:10,textStyle:{fontSize:17},majorGridlines:{fill:'#DCE5E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('Mean word F1 (%)',62,217,260,33,16,false,teal);
text('Correct length, wrong words',930,245,508,53,26,true,teal);
text('When given the reference length, the visual writer still missed 434 of 435 difficult answers. An incorrect word image was already fixed on the first pass in 393 cases.',930,314,478,200,24);
text('b   Reading the words the model wrote',49,573,1395,50,30,true);
text('Who is a senior researcher?',55,641,1350,46,24,true);
text('Pass 1',55,704,180,45,22,false,teal);await bitmap('first-pass',255,695,955,38);
text('Pass 3',55,759,180,45,22,false,teal);await bitmap('draft',255,751,955,38);
text('Which led to what?',55,837,700,46,24,true);
text('Masked writer',55,903,210,44,22,false,teal);await bitmap('miss',290,898,590,35);
text('Reference: breaking his back',846,900,553,54,22);
text('How did he die?',55,974,700,48,24,true);
text('Causal writer',55,1038,210,44,22,false,teal);await bitmap('causal',290,1030,755,35);
text('c   Is the reference among the generated proposals?',49,1112,1410,55,30,true);
table(stats.table,1183,182,[500,202,202,202,279]);
text('All 279 references require new phrasing whose normalized words can be assembled from visible inputs.',54,1413,1400,40,20);
slide.speakerNotes.textFrame.setText('The first source-sensitive ordinary multiword success, first difficult masked miss and first difficult source-sensitive causal success are shown with actual replayed output pixels. Empty areas on the first pass are aligned with the final answer for display. Search proceeds without candidates from the reference. Counts are normalized and include completion.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Word_canvas.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_132_word_canvas.png'),new Uint8Array(await png.arrayBuffer()));
