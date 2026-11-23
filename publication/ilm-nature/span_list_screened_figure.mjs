import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/span_list_screened_20261124');
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


text('Reading beyond the first answer strip',50,23,1410,68,43,true);
text('a   Exact answers on 359 screened development questions',50,105,1410,52,29,true);
const chart=slide.charts.add('bar',{barOptions:{direction:'bar',grouping:'clustered'},
 position:{left:45,top:169,width:865,height:338},categories:['Shuffled labels','Trained, depth 1','Generic reader','Trained, depth 8','Available at depth 8'],
 hasLegend:false,series:[{name:'Questions',values:stats.series[0],fill:'#367c80'}],
 xAxis:{textStyle:{fontSize:18}},yAxis:{min:0,max:359,majorUnit:50,textStyle:{fontSize:17},majorGridlines:{fill:'#DFE6E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('What widening changes',950,185,472,56,28,true,teal);
text('The correct strip was available on 312 questions. The trained reader selected it on 234. Against the generic reader, 16 replies became exact and 24 lost an exact match.',950,250,478,218,24);
text('b   Complete replies after changing the requested word order',50,530,1420,48,29,true);
table(stats.table,593,212,[393,159,227,305,301]);
text('c   A recovered second item and an unnecessary title',50,848,1410,50,29,true);
async function strip(name,x,y,maxWidth,maxHeight){
 const blob=await fs.readFile(path.join(src,name+'.png'));
 const w=blob.readUInt32BE(16),h=blob.readUInt32BE(20),height=Math.min(maxHeight,maxWidth*h/w);
 await image(name,x,y,w*height/h,height);
}
async function comparison(tag,y){
 text('Question',56,y,220,45,22,true,teal);
 await strip(tag+'_question',279,y,1140,42);
 for(const [label,suffix,x,width] of [['Generic reader','generic',60,338],['Eight-span reader','answer',440,478],['Human reference','reference',960,472]]){
   text(label,x,y+62,width,39,22,true,teal);
   await strip(tag+'_'+suffix,x,y+109,width,40);
 }
}
await comparison('gain',933);
await comparison('loss',1128);
text('The first question requires two game modes. The second reply names the correct person but adds her professional title, which is absent from the reference.',55,1322,1370,92,24);
slide.speakerNotes.textFrame.setText('Chinese strips preserve original pixels. Panel c uses the first recovered answer absent at depth one and the first loss against the generic reader, in the fixed screened order.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Span_list.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_97_span_list.png'),new Uint8Array(await png.arrayBuffer()));
