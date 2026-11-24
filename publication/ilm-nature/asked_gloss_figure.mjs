import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src=path.join(root,'references/evidence/asked_gloss_v1');
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
text('Reading the order of a pictured question',50,16,1420,64,43,true);
text('Learn the source position from the probability of an English answer image at each Chinese glyph.',55,88,1390,56,23,false,teal);
text('a   Both words located and glossed across reversed instructions',50,155,1410,52,29,true);
const labels=['Local pooling','Ordered question','Exchanged answers'];
const colors=['#AD694B','#276E73','#8393A1'];
const chart=slide.charts.add('bar',{barOptions:{direction:'column',grouping:'clustered'},
 position:{left:52,top:209,width:882,height:295},categories:['Chinese · 4 requests','English · 4 requests','Both · 8 requests'],
 hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:17}},
 series:labels.map((name,i)=>({name,values:stats.series[i],fill:colors[i]})),xAxis:{textStyle:{fontSize:17}},
 yAxis:{min:0,max:300,majorUnit:100,textStyle:{fontSize:17},majorGridlines:{fill:'#DCE5E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('539 held clauses',997,223,450,56,26,true,teal);
text('Each request names both roles. Swapping their places leaves the counts of every three-glyph window unchanged. Both controls scored zero on the four- and eight-request tests.',997,281,428,200,22);
text('b   Complete passes by source',50,524,1400,50,28,true);
table(stats.table,586,235,[311,136,280,326,332]);
text('¹ Both answers accepted with the source positions supplied and the training gloss bank available.',61,835,1380,43,20);
text('c   Two requests about one unmarked passage',50,910,1400,50,28,true);
await bitmap('hold_page',62,974,168,45);
text('Pictured instructions',306,965,720,40,21,true,teal);
text('Written reply',1130,965,300,40,21,true,teal);
for(let i=0;i<2;i++){await bitmap('order_question'+i,292,1013+i*62,805,46);await bitmap('hold_free'+i,1124,1013+i*62,295,46);}
text('d   A word bank can repair the spelling and still miss the meaning',50,1160,1410,50,28,true);
for(const [txt,x,w] of [['Passage',61,250],['Free writing',341,333],['Attested gloss',743,329],['Reference reading',1100,355]])text(txt,x,1225,w,40,20,true,teal);
for(const [i,name,ref] of [[0,'male','male'],[1,'female','female']]){
 const y=1280+i*67;await bitmap(name+'_page',59,y,234,45);await bitmap(name+'_free1',332,y,344,45);await bitmap(name+'_bank1',734,y,327,45);text(ref,1101,y,320,50,24);
}
slide.speakerNotes.textFrame.setText('Counts are complete judged source-position and stopped image pairs. Panels c and d use actual frozen outputs. The pictured Chinese queries have identical local windows and request opposite roles. The two images in panel d compare the same model at the same location with independent decoding choices. The required reading for 雌 was female or feminine; turn remains incorrect.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Asked_gloss.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_120_asked_gloss.png'),new Uint8Array(await png.arrayBuffer()));
