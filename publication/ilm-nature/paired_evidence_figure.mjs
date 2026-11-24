import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src=path.join(root,'references/evidence/paired_evidence_v1');
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
text('Two questions about the same pictured passage',50,20,1410,72,46,true);
text('a   What the optical reader wrote',55,111,1390,48,29,true);
text('Question image',70,182,580,43,23,true,teal);
text('Answer image',830,182,570,43,23,true,teal);
for (const [i,name] of ['name','class','place'].entries()){
 const y=244+i*108;
 await bitmap(name+'-query',70,y,670,57);
 await bitmap(name+'-answer',830,y,590,57);
}
text('The first two questions ask for a fish’s common name and classification. Both answers match. The third asks where a dinosaur was found. The writer returned a two-character fragment. A reference answer is Niger.',55,555,1390,88,23);
text('b   Passages on which both answers were written correctly',55,696,1390,48,29,true);
const chart=slide.charts.add('bar',{barOptions:{direction:'column',grouping:'clustered'},
 position:{left:63,top:787,width:920,height:275},categories:['English dialogue','English articles','Simplified Chinese','Traditional Chinese'],
 hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:15}},
 series:['Previous optical','Optical','Optical + evidence','Bytes + evidence'].map((name,i)=>({name,values:stats.series[i],fill:['#ACB7BB','#80ABAD','#146D72','#B48460'][i]})),
 xAxis:{textStyle:{fontSize:15}},yAxis:{min:0,max:32,majorUnit:8,textStyle:{fontSize:16},majorGridlines:{fill:'#DCE5E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('Passages out of 64 per corpus',70,754,850,36,19,false,teal);
text('0 of 256',1040,800,350,70,48,true,teal);
text('Passages with both answers correct when the pictured question was erased from the optical evidence reader.',1040,882,350,155,22);
text('c   Complete answers and sensitivity to changed facts',55,1090,1390,50,29,true);
table(stats.table,1160,200,[453,264,261,407]);
text('For the fact edit, the question stays fixed while one numeral in the source image changes. Both versions must be answered correctly.',55,1390,1390,70,20);
slide.speakerNotes.textFrame.setText('The first both-correct and first incomplete simplified Chinese pairs in the frozen order supply the displayed examples. Native chart and table count complete stopped responses. Evidence locations were given only during fitting.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Paired_evidence.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_145_paired_evidence.png'),new Uint8Array(await png.arrayBuffer()));
