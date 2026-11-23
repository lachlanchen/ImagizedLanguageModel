import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src=path.join(root,'references/evidence/optical_role_adapter_20261124');
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
text('Reading two people’s roles from images of classical prose',50,20,1420,65,41,true);
text('a   Correct unrestricted answer images across three classical works (%)',50,100,1420,52,30,true);
const colors=['#A3AEB8','#BB734A','#347C80'];
const captions=['Original optical reader','Independent names','Assigned names'];
const chart=slide.charts.add('bar',{barOptions:{direction:'column',grouping:'clustered'},
 position:{left:62,top:159,width:943,height:330},categories:['Guoyu · zh','Guoyu · en','Zuo · zh','Zuo · en','Shishuo · zh','Shishuo · en'],
 hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:17}},
 series:captions.map((name,i)=>({name,values:stats.series[i],fill:colors[i]})),
 xAxis:{textStyle:{fontSize:17}},yAxis:{min:0,max:Math.max(40,10*Math.ceil(Math.max(...stats.series.flat())/10)),majorUnit:10,textStyle:{fontSize:17},majorGridlines:{fill:'#DFE6E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('What learned?',1040,190,360,60,28,true,teal);
text('50,690 weights were updated in an 86-million-parameter reader. Both trials used the same modern questions and the same pictured classical passages.',1040,250,360,217,23);
text('b   Complete four-response contrasts by work and question language',50,505,1410,52,29,true);
table(stats.table,576,300,[368,248,252,249,268]);
text('c   The first reserved Guoyu excerpt and the reader’s actual responses',50,905,1400,54,29,true);
text('As printed',60,964,255,43,21,true,teal);
await bitmap('page0',325,964,1080,32);
text('Names exchanged',60,1009,255,43,21,true,teal);
await bitmap('page1',325,1009,1080,32);
text('Page',60,1072,245,43,21,true,teal);
text('Pictured question',325,1072,560,43,21,true,teal);
text('Source answer',931,1072,232,43,21,true,teal);
text('Reader output',1190,1072,260,43,21,true,teal);
for(let i=0;i<stats.strips.length;i++){
 const r=stats.strips[i],y=1126+54*i;
 text(r.work,60,y,255,49,21,true);
 await bitmap(r.base+'_q',325,y,580,36);
 await bitmap(r.base+'_gold',931,y,230,36);
 await bitmap(r.base+'_out',1190,y,250,36);
}
text('The edited page is a counterfactual reading test. Every output is the original image strip selected by the model.',60,1382,1370,67,20);
slide.speakerNotes.textFrame.setText('Equal-size models read each pictured question independently. The top panel separates the three classical works and two question languages. The table scores all four replies for a full classical contrast.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Optical_role_adapter.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_116_optical_role_adapter.png'),new Uint8Array(await png.arrayBuffer()));
