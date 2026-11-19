import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/visual_role_writer_20261119');
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
text('Learning who a pictured statement describes',50,22,1450,65,39,true);
text('a   Both answers correct after exchanging the participants (%)',50,103,1450,50,29,true);
const chart=slide.charts.add('bar',{barOptions:{direction:'bar',grouping:'clustered'},position:{left:40,top:172,width:836,height:315},
 categories:['Held-out predicates','Earlier situations','New situations'],hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:17}},
 series:[{name:'Randomized roles',values:stats.series[0],fill:'#86ABA9'},{name:'Matched roles',values:stats.series[1],fill:teal}],
 xAxis:{textStyle:{fontSize:20}},yAxis:{min:0,max:100,textStyle:{fontSize:19},majorGridlines:{fill:'#DBE6E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('The same visible words',913,172,514,70,27,true,teal);
text('Two clauses assign opposite predicates to different participants. Exchanging the names reverses the correct answer to an unchanged question.',913,247,490,156,25);
text('Held-out predicates use new names and complements in familiar layouts.',913,412,480,78,21);
text('b   Complete autonomous responses and natural inference',50,515,1450,52,29,true);
table(stats.table,576,225,[365,260,240,260,260]);
text('Reordered statements place the relevant name after the predicate. Novel layouts use an unfamiliar request and a two-sentence passage.',60,818,1390,75,22);
text('c   The same pictured request about a bronze name',50,929,1450,52,29,true);
text('Pictured request (both rows)',62,1002,455,52,24,true);
const [qw,qh]=stats.dimensions.bronze0_query;
await image('bronze0_query',535,1000,Math.min(qw,850),Math.min(qw,850)*qh/qw);
text('Pictured passage',62,1080,680,52,24,true);
text('Complete response',805,1080,600,52,24,true);
for(const i of [0,1]) {
 const y=1150+90*i;
 for(const [field,x,space] of [['premise',62,680],['answer',805,610]]){
   const name='bronze'+i+'_'+field, [w,h]=stats.dimensions[name], scale=Math.min(1.55,space/w,64/h);
   await image(name,x,y,w*scale,h*scale);
 }
}
text('Two authored counterfactual passages use the Qin and Chu bronze images. Each row shows the actual model emission. The inscriptions identify the participants in this reading task.',62,1339,1380,100,22);
slide.speakerNotes.textFrame.setText('The reports and original PNGs bind the native chart, table and answer images to the complete final checkpoints.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Visual_role_writer.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_84_visual_role_writer.png'),new Uint8Array(await png.arrayBuffer()));
