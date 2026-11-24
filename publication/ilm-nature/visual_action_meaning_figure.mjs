import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src=path.join(root,'references/evidence/visual_action_meaning_v1');
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
text('Reading an action described in other words',48,15,1420,63,42,true);
text('The page contains two Chinese lines. Each question asks for the object of one action.',54,89,1400,52,24,false,teal);
text('a   Pages with both objects located and written correctly',49,162,1410,54,30,true);
const chart=slide.charts.add('bar',{barOptions:{direction:'column',grouping:'clustered'},
 position:{left:50,top:220,width:900,height:335},categories:['Alternate names (188)','Definitions (268)','Multiword only (241)'],
 hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:16}},
 series:['Parent','Aligned','Exchanged'].map((name,i)=>({name,values:stats.series[i],fill:['#8395A1','#16686B','#BE906D'][i]})),
 xAxis:{textStyle:{fontSize:16}},yAxis:{min:0,max:50,majorUnit:10,textStyle:{fontSize:16},majorGridlines:{fill:'#DCE5E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('What counts as a success?',985,244,447,70,24,true,teal);
text('Both actions must elicit the correct stopped word and object position under both line orders and two question frames.',985,321,432,175,23);
text('b   Correct line selection can still yield a wrong word',49,595,1425,50,30,true);
for(const [label,x] of [['Screened source page',52],['Description in the question',487],['Actual output',1042]])text(label,x,680,540,47,22,true,teal);
await bitmap('pu-page',54,742,345,145);
text('give admittance to',491,749,490,61,26);text('seize with the hands',491,839,490,62,26);
await bitmap('pu-answer-0',1043,749,367,57);await bitmap('pu-answer-1',1043,839,367,57);
text('The second line asks someone to fetch a peach. The expected object is peach. With the object position supplied, the same writer produced peach.',55,937,1375,78,23);
text('c   Complete written answers on the original and literary pages',49,1051,1415,56,29,true);
table(stats.table,1129,193,[645,240,250,250]);
text('Both lines of the successful Tang pair occurred in training. All ten Pu Songling pairs passed the overlap screen.',56,1365,1382,77,22);
slide.speakerNotes.textFrame.setText('The three bars count pages, with denominators shown in parentheses. Multiword questions are a subset of the definitions benchmark. Fresh Pu Songling answers were checked from 480 physical outputs and each case counted under both question frames and row orders. The illustrated case is pair nine.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Visual_action_meaning.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_125_visual_action_meaning.png'),new Uint8Array(await png.arrayBuffer()));
