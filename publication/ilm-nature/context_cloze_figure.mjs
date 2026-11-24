import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src=path.join(root,'references/evidence/context_cloze_v1');
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
text('Learning the word between two pieces of ink',48,17,1410,65,42,true);
text('Held English passages and questions · 896,983 parameters · one RTX 4090 D',53,89,1390,54,25,false,teal);
text('a   Can local ink predict the missing word?',50,156,1400,50,30,true);
const chart=slide.charts.add('bar',{barOptions:{direction:'column',grouping:'clustered'},
 position:{left:47,top:245,width:827,height:269},categories:['Two words on each side','Two words on the left'],
 hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:17}},
 series:['Visual network','Count backoff','Unigram'].map((name,i)=>({name,values:stats.series[i],fill:['#197B79','#687C91','#C5AC87'][i]})),
 xAxis:{textStyle:{fontSize:17}},yAxis:{min:0,max:15,majorUnit:5,textStyle:{fontSize:17},majorGridlines:{fill:'#DCE5E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('Exact word image (%) · all 50,012 positions',60,215,705,39,18,false,teal);
text('A new combination of neighbours',897,238,545,55,25,true,teal);
text('The four flanking words had never occurred together in sampled training windows. The count predictor chose “to”.',897,309,506,149,23);
text('Observed',60,541,214,42,20,false,teal);await bitmap('left',265,531,366,32);
text('Prediction',667,541,195,42,20,true,teal);await bitmap('gap',844,531,172,34);
text('Observed',1042,541,180,42,20,false,teal);await bitmap('right',1194,531,242,32);
text('b   A correct isolated word does not guarantee a complete answer',50,628,1410,50,30,true);
table(stats.table,700,176,[370,334,313,368]);
text('What had he decided about the doctor earlier?',56,917,766,52,23,true);
await bitmap('answer',834,915,583,31);
text('Why is her life memorable?',56,992,766,52,23,true);await bitmap('miss',834,990,583,31);
text('Reference: she is the author of many books',832,1036,595,44,18,false,teal);
text('c   Eight forward steps from the first reserved prefix',50,1112,1380,55,30,true);
text('Prefix',58,1197,174,42,21,false,teal);await bitmap('prefix',230,1188,563,34);
text('Output',58,1261,174,42,21,true,teal);await bitmap('continuation',230,1254,1175,35);
text('Across 147 prompts the writer produced 36 distinct eight-word streams. No stream matched its full reference.',55,1372,1380,67,23);
slide.speakerNotes.textFrame.setText('Every image strip comes from the physical inputs or the released model. Copy and spelling compete with retrieval during QA. In panel b the first two numeric columns supply the other reference words. The final column uses a blank answer strip. Panel c uses a user-fixed length and feeds the emitted image back to the local predictor.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Context_cloze.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_134_context_cloze.png'),new Uint8Array(await png.arrayBuffer()));
