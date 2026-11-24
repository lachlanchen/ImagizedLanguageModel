import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src=path.join(root,'references/evidence/dialogue_efficiency_v1');
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
text('Learning curves for glyphs and bytes',48,16,1410,66,43,true);
text('The same 73,714 training turns, the same 8,000 batches and one RTX 4090 D.',54,92,1395,55,25,false,teal);
text('a   A controlled comparison from random initialization',49,172,1400,53,30,true);
const chart=slide.charts.add('line',{position:{left:45,top:236,width:888,height:405},
 categories:['100','800','2,400','8,000'],hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:19}},
 series:['Glyph images','Byte tokens'].map((name,i)=>({name,values:stats.series[i],line:{fill:['#17676A','#BD6843'][i],width:4},marker:{symbol:i?'square':'circle',size:8}})),
 xAxis:{textStyle:{fontSize:18}},yAxis:{min:0,max:40,majorUnit:10,textStyle:{fontSize:18},majorGridlines:{fill:'#DCE5E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('Development exact match (%)',953,259,494,65,25,true,teal);
text('Only the observed letter representation changes. Both systems have identical word boundaries, answer actions, parameter counts and optimizer batches.',955,346,480,211,24);
text('Training updates (four saved checkpoints)',67,663,873,41,21,false,teal);
text('b   Whole answers and measured energy',49,762,1400,54,30,true);
table(stats.table,842,290,[420,327,343,295]);
text('Fit energy covers dialogue updates. The inherited writer also depends on earlier supervised training. The full reference allows four independently written answers per development turn.',54,1178,1380,97,23);
text('c   A diagnostic that can be checked on every answer',49,1310,1400,52,30,true);
text('Across 20,002 visual replies, the word-by-word and stop checks agreed with every autonomous exact-image decision.',54,1375,1390,77,23,false,teal);
slide.speakerNotes.textFrame.setText('The horizontal axis gives the four measured checkpoints as categories at equal spacing. The table uses all 5,588 eligible development turns, first-reference exact matches and official leave-one-out word F1. Device preparation energy is included separately in the matched target-quality analysis. One seed per system.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Dialogue_efficiency.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_129_dialogue_efficiency.png'),new Uint8Array(await png.arrayBuffer()));
