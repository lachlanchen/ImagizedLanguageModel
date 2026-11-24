import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src=path.join(root,'references/evidence/visual_discourse_v1');
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
text('Writing an answer from a pictured prompt',50,18,1410,72,46,true);
text('a   Two held instructions and their actual output',55,111,1390,48,29,true);
text('English prompt',55,180,270,42,23,true,teal);
await bitmap('en-input',360,175,1000,55);
text('Written response',55,255,285,43,23,true,teal);
await bitmap('en-answer',360,248,1040,97);
text('Chinese prompt',55,375,280,41,23,true,teal);
await bitmap('zh-input',360,365,950,52);
text('Written response',55,456,290,42,23,true,teal);
await bitmap('zh-answer',360,447,1050,62);
text('The English prompt asks how to conserve energy. The Chinese prompt asks for a five-word alliteration.',55,548,1400,60,23);
text('b   Predicting the next printed glyph on held pages',55,655,1390,52,29,true);
const chart=slide.charts.add('bar',{barOptions:{direction:'column',grouping:'clustered'},
 position:{left:62,top:746,width:881,height:283},categories:['English fiction','Chinese fiction','English instructions','Chinese instructions'],
 hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:15}},
 series:['Initial','Trained','Trained, crossed prompt'].map((name,i)=>({name,values:stats.series[i],fill:['#ABB7BB','#157478','#BE946C'][i]})),
 xAxis:{textStyle:{fontSize:15}},yAxis:{min:0,max:12,majorUnit:3,textStyle:{fontSize:16},majorGridlines:{fill:'#DCE5E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('Bits per observed glyph (lower is better)',70,714,850,36,19,false,teal);
text('0 of 120',986,760,420,65,49,true,teal);
text('On external Winograd pairs the trained writer never wrote both changed answers correctly. Choosing between two supplied answers by likelihood succeeded on 13 pairs.',986,838,420,181,22);
text('c   Answering two human questions per unseen passage',55,1084,1390,50,29,true);
table(stats.table,1160,205,[440,290,325,330]);
text('Each cell counts individually correct stopped answers among 128 questions. Each source contributed 64 passages. The writer receives no candidate answers.',55,1381,1390,74,20);
slide.speakerNotes.textFrame.setText('Panel a uses the first held instructions in fixed hash order and the paired optical writer. Only blank sidebearings and line wrapping change for display. Panel b supplies actual preceding answer cells for scoring. Panel c compares free running responses.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Visual_discourse.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_144_visual_discourse.png'),new Uint8Array(await png.arrayBuffer()));
