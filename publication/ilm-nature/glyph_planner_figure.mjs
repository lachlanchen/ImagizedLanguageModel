import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/glyph_planner_20261118');
const stats = JSON.parse(await fs.readFile(path.join(src,'figure.json')));

const deck = Presentation.create({slideSize:{width:1500,height:1300}});
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
text('Choosing what to draw from a pictured passage',44,12,1410,65,42,true);
text('a   Two recurrent states',44,91,1400,45,30,true);
text('Passage and previous output  →  glyph plan  →  48 ink measurements  →  32 × 32 image',54,144,1400,45,27);
text('The passage state persists. Each new drawing starts with its own plan.',54,196,1400,36,22);
text('b   Complete expressions written exactly',44,252,1410,48,30,true);
table(stats.table,314,324,[697,344,344]);
text('c   First gain and loss against the previous uniform writer',44,652,1410,48,30,true);
for(const [label,x,w] of [['Case',55,150],['Before the gap',234,330],['After the gap',639,330],['Output',1034,175],['Reference',1264,175]])text(label,x,715,w,38,23,true);
for(const [i,[key,label]] of [['gain','Gained'],['loss','Lost']].entries()){
  const y=769+i*65;text(label,55,y-2,160,42,23);
  for(const [part,x,w] of [['before',234,338],['after',639,338],['actual',1034,140],['reference',1264,140]])await image(key+'_'+part,x,y,w,34);
}
text('d   Correct prefixes on the first test',44,905,1410,54,30,true);
const main=JSON.parse(await fs.readFile(path.join(src,'evaluation.json'))).arms;
const wide=JSON.parse(await fs.readFile(path.join(src,'search.json'))).arms.aligned;
const data=[['Previous uniform writer',main.uniform.prefix_exact,'#657680'],
  ['Planner, one history',main.aligned.prefix_exact,'#A64F39'],['Planner, four histories',wide.prefix_exact,teal]];
const chart=slide.charts.add('line',{position:{left:60,top:973,width:925,height:276},
  categories:['1 glyph','2 glyphs','3 glyphs','4 glyphs'],hasLegend:false,lineOptions:{smooth:false},
  series:data.map(([name,values,color])=>({name,values,line:{fill:color,width:5}})),
  xAxis:{textStyle:{typeface:'Arial',fontSize:21}},yAxis:{min:0,max:1500,majorUnit:300,
    textStyle:{typeface:'Arial',fontSize:20},majorGridlines:{fill:'#DCE4E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
for(const [i,[name,values,color]] of data.entries()){
  text(name,1010,995+74*i,450,41,25,true,color);
}
slide.speakerNotes.textFrame.setText('The planner reads the landmarks of its actual prior image. The local writer resets each glyph. Counts in the table are complete stopped answers. Blank entries indicate conditions that were not run on that split. Training receives reference images and, only in the final condition, teacher probabilities. Neither is available during free generation. The first gain and loss show the original aligned local-commitment outputs. The prefix curves score the same 17,686 passages.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Glyph_planner.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,12382500','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_46_glyph_planner.png'),new Uint8Array(await png.arrayBuffer()));
