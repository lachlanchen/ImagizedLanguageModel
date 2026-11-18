import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/ordered_neighborhood_20261118');
const stats = JSON.parse(await fs.readFile(path.join(src,'figure.json')));

const deck = Presentation.create({slideSize:{width:1500,height:1120}});
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
    cell.text.style={typeface:'Arial',fontSize:23,bold:i===0,color:j===1?teal:ink};
  }
}
text('Writing familiar glyphs in new combinations',44,20,1410,65,42,true);
text('a   Exact expression writing on 26,213 held-out passages',44,112,1410,55,31,true);
table(stats.table,176,254,[315,185,190,200,495]);
text('b   Whole expressions gained and lost by adding the local view',44,458,1410,50,31,true);
for(const [label,x,w] of [['Case',55,150],['Before the gap',234,330],['After the gap',639,330],['Output',1034,175],['Reference',1264,175]])text(label,x,530,w,42,24,true);
for(const [i,[key,label]] of [['gain','Gained'],['loss','Lost']].entries()){
  const y=596+i*82;text(label,55,y-2,160,42,24);
  for(const [part,x,w] of [['before',234,338],['after',639,338],['actual',1034,140],['reference',1264,140]])await image(key+'_'+part,x,y,w,34);
}
const positions=JSON.parse(await fs.readFile(path.join(src,'position_counts.json')));
const boundary=JSON.parse(await fs.readFile(path.join(src,'boundary_reset.json')));
const rust='#A64F39';
function curve(left, title, data, legend){
  text(title,left,773,716,45,27,true);
  text(legend[0],left+5,824,370,35,21,true,teal);
  text(legend[1],left+374,824,335,35,21,true,rust);
  const series=data.map(([name,values],i)=>({name,values,line:{fill:i?rust:teal,width:5}}));
  const chart=slide.charts.add('line',{position:{left,top:872,width:690,height:205},series,
    categories:['1 glyph','2 glyphs','3 glyphs','4 glyphs'],hasLegend:false,lineOptions:{smooth:false},
    xAxis:{textStyle:{typeface:'Arial',fontSize:19}},yAxis:{min:0,max:4000,majorUnit:1000,
      textStyle:{typeface:'Arial',fontSize:18},majorGridlines:{fill:'#DCE4E6',width:1}}});
  applyPresentationChartFont(chart,{fontFamily:'Arial'});
}
curve(44,'c   Copying supplied glyphs',[
 ['Training expressions',positions.aligned.observed.prefix_exact],['New combinations',positions.aligned.mixed.prefix_exact]
],['Training expressions','New combinations']);
curve(760,'d   Drawing history on new combinations',[
 ['Reset',boundary.splits.mixed.reset.prefix_exact],['Carry',boundary.splits.mixed.carry.prefix_exact]
],['Reset at each glyph','Carry history']);
slide.speakerNotes.textFrame.setText('Panel a uses independent passage input. Panel b shows the first source-order gain and loss against the uniform reader and the actual emitted glyphs. Panel c supplies known target images to a separate shared control network. Its unseen combinations reuse the same position-specific glyph inventory. Prefix counts come from stored emitted pixels; complete answers also require the correct length and a stop. Panel d imposes four output positions on both frozen decoding arms and compares the effect of clearing physical writing history.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Ordered_neighborhood.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,10668000','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_45_ordered_neighborhood.png'),new Uint8Array(await png.arrayBuffer()));
