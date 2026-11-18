import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/anchor_gap_20261119');
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
text('Copying source glyphs during visual translation',44,18,1410,64,40,true);
text('a   Four complete translations from held-out chapters',44,100,1410,52,29,true);
for (const [x,title] of [[60,'Classical source'],[525,'Neural gap writer'],[991,'Finite gap writer']]) text(title,x,162,442,44,24,true,teal);
for(const [i,tag] of [45,184,6,105].entries()){
  for(const [suffix,x] of [['source',60],['learned',525],['finite',991]]){
    const key=tag+'_'+suffix;
    const scale=Math.min(31,455/stats.widths[key]);
    await image(key,x,219+i*62,stats.widths[key]*scale,scale);
  }
}
text('The neural writer conveys the neighbouring state’s refusal in the first row. In the third it turns an evening meal into an act of writing. Each strip contains the complete emitted ink.',60,475,1370,78,22);
text('b   Agreement with the reference and preservation of meaning',44,586,1420,52,29,true);
table(stats.table,647,288,[412,235,220,250,268]);
text('¹ Reference-derived source positions supplied for diagnosis.  ² Exploratory comparison on the same passages. A dash denotes an algorithm with a finite source traversal.',60,949,1390,76,21);
text('c   Glyph edits by source length',44,1052,1410,53,29,true);
const colors=['#A54734',teal,'#5079A1'];
const chart=slide.charts.add('bar',{position:{left:45,top:1112,width:850,height:320},categories:['3–12','13–32','33–64','65–96'],hasLegend:true,legend:{position:'bottom'},
 barOptions:{direction:'column',grouping:'clustered'},series:['Neural gaps','Composer','Finite gaps'].map((name,i)=>({name,values:stats.series[i],fill:colors[i]})),
 xAxis:{textStyle:{typeface:'Arial',fontSize:16}},yAxis:{min:0,max:16000,majorUnit:4000,textStyle:{typeface:'Arial',fontSize:15},majorGridlines:{fill:'#DBE6E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('128 passages per length range',935,1125,490,64,24,true,teal);
text('The source planner found 8,512 of 10,085 reference copy positions. Providing all the correct positions still led the neural writer to exhaust its length limit on 147 passages.',935,1200,497,200,22);
slide.speakerNotes.textFrame.setText('The 24 meaning judgments came from a method-masked model reading in the neural comparison. New finite-gap answers received an identified follow-up reading. All four sets of strips are saved model output or original source.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Anchor_gap.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_66_anchor_gap.png'),new Uint8Array(await png.arrayBuffer()));
