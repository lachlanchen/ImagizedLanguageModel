import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/conditional_ink_20261118');
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
text('Do adjacent ink patterns improve phrase choice?',44,12,1410,65,42,true);
text('a   Two independently generated Chinese insertions',44,86,1430,48,30,true);
text('New as a complete insertion',355,149,540,40,23,true,teal);
text('An extra glyph at the boundary',908,149,550,40,23,true,teal);
for(const [i,[field,label]] of [['source','Classical sentence'],['prefix','Modern prefix, last 14'],['marginal','Existing visual writer'],['prediction','Adjacent patterns'],['reference','Recorded insertion']].entries()){
  const y=210+i*49;
  text(label,64,y,279,38,23,true);
  for(const [name,x] of [['novel',355],['failed',908]]){
    const key=name+'_'+field;
    await image(key,x,y,stats.widths[key]*26,26);
  }
}
text('At right, the following modern text already begins with the extra glyph. Both models receive the boundary and full prefix.',62,464,1390,62,22);
text('b   Exact insertions, with a learned stop',44,548,1420,45,30,true);
table(stats.table,615,250,[395,245,245,255,245]);
text('c   Does the wider sentence favor its recorded phrase?',44,922,1420,46,30,true);
const chart=slide.charts.add('bar',{position:{left:50,top:1001,width:845,height:270},
 categories:['Discovery','Replication','Previous holdout','New holdout'],hasLegend:true,legend:{position:'bottom'},
 barOptions:{direction:'column',grouping:'clustered'},series:[
 {name:'Authentic assignment',values:stats.exchange_counts[0],fill:teal},
 {name:'Exchanged assignment',values:stats.exchange_counts[1],fill:'#8495A5'}],
 xAxis:{textStyle:{typeface:'Arial',fontSize:17}},yAxis:{min:0,max:150,majorUnit:30,
 textStyle:{typeface:'Arial',fontSize:18},majorGridlines:{fill:'#DCE4E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('Natural sentence pairs',920,1014,505,48,26,true,teal);
text('The flanking source glyphs and prior modern glyphs match. Bars count pairs favoring each assignment by complete phrase probability. Two pairs tie across the four cohorts.',920,1082,514,150,22);
slide.speakerNotes.textFrame.setText('Panels show original PNGs from the scored streams, including the previous holdout’s only novel exact insertion and the first bounded new-cohort mismatch for which local counts succeed. Full inputs and control outputs are archived. Table and chart are derived from full cohort records and matched sentence exchanges.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Conditional_ink.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,12382500','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_54_conditional_ink.png'),new Uint8Array(await png.arrayBuffer()));
