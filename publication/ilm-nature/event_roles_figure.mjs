import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/event_roles_20261119');
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
text('Writing a modern sentence with unfamiliar bronze glyphs',44,18,1420,65,39,true);
text('a   Actual model outputs with the historical forms of Qin and Chu',44,104,1415,50,29,true);
text('Input image',65,177,242,42,23,true,teal);
text('Autonomous answer image',385,177,570,42,23,true,teal);
text('Reading of the answer',961,177,470,42,23,true,teal);
for(const [r,[i,meaning]] of [[4,'Qin surrounds Chu.'],[5,'Chu surrounds Qin.'],[0,'Qin attacks Chu.'],[2,'Qin, Qin comes to the aid of Chu.']].entries()){
 const y=240+r*68;
 for(const [name,x] of [['source',66],['context',387]]){
  const k='predicate_'+i+'_'+name;await image(k,x,y,stats.widths[k]*38,38);
 }
 text(meaning,962,y+1,460,54,23);
}
text('b   Faithful translations in both participant orders',44,545,1415,52,30,true);
table(stats.table,611,280,[417,336,342,290]);
text('Role scores require two complete modern answers. Broad edits use 1,024 separate passages. Dashes indicate comparisons without a corresponding reading or measurement.',65,916,1370,76,22);
text('c   Exact reproduction of the two historical names in both directions',44,1022,1410,52,29,true);
const chart=slide.charts.add('bar',{position:{left:46,top:1090,width:785,height:337},categories:['Source copy','Phrase composer','Exchange writer','Frequency aliases','Contextual aliases'],hasLegend:true,
 barOptions:{direction:'bar',grouping:'clustered'},series:[{name:'Both names in order',values:stats.series[0],fill:'#849FB1'},{name:'Both names + rewritten answers',values:stats.series[1],fill:teal}],
 xAxis:{textStyle:{typeface:'Arial',fontSize:17}},yAxis:{min:0,max:20,majorUnit:5,textStyle:{typeface:'Arial',fontSize:17},majorGridlines:{fill:'#DBE6E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('What counts as a preserved name?',879,1085,538,65,24,true,teal);
text(`Each supplied bitmap must occur exactly once in the intended order. In ${stats.series[1][4]} of the 20 pairs the contextual writer preserved both names and changed both answers. It produced two faithful pairs among eight read for meaning.`,879,1161,535,230,23);
slide.speakerNotes.textFrame.setText('Original 32 by 32 ink from Hanziyuan bronze exemplars B11149 (Qin) and B08953 (Chu). The final output repeats the rescuer. The table compares paired masked readings rather than isolated glyph hits. The tested mixed-script sentences are experimental substitutions.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Event_roles.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_63_event_roles.png'),new Uint8Array(await png.arrayBuffer()));
