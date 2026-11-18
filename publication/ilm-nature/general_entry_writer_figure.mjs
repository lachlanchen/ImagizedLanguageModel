import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/general_entry_writer_20261119');
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
text('Writing a lexical fact and a historical form',50,20,1450,65,40,true);
text('a   Whose historical shape did the writer put in its answer?',50,108,1445,55,29,true);
const chart=slide.charts.add('bar',{barOptions:{direction:'bar',grouping:'clustered'},
 position:{left:45,top:174,width:860,height:270},categories:['Requested entry and stage','Other entry, requested stage','Other or unfinished'],hasLegend:false,
 series:[{name:'Responses',values:stats.series[0],fill:teal}],xAxis:{textStyle:{fontSize:20}},yAxis:{min:0,textStyle:{fontSize:20},majorGridlines:{fill:'#DBE6E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('Two entries on each card',960,183,485,50,26,true,teal);
text('All 1,136 responses start with the correct headword. Among stopped answers, 561 end with its requested form and 568 use the other headword’s form from the requested stage.',960,252,474,183,22);
text('b   Exact complete answers',50,467,1445,55,29,true);
table(stats.table,535,222,[175,215,251,222,269,253]);
text('c   Two requests about a single pictured entry for 言',50,796,1445,55,29,true);
text('Meaning and bronze form',62,891,430,52,24,true,teal);
text('Pronunciation and oracle form',62,976,430,52,24,true,teal);
text('d   First held-out answer using the other entry’s form',50,1080,1445,55,29,true);
text('Output',62,1183,385,52,24,true,teal);
text('Located output',62,1280,385,52,24,true,teal);
text('Reference',62,1370,385,52,24,true,teal);
for(const [name,y] of [['yan_bronze',885],['yan_oracle',970],['failed_output',1176],['located_output',1274],['failed_reference',1364]]){
 const [w,h]=stats.dimensions[name],scale=Math.min(1.48,937/w);
 await image(name,489,y,w*scale,h*scale);
}
slide.speakerNotes.textFrame.setText('Paired and scrambled training used identical examples and compute. The two outputs for 言 are exact. Each uses a different attested Hanziyuan shape physically supplied on the same card. Cross-task retention fell after dictionary training. The lower panel preserves the first wrong-head, right-stage error in original evaluation order.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/General_entry_writer.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_79_general_entry_writer.png'),new Uint8Array(await png.arrayBuffer()));
