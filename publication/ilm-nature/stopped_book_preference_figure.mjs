import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src=path.join(root,'references/evidence/stopped_book_preference_20261124');
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



async function strip(name,label,y){
 text(label,60,y,335,48,22,true,teal);
 const png=await fs.readFile(path.join(src,name+'.png'));
 const w=png.readUInt32BE(16),h=png.readUInt32BE(20),height=Math.min(42,1055*h/w);
 await image(name,402,y,w*height/h,height);
}
const cfg={
 title:'Can a writer learn from its own wrong book answers?',
 a:'a   Training pairs in which the accepted answer outranks the rival (%)',
 categories:['All replies','Shorter rival','Equal length','Longer rival'],
 names:['Released','Gold replay','Preference'],
 max:100,tick:20,
 note:'Rivals were wrong answers emitted by the released writer. Scores include every glyph and the final stop. Each comparison uses the same pair of images.',
 b:'b   Passages or headwords with every requested answer correct',
 widths:[435,175,215,280,280],c:'c   Input and output pixels from the first Chinese success and English miss on six new works',
 rows:stats.strips
};
text(cfg.title,50,22,1420,65,42,true);
text(cfg.a,50,111,1410,52,29,true);
const colors=['#347C80','#BB734A','#6D7EA4','#922F42'];
const chart=slide.charts.add('bar',{barOptions:{direction:'column',grouping:'clustered'},
 position:{left:60,top:176,width:900,height:325},categories:cfg.categories,
 hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:17}},
 series:cfg.names.map((name,i)=>({name,values:stats.series[i],fill:colors[i]})),
 xAxis:{textStyle:{fontSize:16}},yAxis:{min:0,max:cfg.max,majorUnit:cfg.tick,textStyle:{fontSize:16},majorGridlines:{fill:'#DFE6E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('Training replies',995,190,420,55,26,true,teal);text(cfg.note,995,248,442,230,22);
text(cfg.b,50,536,1410,50,29,true);table(stats.table,609,315,cfg.widths);
text(cfg.c,50,968,1410,55,29,true);
for(let i=0;i<cfg.rows.length;i++) await strip(...cfg.rows[i],1023+i*52);
text('Excerpts show original input cells around an accepted answer. The writer received the full passage. Writer replies are shown exactly as emitted.',60,1396,1390,54,20);
slide.speakerNotes.textFrame.setText('Four responses per literary, BiPaR and XQuAD passage. CMRC and historical rows require both responses. Paired training preferences and held-out free reading are measured separately.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Stopped_book_preference.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_106_stopped_book_preference.png'),new Uint8Array(await png.arrayBuffer()));
