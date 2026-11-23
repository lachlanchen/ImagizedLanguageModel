import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src=path.join(root,'references/evidence/stroke_scale_20261124');
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
 title:'Reading unfamiliar Chinese print',
 a:'a   Passages with two exact answers (%)',
 categories:['Dev HanaMin','Dev Fandol','Trial HanaMin','Trial Fandol','Trial regular'],
 names:['Shift only','Stroke + scale','Print routing'],max:10,tick:2,
 note:'Both training runs use the same 230,400 presentations. Routing selects the original reader when at least 80% of nonblank cells in each strip match the fixed inventory.',
 b:'b   Human translations of questions about the same Chinese passages',
 widths:[320,160,220,220,220,245],c:'c   Two questions about one page in HanaMin',
 rows:[['first_question','First question'],['first_answer','Correct reply'],['second_question','Second question'],['second_answer','Exact-match miss'],['second_reference','Reference answer']]
};
text(cfg.title,50,22,1420,65,42,true);
text(cfg.a,50,111,1410,52,29,true);
const colors=['#347C80','#BB734A','#6D7EA4','#922F42'];
const chart=slide.charts.add('bar',{barOptions:{direction:'column',grouping:'clustered'},
 position:{left:60,top:176,width:900,height:325},categories:cfg.categories,
 hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:16}},
 series:cfg.names.map((name,i)=>({name,values:stats.series[i],fill:colors[i]})),
 xAxis:{textStyle:{fontSize:16}},yAxis:{min:0,max:cfg.max,majorUnit:cfg.tick,textStyle:{fontSize:16},majorGridlines:{fill:'#DFE6E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('Training and routing',995,190,420,55,26,true,teal);text(cfg.note,995,248,442,230,22);
text(cfg.b,50,536,1410,50,29,true);table(stats.table,609,315,cfg.widths);
text(cfg.c,50,968,1410,55,29,true);
for(let i=0;i<cfg.rows.length;i++) await strip(...cfg.rows[i],1045+i*68);
text('Both answers are the stroke-and-scale writer’s original pixels. Each returned answer was scored after the writer stopped.',60,1405,1390,45,21);
slide.speakerNotes.textFrame.setText('The two language columns of the experiment use human translations of the same questions and the same Chinese answer references. XQuAD is a translation of SQuAD development data.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Stroke_scale.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_103_stroke_scale.png'),new Uint8Array(await png.arrayBuffer()));
