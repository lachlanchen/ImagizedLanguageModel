import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src=path.join(root,'references/evidence/visual_clause_roles_v1');
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
text('Reading the same glyph in two grammatical roles',48,15,1420,64,43,true);
text('Two pictured questions ask for either the action or the object of a classical Chinese passage.',54,88,1395,57,23,false,teal);
text('a   When the same glyph changes role and meaning',49,156,1410,54,29,true);
const colors=['#8393A1','#276E73','#AD694B'];
const chart=slide.charts.add('bar',{barOptions:{direction:'column',grouping:'clustered'},
 position:{left:48,top:220,width:865,height:300},categories:['Located in both passages','Located and written in both'],
 hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:17}},
 series:['Prior reader','Distilled reader','Glyph counts¹'].map((name,i)=>({name,values:stats.series[i],fill:colors[i]})),
 xAxis:{textStyle:{fontSize:17}},yAxis:{min:0,max:250,majorUnit:50,textStyle:{fontSize:17},majorGridlines:{fill:'#DCE5E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('429 pairs · 124 target glyphs',944,239,509,49,24,true,teal);
text('Each pair requires a different English word for the same printed character. A pass requires both English and Chinese requests to reach the intended glyph in each passage.',945,299,470,195,22);
text('b   Actual replies to questions about the same printed character',49,548,1405,53,29,true);
for(const [label,x] of [['Passage',52],['Requested word',584],['Written reply',1075]])text(label,x,627,410,48,22,true,teal);
for(const [i,id,req,word] of [[0,3669,0,'Action'],[1,2021,1,'Object'],[2,3669,20,'Action · different wording']]){
 const y=703+90*i;await bitmap('samples/passage-'+id,51,y,410,53);text(word,583,y,480,58,24);
 await bitmap('samples/reply-'+id+'-'+req,1071,y,295,53);
}
text('The last question is misread as an object request. The source reader and writer are the same in all three rows.',55,982,1395,68,20);
text('c   Entire passages with both English and Chinese questions correct',48,1080,1410,51,29,true);
table(stats.table,1152,194,[544,255,322,264]);
text('¹ The count baseline receives the intended role and scores each source cell using only its bitmap.',57,1381,1380,59,21);
slide.speakerNotes.textFrame.setText('The chart counts pairs where the target is the verb on one page and the object on another, with disjoint registered readings. The first two selected images show speak and speech for 言. The last output is it. Panel c first requires both complete pages in each opposite-order pair, then reports complete individual pages on prior material. The new balanced arc reader without new question data scored zero on the four classroom questions across all passages. For all twelve new wording pairs the distilled router still misses three. No balancing advantage over uniform sampling on the paired source-location test was found.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Visual_clause_roles.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_123_visual_clause_roles.png'),new Uint8Array(await png.arrayBuffer()));
