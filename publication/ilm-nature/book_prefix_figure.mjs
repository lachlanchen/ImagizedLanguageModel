import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src=path.join(root,'references/evidence/book_prefix_v1');
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
text('Reading beyond the last two glyphs',48,18,1410,66,43,true);
text('Classical Chinese book continuation     /     1,805 held passages     /     one RTX 4090 D',53,89,1390,48,25,false,teal);
text('a   What survives self-feedback?',50,167,1380,48,32,true);
text('Exact next glyph (%)',58,228,705,32,19,false,teal);
const chart=slide.charts.add('bar',{barOptions:{direction:'column',grouping:'clustered'},
 position:{left:45,top:277,width:815,height:297},categories:['Ink + counts','Ink','Counts','Bytes + counts'],
 hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:17}},
 series:['Reference history','Self-fed history'].map((name,i)=>({name,values:stats.series[i],fill:['#16686B','#C39963'][i]})),
 xAxis:{textStyle:{fontSize:17}},yAxis:{min:0,max:12,majorUnit:3,textStyle:{fontSize:17},majorGridlines:{fill:'#DCE5E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('A correct first step can unravel',910,270,522,72,26,true,teal);
text('The visual mixture matched 162 first glyphs. At the second position, supplying the original first glyph gave 187 matches. Feeding back its prediction gave 57.',910,355,514,201,24);
text('b   The first held passage from each work',50,608,1400,48,31,true);
text('Analects',54,699,190,38,22,true,teal);await bitmap('KR1h0004-prefix',275,686,1155,50);
text('Reference',54,769,190,38,21);await bitmap('KR1h0004-reference',275,759,510,38);
text('Output',813,769,119,38,21,true);await bitmap('KR1h0004-generated',938,759,508,38);
text('Chu Ci',54,857,190,38,22,true,teal);await bitmap('KR4a0001-prefix',275,847,1155,50);
text('Reference',54,927,190,38,21);await bitmap('KR4a0001-reference',275,917,510,38);
text('Output',813,927,119,38,21,true);await bitmap('KR4a0001-generated',938,917,508,38);
text('c   Correct glyphs across sixteen self-fed steps',50,1033,1400,52,31,true);
table(stats.table,1115,190,[418,305,300,362]);
text('Exchanging the distant 30 glyphs preserved the last two. The original context gained 1.43 percentage points (95% volume-bootstrap interval 1.04 to 1.95).',55,1351,1390,85,24);
slide.speakerNotes.textFrame.setText('The two pictured strips per output row are the original reference and greedy physical output. The table reports counts. No complete eight-glyph prefix was recovered. Byte inputs share the same four-byte cell boundaries and architecture.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Book_prefix.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_135_book_prefix.png'),new Uint8Array(await png.arrayBuffer()));
