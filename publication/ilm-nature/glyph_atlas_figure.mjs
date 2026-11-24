import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src=path.join(root,'references/evidence/glyph_atlas_v1');
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
text('Looking up a character from a pictured question',48,18,1400,66,41,true);
text('6,971 illustrated entries   /   795 held headwords   /   two writing systems for the question',53,92,1395,48,25,false,teal);
text('a   Read the question, locate the entry and copy the requested fields',50,166,1400,50,30,true);
text('Both fields and all source pixels correct (%)',62,229,820,34,19,false,teal);
const chart=slide.charts.add('bar',{barOptions:{direction:'column',grouping:'clustered'},
 position:{left:43,top:269,width:860,height:260},categories:['Image','Image, varied','Byte','Byte, varied'],
 hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:17}},
 series:['Find the entry','Entry supplied'].map((name,i)=>({name,values:stats.series[i],fill:['#16686B','#C39963'][i]})),
 xAxis:{textStyle:{fontSize:16}},yAxis:{min:0,max:100,majorUnit:25,textStyle:{fontSize:17},majorGridlines:{fill:'#DCE5E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('One question, two possible headwords',933,249,510,83,25,true,teal);
text('Changing the headword or the order of the fields changes the expected answer. Each group contains four requests. The fresh test comprises 1,590 such groups.',933,341,497,182,22);
text('b   Actual questions and returned evidence',50,555,1400,55,31,true);
text('Origin of 言',60,621,450,43,23,true,teal);
text('Two forms of 马',560,621,405,43,23,true,teal);
text('Wrong entry for 丝',995,621,460,43,23,true,teal);
await bitmap('yan-question',59,683,462,42);
await bitmap('ma-question',560,678,390,42);
await bitmap('miss-question',991,683,450,42);
await bitmap('yan-answer',61,737,454,395);
await bitmap('ma-answer',554,737,396,357);
await bitmap('miss-answer',987,737,449,330);
text('Both images appear in the requested order.',560,1038,395,85,20);
text('The reader copied the fields of 丂. The requested record was 丝.',994,960,434,135,20);
text('c   What happens when the requested record is absent?',50,1148,1400,53,30,true);
table(stats.table,1220,147,[448,464,473]);
text('The deletion test removes the true entry while retaining the pictured question. Choosing among the remaining entries can return a different character.',55,1390,1385,66,21);
slide.speakerNotes.textFrame.setText('All six bitmaps come from the released reader and registered examples. 言 was available during fitting; 马, 丂 and 丝 belong to held headword families. The study measures source-grounded retrieval in a structured reference collection. The article specifies the layout, role of the static page compositor, unknown fields and energy accounting.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Glyph_atlas.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_138_glyph_atlas.png'),new Uint8Array(await png.arrayBuffer()));
