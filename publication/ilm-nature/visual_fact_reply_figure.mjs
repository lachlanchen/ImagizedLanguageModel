import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src=path.join(root,'references/evidence/visual_fact_reply_v1');
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
text('Reading the question and writing the requested fact',50,20,1400,66,41,true);
text('Chinese dialogue    /    5,097 human training replies    /    one complete card per question',54,91,1390,42,25,false,teal);
text('a   Two questions about the same restaurant',52,158,1400,45,30,true);
text('Pictured question',57,216,685,41,23,true,teal);
text('Actual answer image',775,216,670,41,23,true,teal);
await bitmap('question-a',54,275,678,95);
await bitmap('question-a-reply',774,275,665,95);
await bitmap('question-b',54,395,678,122);
await bitmap('question-b-reply',774,395,665,122);
text('The first question asks for opening hours. The second asks where the restaurant is, without the printed word for address.',57,531,1378,61,22);
text('b   Reading an observed value under held questions (%)',52,620,1410,46,30,true);
const chart=slide.charts.add('bar',{barOptions:{direction:'column',grouping:'clustered'},
 position:{left:42,top:680,width:840,height:237},categories:['All','New question','New entity','Both new'],
 hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:17}},
 series:['Image','Question erased','Bytes'].map((name,i)=>({name,values:stats.series[i],fill:['#17696B','#C6A46C','#8EA7AD'][i]})),
 xAxis:{textStyle:{fontSize:17}},yAxis:{min:0,max:100,majorUnit:25,textStyle:{fontSize:17},majorGridlines:{fill:'#DCE5E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('First incorrect field',920,690,486,40,23,true,teal);
await bitmap('wrong-field',918,741,500,95);
await bitmap('wrong-field-reply',918,850,500,60);
text('The guest asked for hotel type. The writer gave its address.',922,918,490,68,21);
text('c   Referring to unfamiliar ink inside an English sentence',52,1000,1400,45,29,true);
await bitmap('historical-reply',54,1065,790,153);
text('Two attested historical forms occupy the venue and landmark roles. Exchanging their pixels changes the written references.',915,1070,520,126,22);
table(stats.table,1240,144,[395,308,357,325]);
text('Both directions must preserve the two exact source names, place the landmark by a proximity phrase, and convey all other recognized facts.',55,1404,1390,58,20);
slide.speakerNotes.textFrame.setText('All displayed strip images are generated, physically reproduced samples. The Chinese value measure permits unverified wording. Historical glyphs are copied and do not demonstrate reading their historical meanings.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Visual_fact_reply.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_140_visual_fact_reply.png'),new Uint8Array(await png.arrayBuffer()));
