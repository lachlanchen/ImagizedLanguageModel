import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src=path.join(root,'references/evidence/compose_dialogue_v1');
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
text('Writing an answer from more than one phrase',48,15,1420,63,42,true);
text('The model reads a pictured passage and question, then writes a sequence of observed or spelled word images.',54,84,1390,67,24,false,teal);
text('a   Complete answers on held list questions',49,166,1415,52,30,true);
const chart=slide.charts.add('bar',{barOptions:{direction:'column',grouping:'clustered'},
 position:{left:48,top:235,width:865,height:330},categories:['Uniform','Assembly','Assembly + lists'],
 hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:17}},
 series:['Exact (all 501)','Exact and source-sensitive (396 difficult)'].map((name,i)=>({name,values:stats.series[i],fill:['#577B91','#18766F'][i]})),
 xAxis:{textStyle:{fontSize:18}},yAxis:{min:0,max:60,majorUnit:15,textStyle:{fontSize:17},majorGridlines:{fill:'#DCE5E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('Transfer to conversation',948,245,482,52,25,true,teal);
text('All three visual writers scored zero exact on 407 new conversational answers that required absent phrasing. A search of 32 candidates contained the reference on 1 of 239 copyable cases.',948,319,465,216,23);
text('b   Words emitted by the visual writer',49,615,1410,52,30,true);
text('List question',54,683,300,45,22,true,teal);
text('Who did the singing in the movie What’s Love Got to Do with It?',345,681,1060,66,24);
text('Uniform',55,765,285,43,21,true,teal);await bitmap('list-composition-plain',350,758,920,39);
text('Assembly + lists',55,821,285,43,21,true,teal);await bitmap('list-composition-lists',350,813,1022,36);
text('Dialogue question',54,887,300,45,22,true,teal);
text('Who is My Left Foot based on?',345,885,1000,58,24);
text('Uniform',55,967,285,42,21,true,teal);await bitmap('fresh-miss-plain',350,961,420,36);
text('Assembly + lists',800,967,231,42,21,true,teal);await bitmap('fresh-miss-lists',1050,961,355,36);
text('The conversational reference was Christy Brown.',55,1029,1380,54,22,false,teal);
text('c   Matching two supplied answers to two pictured questions',49,1110,1410,52,30,true);
table(stats.table,1182,174,[365,300,345,375]);
text('Each pair concerns one passage. Both questions were scored without conversational history.',54,1400,1390,58,22);
slide.speakerNotes.textFrame.setText('The first new source-sensitive success for the list-trained writer among questions requiring absent wording appears beside the first missed difficult conversational question. Original answer pixels are reproduced from PNG inputs. The paired assignment score uses stopped sequence likelihoods for both reference images, including both writing routes. The search budget is eight live copy-only prefixes; the candidate ceiling uses reference information at scoring only.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Compose_dialogue.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_131_compose_dialogue.png'),new Uint8Array(await png.arrayBuffer()));
