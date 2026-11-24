import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src=path.join(root,'references/evidence/dialogue_ink_v1');
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
text('Writing answers from pictured conversations',48,15,1420,63,42,true);
text('Human questions and answers from CoQA, rendered as streams of printed glyphs.',54,89,1400,52,25,false,teal);
text('a   Does the answer depend on the pictured inputs?',49,163,1400,50,30,true);
const chart=slide.charts.add('bar',{barOptions:{direction:'column',grouping:'clustered'},
 position:{left:48,top:224,width:920,height:350},categories:['Reserved stories','Development stories'],
 hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:16}},
 series:['Span reader','Direct writer','Word code','Question swap','Passage swap'].map((name,i)=>({name,values:stats.series[i],fill:['#8195A3','#255078','#16686B','#BA9770','#A65448'][i]})),
 xAxis:{textStyle:{fontSize:18}},yAxis:{min:0,max:40,majorUnit:10,textStyle:{fontSize:17},majorGridlines:{fill:'#DCE5E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('Exact answers (%)',1005,245,425,54,26,true,teal);
text('Swapping the current question and history, or swapping the passage, reduces agreement with the original human answer. Both swaps use the word-code writer.',1005,314,426,218,23);
text('b   Two consecutive turns on the same light-bulb passage',49,603,1410,50,30,true);
text('Pictured question (transcription)',56,681,750,45,23,true,teal);
text('Direct writer output (original pixels)',805,681,620,45,23,true,teal);
text('How much energy do they convert to light?',56,762,741,62,24);
await bitmap('energy-fraction',808,764,568,45);
text('What happens to the rest of the energy?',56,851,741,62,24);
await bitmap('energy-fate',808,852,595,45);
text('The second answer is incorrect. The human respondent wrote “it is converted into heat”.',56,952,1384,72,23,false,teal);
text('c   Exact answers when the reference requires new wording',49,1061,1412,50,30,true);
table(stats.table,1142,166,[340,160,310,290,285]);
text('Normalized first-reference matches. Exact reference pixels occurred once with word coding on development and zero times on reserved stories.',55,1351,1385,92,22);
slide.speakerNotes.textFrame.setText('The word-code writer predicts a continuous representation of the next word. On 83 development questions for which all four references required multiword phrasing absent from every input, neither writer matched any human answer. The two answer images were reproduced independently on CPU. Chart and table entries use only stopped outputs.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Dialogue_ink.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_128_dialogue_ink.png'),new Uint8Array(await png.arrayBuffer()));
