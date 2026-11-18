import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/entry_question_reader_20261119');
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
text('What changes when a reader preserves question order?',48,20,1420,65,39,true);
text('a   Complete questions and answers on a held-out dictionary card',48,99,1435,55,29,true);
for(const [i,[key,title]] of [['meaning','Meaning and first form'],['sound','Sound and second form'],['cue_first','Cues moved before fields']].entries()){
  const y=170+i*129;
  text(title,52,y,325,70,23,true,i===2?'#844426':teal);
  for(const [j,suffix] of ['question','answer'].entries()){
    const name=key+'_'+suffix, n=stats.strips[name].cells, scale=Math.min(32,1005/n);
    text(j===0?'Prompt':'Output',380,y+49*j,104,38,19,false);
    await image(name,493,y+49*j,n*scale,scale);
  }
}
text('The first two requests contain identical glyph inventories. Moving the same cues before the fields changes the syntax; the third output answers a different request.',55,567,1390,72,23);
text('b   Complete answers on 4,544 requests from 284 held-out families',48,669,1410,55,28,true);
table(stats.table,745,226,[325,250,240,280,290]);
text('Each group of four uses one headword and four different targets. All four are credited only when every complete answer matches and stops. The cue-placement test uses the frozen ordered writer.',55,989,1390,83,22);
text('c   Mean loss per writing action',48,1107,1410,52,28,true);
const chart=slide.charts.add('bar',{barOptions:{direction:'column',grouping:'clustered'},position:{left:48,top:1164,width:831,height:271},categories:['1','2','3','4','5','6','7','8'],hasLegend:true,legend:{position:'bottom'},
 series:['Sorted question','Ordered question'].map((name,i)=>({name,values:stats.series[i],fill:['#6483A2',teal][i]})),
 xAxis:{textStyle:{typeface:'Arial',fontSize:16}},yAxis:{min:0,textStyle:{typeface:'Arial',fontSize:16},majorGridlines:{fill:'#DBE6E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('Eight passes through 44,600 requests',906,1166,522,73,23,true,teal);
text('Both models start from the same visual writer. Only the new question encoder receives sorted pixels in the control. Both reproduce all 1,136 original simple requests.',906,1262,517,165,22);
slide.speakerNotes.textFrame.setText('Rows one and two are fixed opposites on the first quartet. Row three is the first archived success lost under cue relocation. The training and reserved frames both place status after its field.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Question_order.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_71_question_order.png'),new Uint8Array(await png.arrayBuffer()));
