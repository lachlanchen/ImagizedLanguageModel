import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src=path.join(root,'references/evidence/pronoun_writer_v1');
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
text('One printed character, two English readings',50,16,1420,64,43,true);
text('A residual trained on pictured answers restores a grammatical use omitted from the original writer’s training.',54,89,1400,60,23,false,teal);
text('a   Distinguishing an object pronoun from a verb of motion',49,162,1410,54,29,true);
const colors=['#8393A1','#276E73','#AD694B'];
const chart=slide.charts.add('bar',{barOptions:{direction:'column',grouping:'clustered'},
 position:{left:49,top:220,width:895,height:293},categories:['Reserved works (8 pairs)','Reserved sections (40 pairs)'],
 hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:18}},
 series:['Original','Aligned answer','Exchanged answer'].map((name,i)=>({name,values:stats.series[i],fill:colors[i]})),
 xAxis:{textStyle:{fontSize:17}},yAxis:{min:0,max:40,majorUnit:10,textStyle:{fontSize:17},majorGridlines:{fill:'#DCE5E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('Both answers required',979,232,440,50,25,true,teal);
text('Each pair contains the same Chinese character. In one passage the writer must produce “go”; in the other it must produce the canonical object pronoun “it”.',979,293,440,180,22);
text('b   Original passages and generated answers',48,551,1400,52,29,true);
for(const [label,x] of [['Input window',57],['Original',545],['Aligned',806],['Reference sense',1096]])text(label,x,631,340,46,22,true,teal);
for(const [i,name,label] of [[0,'held_book-object_person-correct','Object pronoun'],[1,'held_book-motion-correct','Motion'],[2,'held_book-motion-miss','Motion · missed']]) {
 const y=705+98*i;await bitmap(name,54,y,424,52);await bitmap('samples/'+name+'-parent',541,y,201,52);await bitmap('samples/'+name+'-aligned',804,y,201,52);text(label,1092,y,355,55,24);
}
text('c   Whole-passage transfer with the source reader fixed',49,1042,1410,52,29,true);
table(stats.table,1118,215,[253,215,215,236,466]);
text('Counts require correct action and object readings. The last two rows reuse the stricter translations of the two new books.',58,1372,1395,65,21);
slide.speakerNotes.textFrame.setText('Each pair joins two distinct windows with the same printed central character. We use actual outputs from the held works and remove only the empty exterior input cells for display. The canonical pronoun it tests person, while a contextually correct translation may require him, her or them. Supplied positions quantify a limit of the writer; whole-passage scores also require the source locations to be found.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Pronoun_writer.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_122_pronoun_writer.png'),new Uint8Array(await png.arrayBuffer()));
