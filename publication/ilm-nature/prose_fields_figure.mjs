import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src=path.join(root,'references/evidence/prose_fields_v1');
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
text('Whose rating did the model read?',50,20,1400,63,44,true);
text('One human-written paragraph   /   one pictured question   /   two different replies',55,92,1385,44,25,false,teal);
text('a   Change only the restaurant named in the prompt',52,155,1400,50,30,true);
text('Paragraph',55,223,240,40,23,true,teal);
await bitmap('restaurant-prose',280,211,1140,127);
text('Question',55,355,215,40,23,true,teal);
await bitmap('restaurant-question',280,349,1030,46);
text('Name in prompt',55,431,460,40,22,true,teal);
text('Generated answer',864,431,500,40,22,true,teal);
await bitmap('name-0',55,484,752,52);
await bitmap('reply-0',863,484,525,52);
await bitmap('name-1',55,555,752,52);
await bitmap('reply-1',863,555,525,52);
text('b   Did both names receive the right value?',52,656,1400,50,30,true);
const chart=slide.charts.add('bar',{barOptions:{direction:'column',grouping:'clustered'},
 position:{left:42,top:735,width:782,height:266},categories:['Original','Name erased','Named image','Named byte'],
 hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:16}},
 series:['Joined replies','Authored prose'].map((name,i)=>({name,values:stats.series[i],fill:['#126B70','#B48553'][i]})),
 xAxis:{textStyle:{fontSize:16}},yAxis:{min:0,max:100,majorUnit:25,textStyle:{fontSize:16},majorGridlines:{fill:'#DCE5E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('Percent of complete comparisons',65,712,750,35,18,false,teal);
text('When a name is replaced by “the latter”',850,715,597,55,22,true,teal);
await bitmap('park-prose',850,783,578,90);
await bitmap('park-reply',850,890,515,46);
text('Asked about the tower, the model returned the garden’s 12 yuan. The tower costs 35 yuan.',850,949,575,86,20);
text('c   One venue in continuous dialogue prose',52,1042,1400,50,30,true);
table(stats.table,1120,229,[627,440,318]);
text('The two-property natural group has only three eligible questions. The larger joined comparison in b requires four correct replies per pair.',55,1384,1390,63,21);
slide.speakerNotes.textFrame.setText('Original evaluated input and emitted bitmaps. Joined replies: 539 comparisons. Authored multi-venue prose: 12. The native restaurant paragraph is the sole naturally occurring two-name pair that passed the identity and literal-value screen in CrossWOZ test.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Prose_fields.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_142_prose_fields.png'),new Uint8Array(await png.arrayBuffer()));
