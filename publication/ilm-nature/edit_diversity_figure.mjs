import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/edit_diversity_20261118');
const stats = JSON.parse(await fs.readFile(path.join(src,'figure.json')));

const deck = Presentation.create({slideSize:{width:1500,height:1300}});
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
text('Learning from more distinct Chinese sentences',44,12,1410,65,42,true);
text('a   A complete translation and an incomplete answer',44,90,1410,46,30,true);
text('Complete translation',380,155,500,38,23,true,teal);
text('Incomplete answer',880,155,545,38,23,true,teal);
for(const [i,[field,label]] of [['source','Classical source'],['repeat','48,000 repeated'],['diverse','192,000 distinct'],['reference','Modern reference']].entries()){
  const y=217+i*57;
  text(label,64,y,305,42,24,true);
  for(const [name,x] of [['success',380],['failure',880]]){
    const key=name+'_'+field;
    await image(key,x,y,stats.widths[key]*31,31);
  }
}
text('b   Exact translations and glyph error rate',44,483,1420,45,30,true);
text('Each column contains 2,048 requests. Errors are edits per 100 renderable reference glyphs.',61,541,1390,45,22);
table(stats.table,609,245,[620,380,385]);
text('c   Exact drawing with correct preceding actions and writing',44,918,1420,46,30,true);
const chart=slide.charts.add('bar',{position:{left:50,top:1001,width:830,height:265},
 categories:['Discovery','Replication'],hasLegend:true,legend:{position:'bottom'},
 barOptions:{direction:'column',grouping:'clustered'},series:[
 {name:'Predicted code',values:stats.drawing_accuracy_percent[0],fill:teal},
 {name:'Reference code',values:stats.drawing_accuracy_percent[1],fill:'#ADC9C8'}],
 xAxis:{textStyle:{typeface:'Arial',fontSize:18}},yAxis:{min:0,max:100,majorUnit:25,
 textStyle:{typeface:'Arial',fontSize:18},majorGridlines:{fill:'#DCE4E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('Exact bitmaps (%)',934,1015,485,52,28,true,teal);
text('Both conditions use the same renderer. Supplying the reference ink measurements isolates its reconstruction accuracy.',934,1085,492,155,23);
slide.speakerNotes.textFrame.setText('Original 32 by 32 glyph images. Success is the first longer exact discovery translation with an absent Han glyph and a shared glyph. Failure is the first incorrect discovery answer that fits fourteen cells. The larger exposure arm saw 192,000 distinct new training examples, in addition to the common pretrained editor. Both arms saw 240,000 prompted presentations.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Edit_diversity.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,12382500','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_50_edit_diversity.png'),new Uint8Array(await png.arrayBuffer()));
