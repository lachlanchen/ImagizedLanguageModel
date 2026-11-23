import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/page_evidence_writer_20261124');
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


text('Writing a fact and its year from a pictured passage',50,20,1410,66,43,true);
text('a   Correct under all three new question wordings',50,102,1400,52,29,true);
const chart=slide.charts.add('bar',{barOptions:{direction:'bar',grouping:'clustered'},
 position:{left:55,top:170,width:850,height:305},categories:['One training instruction','Varied instructions'],
 hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:17}},
 series:[{name:'Single passage (64)',values:stats.series[0],fill:'#317E82'},
         {name:'Two paragraphs (48)',values:stats.series[1],fill:'#B66A41'}],
 xAxis:{textStyle:{fontSize:18}},yAxis:{min:0,max:64,majorUnit:16,textStyle:{fontSize:16},majorGridlines:{fill:'#DFE6E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('What counts as correct',950,180,470,54,27,true,teal);
text('A single passage requires six complete replies. Two paragraphs require twelve. Every answer must name the correct participant, give the year and stop.',950,247,464,202,24);
text('b   The same source passages under the first held-out wording',50,485,1430,54,29,true);
table(stats.table,557,288,[398,114,208,342,323]);
text('¹ The control receives the correct subject, relation and object.',60,858,1380,43,21,false,teal);
text('c   Original model outputs from the new passages',50,922,1400,50,29,true);
async function strip(name,label,y){
 text(label,61,y,330,50,22,true,teal);
 const png=await fs.readFile(path.join(src,name+'.png'));
 const w=png.readUInt32BE(16),h=png.readUInt32BE(20),height=Math.min(42,1035*h/w);
 await image(name,400,y,w*height/h,height);
}
await strip('original_output','Before the year edit',1000);
await strip('edit_output','After the year edit',1076);
await strip('failure_reference','Two-paragraph reference',1195);
await strip('failure_output','Two-paragraph output',1275);
text('Each strip preserves generated or reference pixels at their original aspect ratio.',60,1376,1380,48,22);
slide.speakerNotes.textFrame.setText('Panel a counts every correct reply for each new source. Panel b gives exact, stopped answers, with both word orders required for DuIE and CMRC. Panel c uses the first complete new single-page pair and first failed two-paragraph reply in fixed evaluation order.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Page_evidence.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_98_page_evidence.png'),new Uint8Array(await png.arrayBuffer()));
