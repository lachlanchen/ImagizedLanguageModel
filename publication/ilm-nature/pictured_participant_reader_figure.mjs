import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/pictured_participant_reader_20261119');
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
text('Reading grammatical roles from glyph images',50,22,1450,65,41,true);
text('a   Both arguments correct for the same held-out verb occurrence',50,110,1450,52,29,true);
const chart=slide.charts.add('bar',{barOptions:{direction:'bar',grouping:'clustered'},position:{left:48,top:184,width:815,height:265},
 categories:['Previous','Exchanged','Matched'],hasLegend:false,
 series:[{name:'Both arguments correct (106 predicates)',values:stats.series[0],fill:teal}],xAxis:{textStyle:{fontSize:21}},
 yAxis:{min:0,max:40,textStyle:{fontSize:19},majorGridlines:{fill:'#DBE6E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('One sentence, two questions',910,186,515,60,26,true,teal);
text('Each question asks for the subject or the object of the same verb. All 106 test predicates occur in sentences excluded from this fit. The model returns the complete answer as glyph images.',910,266,506,164,23);
text('b   Complete answers under changed wording or task',50,483,1450,52,29,true);
table(stats.table,551,195,[218,290,280,290,307]);
text('New scenes use independently worded questions. New verbs appear in active and passive frames. XWinograd asks the reader to supply the referent of a blank.',60,763,1380,74,22);
text('c   Original passage, pictured questions and returned images',50,860,1450,52,29,true);
async function strip(name,x,y,w,h){
 const [iw,ih]=stats.dimensions[name],f=Math.min(w/iw,h/ih);await image(name,x,y,iw*f,ih*f);
}
for(const [i,key] of ['grammar','fresh'].entries()){
 const y=947+i*243;
 text(i===0?'Treebank sentence':'New situation',55,y+6,327,42,24,true,teal);
 await strip(key+'_source',386,y-5,1030,83);
 text('Question',60,y+86,400,32,18,true);
 text('Returned image',1000,y+86,400,32,18,true);
 for(const j of [0,1]){
  await strip(key+'_q'+j,60,y+119+j*51,914,40);
  await strip(key+'_a'+j,1000,y+119+j*51,410,40);
 }
}
slide.speakerNotes.textFrame.setText('Chinese GSD r2.16 under CC BY-SA 4.0. The first treebank test pair answered completely and correctly and the first fresh scene in evaluation order. All answer strips are original checkpoint outputs. See the figure caption for attribution and selection.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Pictured_participant_reader.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_87_pictured_participant_reader.png'),new Uint8Array(await png.arrayBuffer()));
