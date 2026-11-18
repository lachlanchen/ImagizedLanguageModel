import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/clause_support_20261118');
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
text('Can the phrase writer express the intended meaning?',44,17,1412,69,43,true);
text('a   The temporal reading of a polysemous glyph',44,113,1430,51,31,true);
for (const [i,[key,label]] of [['copy','Classical source'],['default','Writer output'],['unrestricted','Closest legal output*'],['reference','Recorded gloss']].entries()){
 const y=185+i*59; text(label,64,y,308,43,23,true);await image(key,390,y,stats.widths[key]*31,31);
}
text('Here the ambiguous character means “finally”. The recorded passage describes a minister dispatching an envoy. The writer breaks the verb phrase even when permitted to choose against the reference.',63,435,1364,89,24);
text('b   Exact search through all observed phrase combinations',44,554,1425,52,31,true);
table(stats.table,630,329,[398,231,230,245,281]);
text('* These two comparisons use the recorded gloss to select an output. Both measure what is reachable. All 1,024 predictions from the released writer appear among the six highest-scoring streams.',61,983,1380,75,23);
text('c   Faithful modern glosses in 18 complete passages',44,1083,1420,53,31,true);
const chart=slide.charts.add('bar',{position:{left:36,top:1164,width:836,height:246},categories:['Released writer','Any of six','Closest legal*','Corpus gloss'],hasLegend:false,
 barOptions:{direction:'bar',grouping:'clustered'},series:[{name:'Accepted glosses',values:stats.series[0],fill:teal}],
 xAxis:{textStyle:{typeface:'Arial',fontSize:19}},yAxis:{min:0,max:18,majorUnit:3,textStyle:{typeface:'Arial',fontSize:18},majorGridlines:{fill:'#DBE6E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('What a wider search rescued',910,1155,510,56,26,true,teal);
text('Among 13 passages with acceptable reference glosses, the writer translated one acceptably. Its six best candidates provided no acceptable alternative for the other 12.',910,1220,518,173,25);
slide.speakerNotes.textFrame.setText('Literal original Chinese images. Frozen review of eighteen passages by one study-involved model reader. The closest legal stream uses the reference. Physical mismatch to a single gloss does not establish semantic error.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Clause_support.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_58_clause_support.png'),new Uint8Array(await png.arrayBuffer()));
