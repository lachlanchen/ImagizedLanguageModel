import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/visual_inference_writer_20261119');
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
text('Reading claims and restoring missing Chinese writing',50,22,1450,65,39,true);
text('a   Exact stopped answers on official OCNLI development (%)',50,103,1450,50,29,true);
const chart=slide.charts.add('bar',{barOptions:{direction:'bar',grouping:'clustered'},position:{left:40,top:172,width:836,height:315},
 categories:['Decision','Missing text','Missing ink¹'],hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:17}},
 series:[{name:'Reassigned premises',values:stats.series[0],fill:'#86ABA9'},{name:'Paired premises',values:stats.series[1],fill:teal}],
 xAxis:{textStyle:{fontSize:20}},yAxis:{min:0,max:100,textStyle:{fontSize:19},majorGridlines:{fill:'#DBE6E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('Two writing tasks',913,172,514,70,27,true,teal);
text('Read a premise and a claim. Write their relation, or recover one to three glyphs covered by a single mask in the claim.',913,247,490,156,25);
text('¹ At least one target glyph is absent from both displayed inputs.',913,412,480,78,21);
text('b   Comparison with erased evidence and local visual baselines',50,515,1450,52,29,true);
table(stats.table,576,225,[365,260,240,260,260]);
text('Triples require three correct decisions about different claims for one premise. Local restoration counts use the two visible neighbors.',60,818,1390,75,22);
text('c   The same missing phrase, changing source evidence',50,929,1450,52,29,true);
text('Visible premise',62,1018,390,52,24,true);
text('Claim to complete',532,1018,400,52,24,true);
text('Written by model',1028,1018,375,52,24,true);
for(const [i,target] of [[0,'水'],[1,'茶']]) {
 const y=1090+115*i;
 await image('premise'+i,62,y,438,438*stats.dimensions['premise'+i][1]/stats.dimensions['premise'+i][0]);
 await image('claim'+i,532,y,300,300*stats.dimensions['claim'+i][1]/stats.dimensions['claim'+i][0]);
 const lab='cli'+i,wh=stats.dimensions[lab],scale=Math.min(2,350/wh[0]);
 await image(lab,1029,y,wh[0]*scale,wh[1]*scale);
}
text('Fixed, newly written examples. The inputs were submitted once as typed text and again as PNGs. Both routes gave identical answer images.',62,1341,1375,83,22);
slide.speakerNotes.textFrame.setText('Native numbers use the fixed official cohort. Pictured outputs for water and tea are actual emitted pixels, whatever their correctness. Full reference and masked test archives were frozen before scoring.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Visual_inference_writer.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_83_visual_inference_writer.png'),new Uint8Array(await png.arrayBuffer()));
