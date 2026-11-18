import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/visual_edit_20261118');
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
text('Editing a pictured classical Chinese sentence',44,12,1410,64,42,true);
text('a   Exact substitutions in held-out chronicles',44,86,1410,48,30,true);
text('Book of Liang • first exact mixed answer',243,152,660,40,23,true,teal);
text('Book passage • replacing an honorific for death',889,152,594,62,23,true,teal);
for(const [i,label] of ['Source','Emitted','Reference'].entries()){
  const field=['source','actual','reference'][i];
  const y=221+i*59;
  text(label,64,y,160,40,24,true);
  await image('first_'+field,241,y,540,36);
  await image('death_'+field,888,y,396,36);
}
text('b   Whole answers after learning how many insertions to allow',44,424,1410,46,30,true);
table(stats.table,490,212,[620,375,390]);
text('c   Glyph edit distance per 100 reference glyphs',44,735,1410,46,30,true);
const chart=slide.charts.add('bar',{position:{left:55,top:811,width:856,height:278},
 categories:['Copy','Local edits','Fixed limit','Learned limit','Joint + limit'],hasLegend:false,
 barOptions:{direction:'column',grouping:'clustered'},series:[{name:'Edit rate (%)',values:stats.edit_rate_percent,fill:teal}],
 xAxis:{textStyle:{typeface:'Arial',fontSize:17}},yAxis:{min:0,max:100,majorUnit:25,
 textStyle:{typeface:'Arial',fontSize:18},majorGridlines:{fill:'#DCE4E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('First translation error',955,809,490,42,26,true,teal);
for(const [i,label] of ['Source','Emitted','Reference'].entries()){
  const field=['source','actual','reference'][i];
  const y=880+i*56;
  text(label,943,y,125,34,20,true);
  await image('error_'+field,1074,y,Math.min(365,({source:9,actual:11,reference:12})[field]*29),29);
}
text('d   Which part of the writer misses the reference?',44,1109,1420,46,30,true);
text('Given the reference ink measurements',64,1184,635,38,23);
text('37,427 / 37,451',64,1231,580,49,31,true,teal);
text('Predicting ink from the correct preceding writing',735,1184,715,38,23);
text('3,816 / 37,451',735,1231,655,49,31,true,teal);
slide.speakerNotes.textFrame.setText('Saved pixels for two exact confirmation sentences and the first failed translation in roster order. The second example is the third exact mixed answer. Quotation includes every renderable request. Exact translations include unsupported requests. Chart is first-roster aggregate cell edit rate. Lower values are better. The conditional writing diagnostic supplies the correct cursor and preceding output.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Visual_edit.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,12382500','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_48_visual_edit.png'),new Uint8Array(await png.arrayBuffer()));
