import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/edit_contrast_20261118');
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
text('Learning which glyph to draw in a translation',44,12,1410,65,42,true);
text('a   Two modernizations of classical Chinese',44,90,1410,46,30,true);
text('First correct changed sentence',377,157,503,35,22,true,teal);
text('First correct long mixed sentence',846,157,626,35,22,true,teal);
for(const [i,[k,title]] of [['source','Source'],['likelihood','Likelihood'],['contrast','Whole-glyph contrast'],['noise_corrected','Corrected contrast'],['reference','Reference']].entries()){
 const y=216+i*53;
 text(title,64,y,300,40,23,true);
 for(const [name,x] of [['short',377],['long',846]]){
   const key=name+'_'+k;
   await image(key,x,y,Math.min(515,stats.widths[key]*34),34);
 }
}
text('b   Complete reference translations and glyph edit rate',44,504,1420,45,30,true);
text('Each column contains 2,048 requests. Lower edit rates indicate closer agreement.',62,561,1390,40,22);
table(stats.table,624,270,[600,395,390]);
text('c   Exact drawings given the recorded preceding writing (%)',44,947,1420,46,30,true);
const chart=slide.charts.add('bar',{position:{left:55,top:1024,width:830,height:244},
 categories:['Initial editor','Likelihood','Contrast','Corrected'],hasLegend:false,
 barOptions:{direction:'column',grouping:'clustered'},series:[{name:'Exact drawings (%)',values:stats.drawing_accuracy_percent,fill:teal}],
 xAxis:{textStyle:{typeface:'Arial',fontSize:18}},yAxis:{min:0,max:Math.ceil(Math.max(...stats.drawing_accuracy_percent)/4)*4,majorUnit:4,
 textStyle:{typeface:'Arial',fontSize:18},majorGridlines:{fill:'#DCE4E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('55,758 draw positions',931,1030,509,47,29,true,teal);
text('Fresh chapters. The source cursor, past actions and previous glyphs were supplied.',931,1092,515,110,23);

slide.speakerNotes.textFrame.setText('Original pixels. Examples are selected in roster order from cases where the likelihood-trained control changed the source and exactly matched the recorded modern form. All versions use the same renderer and learned insertion allowance. The further chapters were reserved before the frequency-corrected treatment.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Edit_contrast.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,12382500','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_49_edit_contrast.png'),new Uint8Array(await png.arrayBuffer()));
