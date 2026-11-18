import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/local_ink_20261118');
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
text('How nearby glyphs change a written answer',44,12,1410,65,42,true);
text('a   Actual writing on newly reserved sentences',44,86,1430,48,30,true);
text('A two-glyph insertion',374,149,515,40,23,true,teal);
text('A readable substitution error',900,149,550,40,23,true,teal);
for(const [i,[field,label]] of [['source','Classical source'],['prefix','Modern prefix'],['prediction','Visual writer'],['local','Local count prediction'],['reference','Recorded insertion']].entries()){
  const y=210+i*49;
  text(label,64,y,308,38,23,true);
  for(const [name,x] of [['success',374],['error',900]]){
    const key=name+'_'+field;
    await image(key,x,y,stats.widths[key]*29,29);
  }
}
text('The source boundary and preceding modern writing are supplied. Each model generates and stops.',62,466,1390,38,22);
text('b   Complete recorded phrases, including the learned stop',44,536,1420,45,30,true);
table(stats.table,608,276,[515,280,280,310]);
text('c   Correct multi-glyph phrases across the three cohorts',44,938,1420,46,30,true);
const chart=slide.charts.add('bar',{position:{left:50,top:1001,width:835,height:266},
 categories:['Discovery','Replication','Fresh sentences'],hasLegend:true,legend:{position:'bottom'},
 barOptions:{direction:'column',grouping:'clustered'},series:[
 {name:'First glyph only',values:stats.multi_counts[0],fill:'#ADC9C8'},
 {name:'Throughout',values:stats.multi_counts[1],fill:teal},
 {name:'Whole-phrase counts',values:stats.multi_counts[2],fill:'#53687C'}],
 xAxis:{textStyle:{typeface:'Arial',fontSize:18}},yAxis:{min:0,max:80,majorUnit:20,
 textStyle:{typeface:'Arial',fontSize:18},majorGridlines:{fill:'#DCE4E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('Complete multi-glyph phrases',915,1007,510,50,27,true,teal);
text('Fresh sentences contain 3,448 multi-glyph targets. With local evidence at every cell, the writer produces 27 exactly, up from one with the correction confined to the first glyph.',915,1074,517,159,22);
slide.speakerNotes.textFrame.setText('Outputs are original held-out image streams. Each column of panel a shows the first qualifying multi-glyph example in its category with a nonempty modern prefix, source-absent target and width bounds. Both trained writers use the same starting weights, training examples and schedule. The local count reference can return observed training phrases. Examples are the first eligible cases in the fresh roster.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Local_ink.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,12382500','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_53_local_ink.png'),new Uint8Array(await png.arrayBuffer()));
