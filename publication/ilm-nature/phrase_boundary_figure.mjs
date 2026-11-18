import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/phrase_boundary_20261119');
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
text('Learning the boundaries of a pictured phrase',44,18,1410,64,40,true);
text('a   Complete input and answer images',44,98,1410,51,29,true);
for (const [x,title] of [[60,'Source image'],[504,'Single-cell answer'],[974,'Joint-span answer']])
 text(title,x,164,430,48,24,true,teal);
for (const [i,tag] of ['camps','bridge','depart','elm','army'].entries()) {
  for(const [suffix,x] of [['source',60],['single',504],['joint',974]]){
    const key=tag+'_'+suffix;
    const scale=Math.min(30, 445/stats.widths[key]);
    await image(key,x,226+i*62,stats.widths[key]*scale,scale);
  }
}
text('The first two answers repair the distance expression. Both writers give the wrong action for an elm and repeat the attack verb in the final line.',60,548,1368,76,22);
text('b   Which complete answers preserve the source meaning?',44,648,1410,50,29,true);
table(stats.table,712,216,[315,280,230,305,255]);
text('¹ A pair receives credit only when both answers are faithful. Other archive and diverse passages are counted individually.',65,956,1370,58,21);
text('c   Reach of a single edit on 1,024 diverse passages',44,1060,1410,50,29,true);
const chart=slide.charts.add('bar',{position:{left:48,top:1134,width:770,height:280},categories:['Unedited','1 cell','Up to 3','Up to 6'],hasLegend:false,
 barOptions:{direction:'bar',grouping:'clustered'},series:[{name:'Exact references',values:stats.series[0],fill:teal}],
 xAxis:{textStyle:{typeface:'Arial',fontSize:17}},yAxis:{min:0,max:200,majorUnit:50,textStyle:{typeface:'Arial',fontSize:17},majorGridlines:{fill:'#DBE6E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('Suppose one phrase were always correct',850,1140,585,62,24,true,teal);
text('The composer supplies the surrounding words. A perfect fill spanning up to three source cells could reproduce 60 complete reference translations. Six cells raise that ceiling to 182.',850,1210,580,190,23);
slide.speakerNotes.textFrame.setText('Each strip reproduces a complete saved pixel stream. The reach calculation has access to the recorded target and is an upper bound on exact reference recovery.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Phrase_boundary.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_65_phrase_boundary.png'),new Uint8Array(await png.arrayBuffer()));
