import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/mixed_ink_20261118');
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
text('Answering a pictured question about a passage',44,12,1410,66,42,true);
text('a   One passage, two questions',44,91,1400,45,30,true);
text('Pictured question',62,154,700,36,23,true,teal);
text('Emitted source ink',970,154,435,36,23,true,teal);
for (const [i,k] of ['a','b'].entries()) {
  const y=207+i*62;
  await image('pair_'+k+'_question',62,y,k==='a'?423:555,42);
  await image('pair_'+k+'_actual',970,y,k==='a'?252:360,42);
}
text('b   Exact answers on official Chinese reading questions',44,353,1410,48,30,true);
const chart=slide.charts.add('bar',{position:{left:65,top:421,width:885,height:258},
  categories:['New reader','Earlier reader','Question exchanged'], hasLegend:false,
  barOptions:{direction:'column',grouping:'clustered'},series:[{name:'Complete answers',values:stats.control_counts,fill:teal}],
  xAxis:{textStyle:{typeface:'Arial',fontSize:21}}, yAxis:{min:0,max:1200,majorUnit:300,
  textStyle:{typeface:'Arial',fontSize:19},majorGridlines:{fill:'#DCE4E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('1,000 / 3,219',988,425,430,58,40,true,teal);
text('All official questions',988,485,440,40,24);
text('49 inputs exceeded font coverage.',988,539,445,68,22);
text('7 / 3,158 after question exchange',988,610,445,62,22);
text('c   Three requests about a gap in a new passage',44,703,1410,49,30,true);
table(stats.table,771,250,[767,325,293]);
text('d   First incorrect official answer',44,1051,1410,48,30,true);
text('Question',62,1119,590,38,23,true,teal);
text('Emitted',716,1119,360,38,23,true,teal);
text('Reference',1122,1119,315,38,23,true,teal);
await image('failure_question',62,1181,617,35);
await image('failure_actual',716,1181,364,35);
await image('failure_reference',1122,1181,303,35);
slide.speakerNotes.textFrame.setText('Original source and emitted images. The two questions concern railway length and operator. Chart counts exact stopped source images. The first two conditions share all 3,219 official questions; 3,170 had renderable inputs. Exchange requires a second question on the same passage. The table uses a separate held-out set of 17,600 cloze windows and one unseen phrasing per instruction. The requested neighbors exist on 16,089 windows.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Mixed_ink.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,12382500','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_47_mixed_ink.png'),new Uint8Array(await png.arrayBuffer()));
