import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/grounded_detail_writing_20261118');
const stats = JSON.parse(await fs.readFile(path.join(src,'qa.json'))).scores;

const deck = Presentation.create({slideSize:{width:1500,height:1120}});
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
text('Reading a question and writing its answer in another typeface',44,18,1435,65,44,true);
text('a   Two questions about the same Chinese passage',44,102,1410,50,33,true);
for(const [value,x,w] of [['Source excerpt',55,645],['Question',735,490],['Written answer',1250,238]])text(value,x,168,w,42,25,true);
for(const [i,k] of ['a','b'].entries()){
  const y=231+75*i;
  await image('passage_'+k,60,y,635,30);
  const qw=k==='a'?477:342;
  await image('pair_question_'+k,738,y,qw,27);
  await image('pair_answer_'+k,1255,y,k==='a'?127:159,32);
}
text('b   Complete answers on official development questions',44,410,1412,50,33,true);
const rows=[['Visible input','Questions','Correct source span','Exact Serif answer']];
for(const [key,label] of [['normal','Intact question'],['exchange','Exchanged question'],['blank_question','Blank question'],['blank_passage','Blank passage']]){
  const r=stats[key];rows.push([label,String(r.count),String(r.source_exact),String(r.detail_exact)]);
}
const table=slide.tables.add({rows:rows.length,columns:4,left:58,top:482,width:1385,height:267,columnWidths:[432,215,369,369],values:rows});
table.styleOptions={headerRow:true,bandedRows:false};table.borders.assign({style:'solid',fill:'#DCE5E6',width:1});
for(let r=0;r<rows.length;r++)for(let c=0;c<4;c++){
  const cell=table.getCell(r,c);cell.fill=r===0?'#EAF1F2':'#FFFFFF';
  cell.text.style={typeface:'Arial',fontSize:23,bold:r===0,color:c===3?teal:ink};
}
text('c   First six development glyphs absent from all training passages',44,796,1412,52,32,true);
for(const [i,[key,label]] of [['source','Sans input'],['count','Local patch writer'],['combined','Neural fallback'],['target','Serif reference']].entries()){
  const x=55+i*360;text(label,x,895,332,44,24,true);
  await image('unseen_'+key,x,974,300,50);
}
slide.speakerNotes.textFrame.setText('The passage is the same throughout panel a. Each original source excerpt contains the selected answer. Both independently written answers match a human reference in Serif. Panel b counts complete pixel matches. The first six out-of-training Han pictures in panel c follow the original bitmap order. Their shapes were unseen during motor fitting.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/detail.pptx'), finalPath=path.join(stage,'output/Grounded_detail.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,10668000','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_38_grounded_detail.png'),new Uint8Array(await png.arrayBuffer()));
