import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/writer_transfer_20261118');
const stats = JSON.parse(await fs.readFile(path.join(src,'result.json')));

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
text('Learning to write the expressions a visual reader recognizes',44,20,1440,66,42,true);
text('a   Free writing after training with pictured answers',44,116,1430,48,32,true);
for(const [v,x,w] of [['Case',55,155],['Before gap',227,336],['After gap',622,335],['Output',1030,188],['Reference',1263,188]])text(v,x,182,w,42,24,true);
for(const [i,[key,label]] of [['gain','Gained'],['miss','Missed']].entries()){
  const y=265+90*i;text(label,55,y-4,150,42,25);
  for(const [suffix,x,width] of [['before',226,337],['after',625,337],['output',1033,140],['reference',1265,140]])await image(key+'_'+suffix,x,y,width,34);
}
text('b   Complete expressions on two independent sets of passages',44,459,1420,50,32,true);
const rows=[['Training target','Dev. exact','Of 930 read','Fresh exact','Of 398 read']];
for(const [name,label] of [['published','Initial writer'],['original_targets','Original targets'],['distilled_targets','Reader-guided targets']]){
  const dev=stats.development.arms[name],fresh=stats.fresh.arms[name];
  rows.push([label,String(dev.exact),String(dev.reader_correct),String(fresh.exact),String(fresh.reader_correct)]);
}
const table=slide.tables.add({rows:rows.length,columns:5,left:59,top:530,width:1383,height:219,columnWidths:[371,253,253,253,253],values:rows});
table.styleOptions={headerRow:true,bandedRows:false};table.borders.assign({style:'solid',fill:'#DCE5E6',width:1});
for(let r=0;r<rows.length;r++)for(let c=0;c<5;c++){
 const cell=table.getCell(r,c);cell.fill=r===0?'#EAF1F2':'#FFFFFF';cell.text.style={typeface:'Arial',fontSize:23,bold:r===0,color:c===1?teal:ink};
}
text('c   Which part of the passage helps the writer?',44,804,1420,52,32,true);
const masks=JSON.parse(await fs.readFile(path.join(src,'context_distance.json')));
for(const [i,[key,label]] of [['intact','Full window'],['nearest_eight','Eight nearest cells'],['outside_eight','Only the outer cells']].entries()){
  const x=62+i*470;text(label,x,905,458,45,26);text(String(masks[key].exact),x,963,445,63,44,true,teal);
}
slide.speakerNotes.textFrame.setText('Panel a shows the first new exact development completion and the first wrong completion whose reference the supplied-expression reader selected correctly. The nearest twelve cells are shown on each side. Panel b counts complete exact stopped image sequences. Dev. includes 22,907 gaps and fresh includes 9,135 gaps from previously unused source groups. The read columns count the subset of reader-correct cases written correctly. Panel c measures the same distilled writer on all 9,135 fresh cases with different visible context.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Writer_transfer.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,10668000','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_41_writer_transfer.png'),new Uint8Array(await png.arrayBuffer()));
