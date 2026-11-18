import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/shared_layout_writing_20261118');
const stats = JSON.parse(await fs.readFile(path.join(src,'result.json'))).arms.ordered;

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
text('A shared ink layout improves the shape of generated writing',44,20,1414,64,44,true);
text('a   Next glyph after the visible passage',44,104,1412,53,34,true);
for (const [name,x,w] of [['Outcome',63,245],['End of passage',340,493],['Earlier writer',860,210],['Shared layout',1080,205],['Source',1335,130]]) {
  text(name,x,179,w,45,25,true);
}
for(const [i,[key,name]] of [['corrected','Newly correct'],['wrong_han','Wrong character'],['lost','Lost match']].entries()){
  const top=240+i*70;
  text(name,63,top+1,265,47,26);
  await image(key+'_context',343,top,444,37);
  await image(key+'_old',906,top,46,46);
  await image(key+'_new',1130,top,46,46);
  await image(key+'_target',1347,top,46,46);
}
text('b   Generated images in 668 contexts from 334 Chinese articles',44,480,1420,55,33,true);
const rows=[['Outcome','Earlier writer','Shared layout']];
for(const [key,label] of [['known_nonempty','Known nonempty glyph'],['blank','Blank'],['exact','Exact successor']]){
  rows.push([label,String(stats.previous[key]),String(stats.free[key])]);
}
const table=slide.tables.add({rows:rows.length,columns:3,left:58,top:555,width:1386,height:230,columnWidths:[674,356,356],values:rows});
table.styleOptions={headerRow:true,bandedRows:false};
table.borders.assign({style:'solid',fill:'#DCE5E6',width:1});
for(let r=0;r<rows.length;r++)for(let c=0;c<3;c++){
  const cell=table.getCell(r,c);
  cell.fill=r===0?'#EAF1F2':'#FFFFFF';
  cell.text.style={typeface:'Arial',fontSize:25,bold:r===0,color:c===1?teal:ink};
}
text('c   Complete shared-layout output to the question about 言',44,835,1420,56,34,true);
await image('ordered_yan',68,924,1024,128);
slide.speakerNotes.textFrame.setText('Fixed-order first examples of a gained match, an incorrect Han bitmap after an earlier non-Han output, and a lost match. The blank in the first row is the actual earlier output. The full answer reached its 128-cell cap.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/layout.pptx'), finalPath=path.join(stage,'output/Shared_layout.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,10668000','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_37_shared_layout.png'),new Uint8Array(await png.arrayBuffer()));
