import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/patch_count_writing_20261118');
const stats = JSON.parse(await fs.readFile(path.join(src,'result.json'))).stats;
const audit = JSON.parse(await fs.readFile(path.join(src,'audit.json')));
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
text('Drawing Chinese continuations from ink patches',44,20,1414,64,46,true);
text('a   The next image after a visible passage',44,108,1412,53,34,true);
text('Outcome',64,182,240,42,26,true);
text('End of the context',355,182,637,42,26,true);
text('Drawn',1005,182,165,42,26,true);
text('Source',1247,182,195,42,26,true);
for(const [i,[key,name]] of [['exact','Exact'],['wrong_ink','Incorrect ink'],['blank','Blank']].entries()){
  const top=238+i*70;
  text(name,65,top+1,270,47,27);
  await image(key+'_context',358,top,480,40);
  await image(key+'_generated',1009,top,44,44);
  await image(key+'_reference',1252,top,44,44);
}
text('b   Free successor images across 334 held-out articles',44,483,1420,55,34,true);
const rows=[['Outcome among 668 images','Ordered training','Rotated training']];
rows.push(['Exact recorded successor',...['ordered','rotated'].map(k=>String(stats[k].exact))]);
rows.push(['Known nonempty glyph',...['ordered','rotated'].map(k=>String(audit[k].observed_nonempty))]);
rows.push(['Blank output',...['ordered','rotated'].map(k=>String(stats[k].blank_emissions))]);
const table=slide.tables.add({rows:rows.length,columns:3,left:58,top:555,width:1386,height:230,columnWidths:[674,356,356],values:rows});
table.styleOptions={headerRow:true,bandedRows:false};
table.borders.assign({style:'solid',fill:'#DCE5E6',width:1});
for(let r=0;r<rows.length;r++)for(let c=0;c<3;c++){
  const cell=table.getCell(r,c);
  cell.fill=r===0?'#EAF1F2':'#FFFFFF';
  cell.text.style={typeface:'Arial',fontSize:25,bold:r===0,color:c===1?teal:ink};
}
text('c   Complete output to the question about the history of 言',44,835,1420,56,34,true);
await image('ordered_yan',68,924,1024,128);
slide.speakerNotes.textFrame.setText('First exact, first nonempty incorrect and first blank outputs in the ordered trial. White space under Drawn in the third row is the actual all-white output. Each displayed context contains the last 12 observed images. The full etymology response reaches the 128-cell cap.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/patches.pptx'), finalPath=path.join(stage,'output/Patch_writing.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,10668000','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_36_patch_writing.png'),new Uint8Array(await png.arrayBuffer()));
