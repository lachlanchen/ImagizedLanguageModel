import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/visual_successor_20261118');
const visual = JSON.parse(await fs.readFile(path.join(src,'visual.json')));
const char = JSON.parse(await fs.readFile(path.join(src,'character.json')));
const deck = Presentation.create({slideSize:{width:1500,height:1080}});
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
text('Choosing the next glyph from context',44,20,1414,62,46,true);
text('a   Two passages ending in the same glyph',44,106,1412,54,34,true);
text('32 preceding images',66,179,1110,44,27,true);
text('Actual successor',1170,179,304,44,27,true);
for(const [i,letter] of ['a','b'].entries()){
  const top=239+i*74;
  await image('history_'+letter,66,top+5,1024,32);
  await image('next_'+letter,1180,top,48,48);
}
text('b   Assignment and independent choice across 512 articles',44,408,1420,56,34,true);
const rows=[['Predictor','Correct assignment','Both choices correct']];
for(const [key,label] of [
  ['native_0_normal','Native writer, initial'],['native_13785_normal','Native writer, final'],
  ['initial_normal','Compact visual reader, initial'],['rotated_normal','Visual reader, rotated training'],
  ['aligned_normal','Visual reader, aligned training'],['character','Character-indexed reader, aligned training'],
  ['trigram_normal','Physical trigram']]){
  const m=key==='character'?char.final:visual.stats[key];
  rows.push([label,(100*m.accuracy).toFixed(2)+'%',String(m.both_strict)]);
}
const table=slide.tables.add({rows:rows.length,columns:3,left:58,top:484,width:1386,height:438,columnWidths:[670,340,376],values:rows});
table.styleOptions={headerRow:true,bandedRows:false};
table.borders.assign({style:'solid',fill:'#DCE5E6',width:1});
for(let r=0;r<rows.length;r++)for(let c=0;c<3;c++){
  const cell=table.getCell(r,c);
  cell.fill=r===0?'#EAF1F2':'#FFFFFF';
  cell.text.style={typeface:'Arial',fontSize:25,bold:r===0,color:c===1?teal:ink};
}
text('Assignment gives half credit for ties. Independent choice requires each history to favour its own successor.',62,958,1390,90,25);
slide.speakerNotes.textFrame.setText('Both 32-cell histories and both successors are the original rasters from the first article in the fixed order. Conditional scores use both candidate images. The compact fits use two CPU threads.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/successors.pptx'), finalPath=path.join(stage,'output/Visual_successors.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,10287000','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_35_visual_successors.png'),new Uint8Array(await png.arrayBuffer()));
