import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation,PresentationFile,FileBlob} from '@oai/artifact-tool';

const [root,stage,skill,python]=process.argv.slice(2);
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const x=path.join(root,'references/evidence/visual_extractive_reading_20261118');
const y=path.join(root,'references/evidence/native_yan_cold_20261118');
const metrics=JSON.parse(await fs.readFile(path.join(x,'result.json'))).comparisons;
const deck=Presentation.create({slideSize:{width:1500,height:1230}});
const slide=deck.slides.add();
slide.background.fill='#FFFFFF';
const ink='#193849',teal='#16686b',rust='#98432D';
function label(text,left,top,width,height,size,bold=false,color=ink){
 const box=slide.shapes.add({geometry:'textbox',position:{left,top,width,height},fill:'none',line:{fill:'none',width:0}});
 box.text=text;box.text.style={typeface:'Arial',fontSize:size,bold,color,autoFit:'none'};
}
async function bitmap(dir,name,left,top,width,height){
 const png=await fs.readFile(path.join(dir,name+'.png'));
 slide.images.add({blob:png,contentType:'image/png',alt:name,fit:'contain',position:{left,top,width,height}});
}
label('Generating strokes and selecting an answer',44,22,1412,64,46,true);
label('a   Autonomous writing about the history of a character',44,119,1408,50,34,true);
label('Question',68,192,290,44,27);
await bitmap(y,'prompt',385,183,528,48);
label('Greedy output',68,269,282,45,27);
await bitmap(y,'greedy_answer',385,250,1024,128);
label('Sampled output',68,431,292,45,27);
await bitmap(y,'sampled_answer',385,411,1024,128);
label('b   Selecting a contiguous answer from a supplied passage',44,584,1410,53,34,true);
label('Question',68,661,250,45,28,true);
label('Returned pixels',896,661,270,45,28,true);
label('Reference',1211,661,240,45,28,true);
await bitmap(x,'first_exact_question',68,722,416,32);
await bitmap(x,'first_exact_answer',896,722,160,32);
await bitmap(x,'first_exact_answer',1211,722,160,32);
await bitmap(x,'first_case_question',68,795,768,32);
await bitmap(x,'first_case_answer',896,795,64,32);
await bitmap(x,'first_case_reference',1211,795,96,32);
label('c   Does the reader use the question?',44,900,1408,52,34,true);
const rows=[['Question control','Original question','Changed question']];
for(const [key,title] of [['blank','Blank question'],['swapped','Another question about the passage']]){
 const m=metrics[key];
 rows.push([title,m.normal+' / '+m.questions.toLocaleString('en-US'),m.control+' / '+m.questions.toLocaleString('en-US')]);
}
const t=slide.tables.add({rows:3,columns:3,left:58,top:980,width:1388,height:173,columnWidths:[665,350,373],values:rows});
t.styleOptions={headerRow:true,bandedRows:false}; t.borders.assign({style:'solid',fill:'#DCE5E6',width:1});
for(let r=0;r<3;r++)for(let c=0;c<3;c++){
 const cell=t.getCell(r,c);cell.fill=r===0?'#EAF1F2':'#FFFFFF';
 cell.text.style={typeface:'Arial',fontSize:25,bold:r===0,color:c===0?ink:(c===1?teal:rust)};
}
label('Counts give complete, pixel-exact answers. Both autonomous outputs reach 128 cells.',60,1167,1380,44,24);
slide.speakerNotes.textFrame.setText('Complete autonomous outputs from one checkpoint. Extraction examples show the first correct case and first case overall in the fixed order. All original PNGs retained. The passage extractor cannot write a missing fact.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/reading_writing.pptx'),finalPath=path.join(stage,'output/Visual_reading_writing.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,11715750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const preview=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_34_visual_reading_writing.png'),new Uint8Array(await preview.arrayBuffer()));
