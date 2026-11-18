import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/historical_plates_20261118');
const stats = JSON.parse(await fs.readFile(path.join(src,'spans.json')));

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
text('Reading a historical form from an illustrated entry',44,20,1440,65,42,true);
text('a   The same plate answers two different requests',44,100,1440,55,32,true);
text('Hanziyuan plate',68,175,380,42,24,true);
await image('yan_plate',74,227,270,216);
text('Output on CPU',428,175,600,42,24,true);
text('English: oracle, then bronze',428,238,540,43,25);
await image('yan_answer',1036,218,168,84);
text('Chinese: bronze, then oracle',428,350,565,43,25);
await image('yan_reverse',1036,326,168,84);
text('b   Complete answers on 600 held-out plates',44,477,1410,48,32,true);
const rows=[['Evidence and prompt','Chinese, one','Chinese, two','English, one','English, two','Total']];
for(const [name,label] of [['intact','Original frame'],['paraphrase','Second frame'],['third','Third frame'],['relocated','Pictures moved']]){
 const arm=stats.arms[name];rows.push([label,...['zh_1','zh_2','en_1','en_2'].map(k=>String(arm.groups[k])),String(arm.exact)]);
}
const table=slide.tables.add({rows:rows.length,columns:6,left:59,top:550,width:1383,height:220,columnWidths:[300,224,224,224,224,187],values:rows});
table.styleOptions={headerRow:true,bandedRows:false};table.borders.assign({style:'solid',fill:'#DCE5E6',width:1});
for(let r=0;r<rows.length;r++)for(let c=0;c<6;c++){
 const cell=table.getCell(r,c);cell.fill=r===0?'#EAF1F2':'#FFFFFF';cell.text.style={typeface:'Arial',fontSize:23,bold:r===0,color:c===1?teal:ink};
}
text('c   Held-out examples and a headword ambiguity',44,808,1410,46,32,true);
text('Ma: bronze and seal',63,891,420,43,25);
text('Nai: oracle requested',542,891,425,43,25);
text('Reference for Nai',1044,891,395,43,25);
await image('ma_answer',87,946,156,78);await image('miss_answer',609,946,78,78);await image('miss_reference',1118,946,78,78);
text('B14639 and S06958',63,1038,430,42,22,false,teal);
text('J21031 (Wen)',546,1038,425,42,22,false,teal);text('J11475 (Nai)',1044,1038,390,42,22,false,teal);
slide.speakerNotes.textFrame.setText('The training-family Yan illustration uses historical images J04919 and B02976. Panel b counts exact stopped image sequences under three question frames and after cyclic relocation of all payloads in the third frame. Each language and length subgroup contains 2,400 questions. The same 600 plates are reused. Panel c gives the first development two-form request and the first incorrect development request in source order. The competing headword Wen also occurs within the oracle-stage name.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Historical_plates.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,10668000','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_42_historical_plates.png'),new Uint8Array(await png.arrayBuffer()));
