import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/expression_preference_20261118');
const stats = JSON.parse(await fs.readFile(path.join(src,'summary.json')));

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
text('How training changes independent expression writing',44,18,1425,64,41,true);
text('a   Complete answers on candidate-free Chinese passages',44,103,1425,54,31,true);
const rows=stats.table;
const table=slide.tables.add({rows:rows.length,columns:5,left:59,top:179,width:1383,height:306,columnWidths:[375,252,252,252,252],values:rows});
table.styleOptions={headerRow:true,bandedRows:false};table.borders.assign({style:'solid',fill:'#DCE5E6',width:1});
for(let r=0;r<rows.length;r++)for(let c=0;c<5;c++){
 const cell=table.getCell(r,c);cell.fill=r===0?'#EAF1F2':'#FFFFFF';cell.text.style={typeface:'Arial',fontSize:23,bold:r===0||r===4,color:c===1?teal:ink};
}
text('b   Two complete outputs from the joint reference writer',44,535,1420,48,31,true);
for(const [v,x,w] of [['Case',55,155],['Before gap',227,336],['After gap',622,335],['Output',1030,188],['Reference',1263,188]])text(v,x,598,w,42,24,true);
for(const [i,[key,label]] of [['gain','Gained'],['loss','Lost']].entries()){
 const y=672+i*83;text(label,55,y-3,150,42,24);
 for(const [suffix,x,w] of [['before',226,337],['after',625,337],['output',1033,140],['reference',1265,140]])await image(key+'_'+suffix,x,y,w,34);
}
text('c   Correct choices with two answer images supplied',44,850,1420,52,31,true);
const paired=JSON.parse(await fs.readFile(path.join(src,'joint_pairs.json')));
for(const [i,[key,label]] of [['guided','Published writer'],['reference','Joint reference'],['preference','Joint preference']].entries()){
 const x=62+i*471;const a=paired[key].all;text(label,x,933,455,44,25);text(a.both_correct+' / '+a.pairs.toLocaleString('en-US')+' pairs',x,994,450,62,39,true,teal);
}
slide.speakerNotes.textFrame.setText('Each column in panel a is a separate passage collection. The local hint fits used the first two displayed collections. Joint fits used the first three. The reference-only model was frozen before the final 17,555-passage confirmation. Dashes denote unevaluated conditions. Panel b shows the first validation result gained and lost by the joint reference fit compared with the published writer, using original bitmaps. Panel c counts both answers preferred over their paired competitors among the 3,126 matched pairs from the 8,790-passage collection. Supplying the two answer trajectories is a distinct diagnostic.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Expression_preference.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,10668000','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_43_expression_preference.png'),new Uint8Array(await png.arrayBuffer()));
