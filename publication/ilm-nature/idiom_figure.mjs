import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/idiom_paper_20261118');
const stats = JSON.parse(await fs.readFile(path.join(src,'measurements.json')));

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
text('Generating an omitted Chinese expression from pictured prose',44,20,1440,66,42,true);
text('a   Complete outputs from the aligned image writer',44,116,1430,48,32,true);
for(const [v,x,w] of [['Case',55,155],['Before gap',227,336],['After gap',622,335],['Output',1030,188],['Reference',1263,188]])text(v,x,182,w,42,24,true);
for(const [i,[key,label]] of [['correct','Exact'],['miss','Error']].entries()){
  const y=265+90*i;text(label,55,y-4,150,42,25);
  for(const [suffix,x,width] of [['before',226,337],['after',625,337],['answer',1033,140],['reference',1265,140]])await image(key+'_'+suffix,x,y,width,34);
}
text('b   Free generation on 22,907 validation gaps',44,472,1420,50,32,true);
const rows=[['Writer','Intact context','Exchanged context','Distinct absent-glyph answers']];
for(const r of stats.models)rows.push([r.label,String(r.exact),String(r.exchanged),String(r.distinct)]);
const table=slide.tables.add({rows:rows.length,columns:4,left:59,top:551,width:1383,height:235,columnWidths:[366,283,301,433],values:rows});
table.styleOptions={headerRow:true,bandedRows:false};table.borders.assign({style:'solid',fill:'#DCE5E6',width:1});
for(let r=0;r<rows.length;r++)for(let c=0;c<4;c++){
 const cell=table.getCell(r,c);cell.fill=r===0?'#EAF1F2':'#FFFFFF';cell.text.style={typeface:'Arial',fontSize:23,bold:r===0,color:c===1?teal:ink};
}
text('c   Where the recorded expression was available',44,854,1420,52,32,true);
for(const [i,[label,n]] of [['Any of 33 generated proposals',stats.proposal_oracle],['Supplied training expressions',stats.inventory],['Local context count baseline',stats.local]].entries()){
 const x=62+i*470;text(label,x,939,462,40,23);text(String(n),x,989,445,58,38,true,teal);
}
slide.speakerNotes.textFrame.setText('Case order was fixed before selecting the first exact answer with all four glyphs absent and the first error. The image columns show the ten nearest cells on each side of the gap. The model sees 24 per side. All table counts require a complete image and a stop. The final column counts distinct exact expressions with at least one absent target glyph. The first number in panel c is an oracle upper bound, and the other two are separate controls.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Pictured_idioms.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,10668000','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_40_pictured_idioms.png'),new Uint8Array(await png.arrayBuffer()));
