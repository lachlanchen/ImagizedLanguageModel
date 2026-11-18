import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/visual_titles_20261118');
const stats = JSON.parse(await fs.readFile(path.join(src,'result.json'))).scores;

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
text('Writing Chinese article titles from ordered image memory',44,20,1440,65,44,true);
text('a   Output from the independent free writer',44,110,1410,48,32,true);
await image('native_eight',60,185,268,34);
text('Readable final answers    0 / 32',418,178,540,53,29);
text('Coherent books    0 / 12',983,178,445,53,29);
text('b   Exact complete titles on held-out articles',44,292,1410,48,32,true);
const rows=[['Reference in pictured prompt','Articles','Full writer','One span','Exchanged article']];
for(const [k,label] of [['contiguous','One span'],['distributed','Distributed'],['missing_glyph','Absent glyph']]){
  const s=stats[k];rows.push([label,String(s.n),...['normal','contiguous','exchange'].map(a=>String(s[a].exact))]);
}
const table=slide.tables.add({rows:4,columns:5,left:58,top:368,width:1380,height:218,columnWidths:[400,190,210,210,370],values:rows});
table.styleOptions={headerRow:true,bandedRows:false};table.borders.assign({style:'solid',fill:'#DCE5E6',width:1});
for(let r=0;r<4;r++)for(let c=0;c<5;c++){
  const cell=table.getCell(r,c);cell.fill=r===0?'#EAF1F2':'#FFFFFF';
  cell.text.style={typeface:'Arial',fontSize:23,bold:r===0,color:c===2?teal:ink};
}
text('c   First development article in each group',44,642,1410,48,32,true);
for(const [v,x,w] of [['Group',55,220],['Article opening',302,440],['Model output',816,350],['Recorded title',1165,302]])text(v,x,723,w,45,23,true);
const examples=JSON.parse(await fs.readFile(path.join(src,'examples.json')));
for(const [j,[k,label]] of [['contiguous','One span'],['distributed','Distributed'],['missing_glyph','Absent glyph']].entries()){
  const y=800+95*j;text(label,55,y-4,230,43,24);
  for(const [suffix,x,maxWidth] of [['source',301,466],['output',815,312],['target',1163,303]]){
    const count=examples[k][suffix];
    if(count>0)await image(k+'_'+suffix,x,y,Math.min(count*32,maxWidth),35);
    else text('Empty output',x,y,maxWidth,39,23);
  }
}
slide.speakerNotes.textFrame.setText('Physical article inputs and outputs. The first eight cells summarize the recurrent failure of the separate free writer; all original streams were read before scoring. A correct title requires a learned stop and exact complete output.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/title.pptx'), finalPath=path.join(stage,'output/Visual_titles.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,10668000','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_39_visual_titles.png'),new Uint8Array(await png.arrayBuffer()));
