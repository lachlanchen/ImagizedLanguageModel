import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src = path.join(root,'references/evidence/passage_diversity_20261118');
const stats = JSON.parse(await fs.readFile(path.join(src,'figure.json')));

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
function table(rows,top,height,widths){
  const t=slide.tables.add({rows:rows.length,columns:widths.length,left:55,top,width:1385,height,columnWidths:widths,values:rows});
  t.styleOptions={headerRow:true,bandedRows:false};t.borders.assign({style:'solid',fill:'#D7E4E5',width:1});
  for(let i=0;i<rows.length;i++)for(let j=0;j<widths.length;j++){
    const cell=t.getCell(i,j);cell.fill=i===0?'#E7F0F0':'#FFFFFF';
    cell.text.style={typeface:'Arial',fontSize:23,bold:i===0,color:j===1?teal:ink};
  }
}
text('Passage diversity and expression frequency',44,20,1410,65,42,true);
text('a   Free writing on 17,862 new passages',44,112,1410,55,31,true);
table(stats.table,183,220,[350,245,245,290,255]);
text('b   Source passages and independently drawn answers',44,458,1410,50,31,true);
for(const [label,x,w] of [['Case',55,150],['Before the gap',234,330],['After the gap',639,330],['Output',1034,175],['Reference',1264,175]])text(label,x,530,w,42,24,true);
for(const [i,[key,label]] of [['gain','Gained'],['loss','Lost']].entries()){
  const y=596+i*82;text(label,55,y-2,160,42,24);
  for(const [part,x,w] of [['before',234,338],['after',639,338],['actual',1034,140],['reference',1264,140]])await image(key+'_'+part,x,y,w,34);
}
const f=JSON.parse(await fs.readFile(path.join(src,'frequency_weight.json'))).fresh;
text('c   Training weight and independent completion on '+f.rows.toLocaleString('en-US')+' further passages',44,815,1410,54,31,true);
const rows=[['Training weight','Exact answers','Rare answers correct','Distinct outputs','Top ten share']];
for(const [key,label] of [['unweighted','Uniform'],['weighted','Inverse square root']]){
 const a=f.arms[key],rare=a.arm_frequency.slice(1,3); rows.push([label,String(a.exact),rare.reduce((n,r)=>n+r.correct,0)+' / '+rare.reduce((n,r)=>n+r.cases,0).toLocaleString('en-US'),String(a.distinct),(100*a.most_frequent_ten/f.rows).toFixed(1)+'%']);
}
table(rows,890,170,[300,240,325,280,240]);
slide.speakerNotes.textFrame.setText('Both data conditions saw 571,512 windows. Both in pair requires two correct free answers to passages sharing their nearest glyphs. Panel b shows the first gained and lost source in the diverse-data condition, relative to repetition. Panel c counts rare expressions appearing one to twenty times in the combined training corpus.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Passage_diversity.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,10668000','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_44_passage_diversity.png'),new Uint8Array(await png.arrayBuffer()));
