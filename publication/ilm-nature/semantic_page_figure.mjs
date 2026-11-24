import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const src=path.join(root,'references/evidence/semantic_page_v1');
const stats = JSON.parse(await fs.readFile(path.join(src,'figure.json')));

const deck = Presentation.create({slideSize:{width:1500,height:1470}});
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
    cell.text.style={typeface:'Arial',fontSize:22,bold:i===0,color:j===1?teal:ink};
  }
}



async function bitmap(name,x,y,width,height){
 const data=await fs.readFile(path.join(src,name+'.png'));
 const w=data.readUInt32BE(16),h=data.readUInt32BE(20),scale=Math.min(width/w,height/h);
 await image(name,x,y,w*scale,h*scale);
}
text('Writing a sentence from a pictured record',48,20,1390,66,42,true);
text('1.37 M writer parameters    /    33,525 human descriptions    /    1,847 held combinations',53,89,1390,45,24,false,teal);
text('a   Change one fact and read the two independently generated answers',50,156,1400,50,29,true);
text('Chinese input card',56,224,430,40,23,true,teal);
await bitmap('adult-card',60,275,438,236);
text('Family policy: no',543,224,425,40,22,true,teal);
await bitmap('adult-answer',535,266,900,151);
text('Same card, family policy: yes',543,425,850,40,22,true,teal);
await bitmap('family-answer',535,466,900,144);
text('First record: unsupported price',58,635,700,40,22,true,teal);
text('Last record: incorrect landmark',782,635,660,40,22,true,teal);
await bitmap('first-answer',54,686,683,120);
await bitmap('last-answer',775,686,668,120);
text('b   Complete fact recovery with equivalent price and rating bands (%)',50,824,1425,46,28,true);
const chart=slide.charts.add('bar',{barOptions:{direction:'column',grouping:'clustered'},
 position:{left:50,top:878,width:900,height:280},categories:['English image','Chinese image','English bytes','Chinese bytes'],
 hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:18}},
 series:['Greedy','Beam','Visual readback'].map((name,i)=>({name,values:stats.series[i],fill:['#A9C0C1','#C6A46C','#16686B'][i]})),
 xAxis:{textStyle:{fontSize:17}},yAxis:{min:0,max:100,majorUnit:25,textStyle:{fontSize:17},majorGridlines:{fill:'#DCE5E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
text('Human reference check',982,899,435,45,24,true,teal);
text('The same pattern reader accepted 4,396 of 4,693 human descriptions with aliases merged and 1,729 under the literal rule.',982,953,425,165,23);
text('c   Literal facts and sensitivity to a change in the record',50,1185,1395,52,29,true);
table(stats.table,1250,173,[300,340,410,335]);
slide.speakerNotes.textFrame.setText('Actual released pixels. The first and last test cases contain errors. Identical eight candidates are available to the two search selectors.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/idiom.pptx'), finalPath=path.join(stage,'output/Semantic_page.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,14001750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const png=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_139_semantic_page.png'),new Uint8Array(await png.arrayBuffer()));
