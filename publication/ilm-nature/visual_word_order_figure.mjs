import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {applyPresentationChartFont, finalizePresentation} = await import(
  pathToFileURL(path.join(skill, 'container_tools/artifact_tool_utils.mjs')).href);
const data = JSON.parse(await fs.readFile(path.join(root, 'references/evidence/visual_word_order_20261118.json')));
const deck = Presentation.create({slideSize:{width:1500,height:1230}});
const slide = deck.slides.add();
slide.background.fill='#FFFFFF';
const ink='#193849', teal='#197478', rust='#A64F39', grey='#788990', font='Arial';
function label(text,left,top,width,height,size,bold=false,color=ink) {
  const box=slide.shapes.add({geometry:'textbox',position:{left,top,width,height},fill:'none',line:{fill:'none',width:0}});
  box.text=text;
  box.text.style={typeface:font,fontSize:size,bold,color,autoFit:'none'};
}
label('Mandarin word order in a visual reader',42,18,1400,70,45,true);
label('a   Two sentence images with the same ink',42,113,1410,54,33,true);
for (const [i, side] of ['good','bad'].entries()){
  const y=184+i*78;
  label(i===0?'Grammatical order':'Reordered alternative',63,y+3,425,50,29);
  const image=await fs.readFile(path.join(stage,side+'.png'));
  slide.images.add({blob:image,contentType:'image/png',fit:'contain',alt:side+' Mandarin benchmark input',
    position:{left:503,top:y,width:576,height:64}});
}
label('b   Grammatical preference during training (% correct)',42,370,1400,56,33,true);
const steps=['0','3447','6893','10339','13785'];
const percent=key=>Number((100*data.metrics[key].macro_accuracy).toFixed(10));
const bigram=percent('bigram');
const series=[
  {name:'Reader',values:steps.map(percent),line:{fill:teal,width:6}},
  {name:'Visual bigram',values:steps.map(()=>bigram),line:{fill:rust,width:3}},
  {name:'Chance',values:steps.map(()=>50),line:{fill:grey,width:2}}
];
const chart=slide.charts.add('line',{position:{left:62,top:443,width:1015,height:350},series,
  categories:steps.map(s=>Number(s).toLocaleString('en-US')),hasLegend:false,lineOptions:{smooth:false},
  xAxis:{textStyle:{typeface:font,fontSize:22}},yAxis:{min:35,max:65,majorUnit:5,
  textStyle:{typeface:font,fontSize:22},majorGridlines:{fill:'#DCE4E6',width:1}}});
applyPresentationChartFont(chart,{fontFamily:font});
label('Visual bigram',1110,486,347,46,27,true,rust);
label('Trained reader',1110,567,347,46,27,true,teal);
label('Chance',1110,621,347,46,27,true,grey);
label('Optimizer updates',480,789,420,43,24);
label('c   Variation across 23 constructions',42,868,1400,56,33,true);
const values=[['Quantity','Estimate','95% interval'],
 ['Final accuracy','51.4%','43.8 to 59.4%'],['Change from initialization','−2.7 pp','−19.6 to +14.7 pp'],
 ['Difference from visual bigram','−5.7 pp','−16.8 to +6.3 pp']];
const table=slide.tables.add({rows:4,columns:3,left:54,top:941,width:1386,height:221,
  columnWidths:[678,270,438],values});
table.styleOptions={headerRow:true,bandedRows:false};
table.borders.assign({style:'solid',fill:'#DCE5E6',width:1});
for(let r=0;r<4;r++)for(let c=0;c<3;c++){
 const cell=table.getCell(r,c);cell.fill=r===0?'#EAF1F2':'#FFFFFF';cell.text.style={typeface:font,fontSize:26,bold:r===0,color:ink};
}
label('552 pairs, 24 per construction. Differences are in percentage points (pp).',54,1172,1407,48,24);
slide.speakerNotes.textFrame.setText('ZhoBLiMP. The original rasterizations in panel a are supplied inputs. Final results are from complete-sentence density and stopping scores. Intervals use 10,000 paired paradigm bootstraps.');
await fs.mkdir(path.join(stage,'draft'));await fs.mkdir(path.join(stage,'output'));
const candidatePath=path.join(stage,'draft/word_order.pptx'),finalPath=path.join(stage,'output/Visual_word_order.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:1,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],
 verifyArtifactToolImport:true,integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','14287500,11715750','--validate-heading-fit','--require-native-table-slide','1'],
 receiptPath:path.join(stage,'presentation_validation.json')});
const reopened=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const preview=await reopened.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_33_visual_word_order.png'),new Uint8Array(await preview.arrayBuffer()));
