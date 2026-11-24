import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {finalizePresentation, applyPresentationChartFont} = await import(pathToFileURL(path.join(skill, 'container_tools/artifact_tool_utils.mjs')));
const base = path.join(root, 'references/evidence');
const story = path.join(base, 'visual_story_writer_v1');
const read = async p => JSON.parse(await fs.readFile(p));
const deck = Presentation.create({slideSize:{width:1088, height:538}});
const ink = '#182526';
function label(slide, text, x, y, w, h, size, bold, color=ink) {
  const box = slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  box.text = text;
  box.text.style = {typeface:'Arial',fontSize:size,bold,color,autoFit:'none',insets:{left:0,right:0,top:0,bottom:0}};
}
for(const plate of await read(path.join(story,'plates.json'))) {
  const slide = deck.slides.add();
  slide.background.fill = '#FFFFFF';
  for(const r of plate.items) {
    const x=2*r.x,y=2*r.y;
    if(r.type==='text') {
      label(slide,r.value,x,y,1088-x,Math.min(538-y,2.8*r.size+4),2*r.size,r.bold,r.color);
      continue;
    }
    const position={left:x,top:y,width:2*r.w,height:2*r.h};
    if(r.type==='image') {
      slide.images.add({blob:await fs.readFile(path.join(story,'plates',r.path)),contentType:'image/png',alt:r.path,fit:'contain',position});
    } else {
      slide.shapes.add({geometry:'rect',position,fill:r.color,line:{fill:'none',width:0}});
    }
  }
  slide.speakerNotes.textFrame.setText(plate.name+'. Original measured ink. Historical source: https://hanziyuan.net/. Reading sources: https://aclanthology.org/D13-1020/ and https://github.com/nlpdata/c3. English and Chinese examples include the second question on each page.');
}
const slide=deck.slides.add();
slide.background.fill='#FFFFFF';
const forms=await read(path.join(base,'visual_form_balance_v1/forms.json'));
const sem=(await read(path.join(story,'semantic.json'))).summary;
label(slide,'Reading a period label and answering a question',25,20,1040,47,30,true);
label(slide,'All 16 interventions correct',25,83,485,36,22,true);
const values=[['Writer','Families / 64'],...['page','fixed','balanced'].map(m=>[m[0].toUpperCase()+m.slice(1),String(forms[m].all_16_per_family)])];
const t=slide.tables.add({rows:4,columns:2,left:25,top:139,width:412,height:198,columnWidths:[204,208],values});
t.styleOptions={headerRow:true,bandedRows:false};
t.borders.assign({style:'solid',fill:'#D4DFD9',width:1});
for(let i=0;i<4;i++)for(let j=0;j<2;j++){
  const cell=t.getCell(i,j);
  cell.fill=i?'#FFFFFF':'#E9F1ED';
  cell.text.style={typeface:'Arial',fontSize:19,bold:i===0,color:ink};
}
label(slide,'Two requested forms × two languages × two orders × two assignments of images.',25,369,412,102,18,false);
label(slide,'Grounded replies in both grading orders',474,83,590,36,22,true);
const chart=slide.charts.add('bar',{barOptions:{direction:'column',grouping:'clustered'},
  position:{left:475,top:135,width:587,height:325},categories:['MCTest','C³','Story','Exam'],
  series:['parent','strip','page','minimind'].map((m,i)=>({name:m[0].toUpperCase()+m.slice(1),values:['mctest','c3','stories','exam'].map(k=>sem[k][m]),fill:['#A99887','#7896AA','#006B66','#B38838'][i]})),
  hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:14}},yAxis:{min:0,max:16,majorUnit:4,textStyle:{fontSize:13}},xAxis:{textStyle:{fontSize:14}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
slide.speakerNotes.textFrame.setText('Development selected the intervention. Historical results use separate sealed families. Qwen3-8B judged sixteen modern answers per benchmark in two orders. MiniMind received the transcript.');
for(const folder of ['draft','output']) await fs.mkdir(path.join(stage,folder),{recursive:true});
const candidatePath=path.join(stage,'draft/forms.pptx');
const finalPath=path.join(stage,'output/Visual_forms_and_reading.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:3,materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[3],requiredNativeTableOwnerSlides:[3],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','10363200,5124450','--require-native-table-slide','3'],receiptPath:path.join(stage,'presentation_validation.json')});
