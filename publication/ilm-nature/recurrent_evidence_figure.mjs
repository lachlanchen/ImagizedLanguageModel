import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation,PresentationFile} from '@oai/artifact-tool';
const [root,stage,skill,python]=process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')));
const folder=path.join(root,'references/evidence/recurrent_evidence_v1');
const plates=JSON.parse(await fs.readFile(path.join(folder,'reading-plates.json')));
const deck=Presentation.create({slideSize:{width:1088,height:538}});
const ink='#182526',teal='#006B66';
function label(slide,txt,x,y,w,h,size,bold,color){
  const box=slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  box.text=txt;
  box.text.style={typeface:'Arial',fontSize:size,bold,color,autoFit:'none',insets:{left:0,right:0,top:0,bottom:0}};
}
for(const plate of plates){
  const slide=deck.slides.add();slide.background.fill='#FFFFFF';
  for(const r of plate.items){
    const x=2*r.x,y=2*r.y;
    if(r.type==='text'){
      label(slide,r.value,x,y,1088-x,Math.min(538-y,2.8*r.size+4),2*r.size,r.bold,r.color);
      continue;
    }
    const position={left:x,top:y,width:2*r.w,height:2*r.h};
    if(r.type==='image'){
      slide.images.add({blob:await fs.readFile(path.join(folder,'plates',r.path)),contentType:'image/png',alt:r.path,fit:'contain',position});
    }else{
      slide.shapes.add({geometry:'rect',position,fill:r.color,line:{fill:'none',width:0}});
    }
  }
  slide.speakerNotes.textFrame.setText(plate.name+'. Each question, evidenced mark and generated response is a measured bitmap embedded at the indicated crop.');
}
const slide=deck.slides.add();slide.background.fill='#FFFFFF';
label(slide,'From reading a fact to writing a sentence',24,16,1025,44,31,true,ink);
label(slide,'Complete matches to human statements',24,69,496,33,19,true,ink);
label(slide,'Fresh passages and changed facts',552,69,520,33,19,true,ink);
const pub=path.join(root,'references/evidence/visual_statements_v1');
const scores=await Promise.all(['before','trained'].map(async n=>JSON.parse(await fs.readFile(path.join(pub,n+'-scores.json')))));
const chart=slide.charts.add('bar',{barOptions:{direction:'column',grouping:'clustered'},position:{left:24,top:113,width:499,height:332},
  categories:['Fact supplied (128)','Page supplied (96)'],series:scores.map((r,i)=>({name:i?'After training':'Before training',values:[r.given.exact,r.pipeline.exact],fill:i?teal:'#A1B3AD'})),
  hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:14}},xAxis:{textStyle:{fontSize:14}},yAxis:{min:0,max:32,majorUnit:8,textStyle:{fontSize:14}}});
applyPresentationChartFont(chart,{fontFamily:'Arial'});
const values=[['Outcome','Before','After'],['Both swapped statements',...scores.map(r=>r.changed.both_exact+'/'+r.changed.n)],
 ['Fact in fresh output',...scores.map(r=>r.fresh.contains+'/'+r.fresh.n)],
 ['Stopped on fresh pages',...scores.map(r=>r.fresh.stopped+'/'+r.fresh.n)]];
const t=slide.tables.add({rows:4,columns:3,left:552,top:125,width:512,height:212,columnWidths:[266,123,123],values});
t.styleOptions={headerRow:true,bandedRows:false};t.borders.assign({style:'solid',fill:'#D4DFD9',width:1});
for(let i=0;i<4;i++)for(let j=0;j<3;j++){const c=t.getCell(i,j);c.fill=i?'#FFFFFF':'#E9F1ED';c.text.style={typeface:'Arial',fontSize:16,bold:i===0,color:ink};}
label(slide,'A fact appearing in a reply can coexist with wrong grammar or a false claim.',552,368,500,86,18,false,ink);
slide.speakerNotes.textFrame.setText('The frozen interval reader may have seen the QA2D passage texts. The separate fresh set uses 96 official SQuAD development passage families.');
for(const d of ['draft','output'])await fs.mkdir(path.join(stage,d),{recursive:true});
const candidatePath=path.join(stage,'draft/plates.pptx'),finalPath=path.join(stage,'output/Visual_evidence_and_statements.pptx');
await(await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:4,
  materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[4],requiredNativeTableOwnerSlides:[4],verifyArtifactToolImport:true,
  integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','10363200,5124450','--require-native-table-slide','4'],
  receiptPath:path.join(stage,'presentation_validation.json')});
