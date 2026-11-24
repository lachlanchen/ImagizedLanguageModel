import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation,PresentationFile} from '@oai/artifact-tool';
const [root,stage,skill,python,mode]=process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')));
const competing=mode==='competing';
const runs=mode==='runs';
const folder=path.join(root,'references/evidence',runs?'multiple_ink_runs_v1':competing?'competing_facts_v1':'recurrent_evidence_v1');
const plates=JSON.parse(await fs.readFile(path.join(folder,(runs||competing)?'plates.json':'reading-plates.json')));
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
if(runs){
  const results=await Promise.all(['scores','survival-scores','confirm-scores','survival-confirm-scores'].map(async n=>(JSON.parse(await fs.readFile(path.join(folder,n+'.json')))).scores.count.original));
  const groups=[['en-multi',0],['en-multi',2],['zh-multi',0],['zh-multi',2]];
  const comp=JSON.parse(await fs.readFile(path.join(folder,'composition.json')));
  label(slide,'Recovering a complete answer from several facts',24,20,1040,45,30,true,ink);
  label(slide,'Every requested fact, with no extra part',24,87,540,32,21,true,ink);
  const chart=slide.charts.add('bar',{barOptions:{direction:'column',grouping:'clustered'},
    position:{left:24,top:138,width:548,height:292},categories:['English (160)','English (96)','Chinese (160)','Chinese (96)'],
    series:['Cell classifier','Conditional'].map((name,i)=>({name,values:groups.map(([g,j])=>results[j+i][g].exact),fill:i?teal:'#9EAFAA'})),
    hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:14}},yAxis:{min:0,max:10,majorUnit:2,textStyle:{fontSize:14}},xAxis:{textStyle:{fontSize:12}}});
  applyPresentationChartFont(chart,{fontFamily:'Arial'});
  label(slide,'What survives in an English sentence?',600,87,475,32,21,true,ink);
  const data=[['Facts supplied','Stopped','All present'],...['read','given'].map((k,i)=>[i?'Human facts':"Reader’s facts",comp[k].stopped+'/'+comp[k].n,comp[k].contains_all+'/'+comp[k].n])];
  const tab=slide.tables.add({rows:3,columns:3,left:600,top:145,width:458,height:160,columnWidths:[179,132,147],values:data});
  tab.styleOptions={headerRow:true,bandedRows:false};tab.borders.assign({style:'solid',fill:'#D4DFD9',width:1});
  for(let i=0;i<3;i++)for(let j=0;j<3;j++){const c=tab.getCell(i,j);c.fill=i?'#FFFFFF':'#E9F1ED';c.text.style={typeface:'Arial',fontSize:16,bold:i===0,color:ink};}
  label(slide,'The sentence measure credits repeated wording if every annotated part is present and generation stops.',600,340,449,99,18,false,ink);
  label(slide,'Initial and fresh pages are scored separately. Each English sentence condition uses the same 256 questions.',24,477,1040,40,16,false,ink);
  slide.speakerNotes.textFrame.setText('Held official development pages. Membership is scored after normalization, with no credit for missing or additional parts. No human judgment of the new sentences.');
}else if(competing){
  const data=JSON.parse(await fs.readFile(path.join(folder,'benchmark.json')));
  label(slide,'Answering two questions on the same page',25,22,1035,48,31,true,ink);
  label(slide,'All 256 pages',25,89,470,32,21,true,ink);
  label(slide,'Same 64 pages, exact answers',551,89,510,35,21,true,ink);
  const arms=['parent','contrast','qwen8-text'];
  const chart=slide.charts.add('bar',{barOptions:{direction:'column',grouping:'clustered'},
    position:{left:21,top:147,width:488,height:293},categories:['Original ILM','Contrast ILM','Qwen3 text'],
    series:[{name:'Both correct',values:arms.map(a=>data.full[a].both),fill:teal}],
    hasLegend:false,yAxis:{min:0,max:128,majorUnit:32,textStyle:{fontSize:14}},xAxis:{textStyle:{fontSize:14}}});
  applyPresentationChartFont(chart,{fontFamily:'Arial'});
  const labels=['Original ILM (images)','Contrast ILM (images)','Qwen3 (text)','Qwen3-VL (images)','Qwen3-VL (text)'];
  const rows=[['System','Questions','Both'],...Object.values(data.subset).map((v,i)=>[labels[i],v.correct+'/'+v.questions,v.both+'/'+v.pages])];
  const t=slide.tables.add({rows:rows.length,columns:3,left:551,top:145,width:515,height:290,columnWidths:[251,138,126],values:rows});
  t.styleOptions={headerRow:true,bandedRows:false};t.borders.assign({style:'solid',fill:'#D4DFD9',width:1});
  for(let i=0;i<rows.length;i++)for(let j=0;j<3;j++){const c=t.getCell(i,j);c.fill=i?'#FFFFFF':'#E9F1ED';c.text.style={typeface:'Arial',fontSize:17,bold:i===0,color:ink};}
  label(slide,'Qwen models use 8B weights at Q4_K_M. Both visual systems receive identical ink.',25,475,1040,34,17,false,ink);
  slide.speakerNotes.textFrame.setText('Short-answer exact match with a stop; two distinct questions per held page, four public corpora. Inputs are aligned glyph grids. Pretraining contamination in the general models is unknown.');
}else{
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
}
const total=plates.length+1;
for(const d of ['draft','output'])await fs.mkdir(path.join(stage,d),{recursive:true});
const candidatePath=path.join(stage,'draft/plates.pptx'),finalPath=path.join(stage,'output',runs?'Visual_lists_from_pictured_pages.pptx':competing?'Competing_facts_and_quantized_readers.pptx':'Visual_evidence_and_statements.pptx');
await(await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:total,
  materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[total],requiredNativeTableOwnerSlides:[total],verifyArtifactToolImport:true,
  integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','10363200,5124450','--require-native-table-slide',String(total)],
  receiptPath:path.join(stage,'presentation_validation.json')});
