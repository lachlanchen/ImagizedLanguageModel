import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation,PresentationFile} from '@oai/artifact-tool';
const [root,stage,skill,python,mode]=process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')));
const competing=mode==='competing';
const runs=mode==='runs';
const sets=mode==='sets';
const exam=mode==='exam';
const branch=mode==='branch';
const views=mode==='views';
const folder=path.join(root,'references/evidence',views?'evidence_views_v1':branch?'branch_ink_v1':exam?'open_exam_v1':sets?'complete_fact_sets_v1':runs?'multiple_ink_runs_v1':competing?'competing_facts_v1':'recurrent_evidence_v1');
const plates=JSON.parse(await fs.readFile(path.join(folder,(views||branch||exam||sets||runs||competing)?'plates.json':'reading-plates.json')));
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
if(views){
  const arms=['parent','whole','highlight'];
  const scores=await Promise.all(arms.map(async a=>JSON.parse(await fs.readFile(path.join(folder,a+'-scores.json')))));
  const labels=['Before fitting','Full-story control','Evidence-trained'];
  label(slide,'Writing answers from a full story',24,19,1038,52,32,true,ink);
  label(slide,'Correct pairs of complete answers',24,89,509,48,21,true,ink);
  const chart=slide.charts.add('bar',{barOptions:{direction:'column',grouping:'clustered'},position:{left:22,top:144,width:486,height:298},
    categories:['CoQA ('+scores[0].coqa.all.pairs+')','DROP ('+scores[0].drop.all.pairs+')'],
    series:scores.map((s,i)=>({name:labels[i],values:[s.coqa.all.both,s.drop.all.both],fill:['#8C9F99','#AA664A',teal][i]})),
    hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:12}},yAxis:{min:0,textStyle:{fontSize:14}},xAxis:{textStyle:{fontSize:15}}});
  applyPresentationChartFont(chart,{fontFamily:'Arial'});
  label(slide,'English first turns with different inputs',524,89,541,49,21,true,ink);
  const values=[['Page supplied','Before','Full-story','Evidence'],...[['full','First 600 cells'],['evidence','Human evidence'],['disjoint','Disjoint span'],['masked','Evidence erased']].map(([k,t])=>[t,...scores.map(s=>String(s[k].all.exact))])];
  const t=slide.tables.add({rows:5,columns:4,left:524,top:147,width:535,height:237,columnWidths:[176,105,124,130],values});
  t.styleOptions={headerRow:true,bandedRows:false};t.borders.assign({style:'solid',fill:'#D4DFD9',width:1});
  for(let i=0;i<5;i++)for(let j=0;j<4;j++){const c=t.getCell(i,j);c.fill=i?'#FFFFFF':'#E9F1ED';c.text.style={typeface:'Arial',fontSize:16,bold:i===0,color:ink};}
  label(slide,'The table gives accepted stopped answers out of '+scores[0].full.all.n+' turns. Only training and the indicated diagnostic supply annotated evidence.',25,467,1035,63,17,false,ink);
  slide.speakerNotes.textFrame.setText('Paired whole-page free generation and source interventions are scored with the original human reference sets.');
}else if(branch){
  const load=async name=>JSON.parse(await fs.readFile(path.join(folder,name+'.json')));
  const [fresh,prior,transfer]=await Promise.all(['fresh','previous','transfer'].map(load));
  const arms=['matched','branch'], names=['Whole answer','Added glyph loss'];
  label(slide,'Does the writer distinguish two requests?',24,18,1040,46,31,true,ink);
  label(slide,'Correct individual answers',25,88,550,45,22,true,ink);
  const chart=slide.charts.add('bar',{barOptions:{direction:'column',grouping:'clustered'},
    position:{left:22,top:141,width:493,height:296},categories:['Fresh (288)','Earlier (384)'],
    series:arms.map((a,i)=>({name:names[i],values:[fresh,prior].map(d=>d[a].intact.free),fill:['#94A7A2',teal][i]})),
    hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:14}},yAxis:{min:0,max:25,majorUnit:5},xAxis:{textStyle:{fontSize:16}}});
  applyPresentationChartFont(chart,{fontFamily:'Arial'});
  label(slide,'Both answers independently correct',523,88,540,45,22,true,ink);
  const values=[['Questions','Control','New writer'],['Fresh Chinese',...arms.map(a=>fresh[a].intact.both_free+' / 80')],
    ['Earlier Chinese',...arms.map(a=>prior[a].intact.both_free+' / 128')],['Bilingual',...arms.map(a=>transfer[a].both+' / 256')],
    ['Edited numbers',...arms.map(a=>transfer[a].changed_both+' / 24')]];
  const t=slide.tables.add({rows:5,columns:3,left:522,top:148,width:538,height:240,columnWidths:[225,149,164],values});
  t.styleOptions={headerRow:true,bandedRows:false};t.borders.assign({style:'solid',fill:'#D4DFD9',width:1});
  for(let i=0;i<5;i++)for(let j=0;j<3;j++){const c=t.getCell(i,j);c.fill=i?'#FFFFFF':'#E9F1ED';c.text.style={typeface:'Arial',fontSize:17,bold:i===0,color:ink};}
  label(slide,'Each exact answer must end after its final glyph. Edited numbers require the original and changed answer.',25,471,1038,49,17,false,ink);
  slide.speakerNotes.textFrame.setText('Counts from the final checkpoints at equal data exposure. The fresh Chinese paired pages supply the primary endpoint.');
}else if(exam){
  const data=JSON.parse(await fs.readFile(path.join(folder,'results.json')));
  const arms=['parent','exchanged','matched'];
  label(slide,'Writing answers to pictured Chinese questions',24,20,1040,46,30,true,ink);
  label(slide,'Exact free replies on 384 questions',24,85,520,42,20,true,ink);
  const chart=slide.charts.add('bar',{barOptions:{direction:'column',grouping:'clustered'},
    position:{left:24,top:146,width:511,height:280},categories:['Original','Other question','Other page'],
    series:arms.map((a,i)=>({name:['Before fitting','Exchanged fit','Matched fit'][i],
      values:['intact','other_question','other_page'].map(k=>data[a][k].free),fill:['#94A7A2','#B46C49',teal][i]})),
    hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:13}},yAxis:{min:0,max:25,majorUnit:5,textStyle:{fontSize:14}},xAxis:{textStyle:{fontSize:14}}});
  applyPresentationChartFont(chart,{fontFamily:'Arial'});
  label(slide,'Two distinct answers on the same page',554,85,520,44,20,true,ink);
  const vals=[['Writer','Both written','Both ranked'],...arms.map((a,i)=>[['Before fitting','Exchanged fit','Matched fit'][i],
    data[a].intact.both_free+' / 128',data[a].intact.both_ranked+' / 128'])];
  const t=slide.tables.add({rows:4,columns:3,left:552,top:147,width:508,height:210,columnWidths:[176,166,166],values:vals});
  t.styleOptions={headerRow:true,bandedRows:false};t.borders.assign({style:'solid',fill:'#D4DFD9',width:1});
  for(let i=0;i<4;i++)for(let j=0;j<3;j++){const c=t.getCell(i,j);c.fill=i?'#FFFFFF':'#E9F1ED';c.text.style={typeface:'Arial',fontSize:17,bold:i===0,color:ink};}
  label(slide,'Ranking supplies the possible answers. Free generation supplies only the rendered glyph grids of the passage and question.',25,470,1035,61,17,false,ink);
  slide.speakerNotes.textFrame.setText('The inputs are digitally rendered aligned glyph grids. The original exam has one labelled option. Free writing requires a stopped match to that wording.');
}else if(sets){
  const data=JSON.parse(await fs.readFile(path.join(root,'references/evidence/fact_binding_v1/scores.json')));
  const arms=['parent','linear','mixture'], names=['Frozen reader','Additive head','Two readings'];
  label(slide,'Pairing a supplied list and writing an answer',23,20,1040,44,30,true,ink);
  label(slide,'Both supplied lists preferred (64 pages)',23,88,495,55,21,true,ink);
  const chart=slide.charts.add('bar',{barOptions:{direction:'column',grouping:'clustered'},
    position:{left:22,top:149,width:472,height:305},categories:['Original question','Unrelated question'],
    series:arms.map((arm,i)=>({name:names[i],values:['witness','witness-exchanged'].map(k=>data[arm][k].witness.both),fill:['#94A7A2','#B46C49',teal][i]})),
    hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:13}},yAxis:{min:0,max:24,majorUnit:4,textStyle:{fontSize:14}},xAxis:{textStyle:{fontSize:14}}});
  applyPresentationChartFont(chart,{fontFamily:'Arial'});
  label(slide,'Whole answers written from the page',520,88,540,55,21,true,ink);
  const groups=[['New Chinese (64)','witness','witness'],['English plural (64)','confirmation','en-multi'],['Chinese plural (96)','confirmation','zh-multi'],['Chinese single (64)','confirmation','zh-single']];
  const values=[['Pages','Frozen','Additive','Two readings'],...groups.map(([name,k,g])=>[name,...arms.map(a=>String(data[a][k][g].exact))])];
  const table=slide.tables.add({rows:5,columns:4,left:520,top:149,width:548,height:250,columnWidths:[185,92,108,163],values});
  table.styleOptions={headerRow:true,bandedRows:false};
  table.borders.assign({style:'solid',fill:'#D4DFD9',width:1});
  for(let i=0;i<5;i++)for(let j=0;j<4;j++){const cell=table.getCell(i,j);cell.fill=i?'#FFFFFF':'#E9F1ED';cell.text.style={typeface:'Arial',fontSize:16,bold:i===0,color:ink};}
  label(slide,'At inference only the question and page images are available. Every missing or additional part counts against the free answer.',23,473,1040,55,17,false,ink);
  slide.speakerNotes.textFrame.setText('The score preference tests two annotated disjoint placements and two exchanges. Free answers are scored against any accepted annotation.');
}else if(runs){
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
const candidatePath=path.join(stage,'draft/plates.pptx'),finalPath=path.join(stage,'output',views?'Whole_stories_and_evidence.pptx':branch?'Distinguishing_the_requested_fact.pptx':exam?'Visual_question_conditioning.pptx':sets?'Binding_visual_answers.pptx':runs?'Visual_lists_from_pictured_pages.pptx':competing?'Competing_facts_and_quantized_readers.pptx':'Visual_evidence_and_statements.pptx');
await(await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:total,
  materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[total],requiredNativeTableOwnerSlides:[total],verifyArtifactToolImport:true,
  integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','10363200,5124450','--require-native-table-slide',String(total)],
  receiptPath:path.join(stage,'presentation_validation.json')});
