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
const region=mode==='region';
const phrase=mode==='phrase';
const marginal=mode==='marginal';
const fact=mode==='fact';
const ranking=mode==='ranking';
const context=mode==='sentence-context';
const folder=path.join(root,'references/evidence',context?'sentence_context_v1':ranking?'fact_span_ranking_v1':fact?'fact_channel_v1':marginal?'phrase_marginal_v1':phrase?'phrase_region_v1':region?'region_writer_v1':views?'evidence_views_v1':branch?'branch_ink_v1':exam?'open_exam_v1':sets?'complete_fact_sets_v1':runs?'multiple_ink_runs_v1':competing?'competing_facts_v1':'recurrent_evidence_v1');
const plates=JSON.parse(await fs.readFile(path.join(folder,(context||ranking||fact||marginal||phrase||region||views||branch||exam||sets||runs||competing)?'plates.json':'reading-plates.json')));
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
if(fact){
  const data=JSON.parse(await fs.readFile(path.join(folder,'trajectory.json')));
  const tr=deck.slides.add();tr.background.fill='#FFFFFF';
  label(tr,'Writing during training',20,10,1040,43,29,true,ink);
  const arms=['direct','human','predicted'], names=['Full page','Fact alone','Fact + region'];
  const steps=Object.keys(data.direct).sort((a,b)=>+a-+b);
  for(const [j,t] of ['statements','exam'].entries()){
    for(const [i,metric] of ['bitmap_exact','fact'].entries()){
      const x=20+j*535,y=60+i*219;
      label(tr,(j?'Chinese':'English')+(i?' · fact present':' · exact image'),x,y,503,24,17,true,ink);
      const chart=tr.charts.add('line',{position:{left:x,top:y+30,width:504,height:171},categories:steps.map(s=>Number(s).toLocaleString('en-US')),
        series:arms.map((arm,k)=>({name:names[k],values:steps.map(s=>data[arm][s].totals[t][metric]),line:{fill:['#885640','#446796',teal][k],width:2.5}})),
        hasLegend:false,yAxis:{min:0,textStyle:{fontSize:11}},xAxis:{textStyle:{fontSize:11}}});
      applyPresentationChartFont(chart,{fontFamily:'Arial'});
    }
  }
  for(const [i,n] of names.entries())label(tr,n,210+i*231,504,225,26,16,true,['#885640','#446796',teal][i]);
  tr.speakerNotes.textFrame.setText('Counts out of 64 questions in each language at each saved writer checkpoint. The optical reader is held fixed. Category labels give the actual update counts.');
}
const slide=deck.slides.add();slide.background.fill='#FFFFFF';
if(fact){
  const data=JSON.parse(await fs.readFile(path.join(folder,'scores.json')));
  const arms=['direct','human','predicted'], names=['Full page','Fact alone','Fact + region'];
  label(slide,'From a pictured fact to an answer',25,18,1040,45,31,true,ink);
  label(slide,'Exact answer images (192 per language)',24,82,524,45,20,true,ink);
  const chart=slide.charts.add('bar',{barOptions:{direction:'column',grouping:'clustered'},position:{left:22,top:139,width:497,height:294},
    categories:['English','Chinese'],series:arms.map((a,i)=>({name:names[i],values:['statements','exam'].map(t=>data[a].original[t].bitmap_exact),fill:['#885640','#446796',teal][i]})),
    hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:13}},yAxis:{min:0,textStyle:{fontSize:13}},xAxis:{textStyle:{fontSize:14}}});
  applyPresentationChartFont(chart,{fontFamily:'Arial'});
  label(slide,'Both questions correct (96 passages)',550,82,510,45,20,true,ink);
  const values=[['Writer','English','Chinese'],...arms.map((a,i)=>[names[i],String(data[a].original.statements.both_exact),String(data[a].original.exam.both_exact)])];
  const table=slide.tables.add({rows:4,columns:3,left:550,top:143,width:490,height:205,columnWidths:[225,135,130],values});
  table.styleOptions={headerRow:true,bandedRows:false};table.borders.assign({style:'solid',fill:'#D4DFD9',width:1});
  for(let i=0;i<4;i++)for(let j=0;j<3;j++){const c=table.getCell(i,j);c.fill=i?'#FFFFFF':'#E9F1ED';c.text.style={typeface:'Arial',fontSize:17,bold:i===0,color:ink};}
  label(slide,'Only the pictured question and source page are supplied at inference.',25,457,1030,54,18,false,ink);
 }else if(context){
  const data=JSON.parse(await fs.readFile(path.join(folder,'scores.json')));
  const trace=JSON.parse(await fs.readFile(path.join(folder,'trajectory.json')));
  const modes=['original','direct','surround'], names=['Earlier','Direct','Context'], colors=['#88786A','#557DA1',teal];
  label(slide,'Writing complete answers from a pictured passage',22,15,1040,45,29,true,ink);
  label(slide,'a   New questions (192 per language)',22,73,520,30,19,true,ink);
  const bars=slide.charts.add('bar',{barOptions:{direction:'column',grouping:'clustered'},position:{left:22,top:111,width:488,height:246},
    categories:['English','Chinese'],series:modes.map((m,i)=>({name:names[i],values:['statements','exam'].map(t=>data[m][t].bitmap_exact),fill:colors[i]})),
    hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:14}},yAxis:{min:0,textStyle:{fontSize:13}},xAxis:{textStyle:{fontSize:13}}});
  applyPresentationChartFont(bars,{fontFamily:'Arial'});
  label(slide,'b   Development questions (48 per language)',530,73,545,30,19,true,ink);
  const steps=Object.keys(trace).sort((a,b)=>Number(a)-Number(b));
  const lines=slide.charts.add('bar',{barOptions:{direction:'column',grouping:'clustered'},position:{left:530,top:111,width:525,height:246},categories:steps,
    series:['statements','exam'].map((t,i)=>({name:['English','Chinese'][i],values:steps.map(k=>trace[k][t].bitmap_exact),fill:[colors[1],teal][i]})),
    hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:14}},yAxis:{min:0,textStyle:{fontSize:13}},xAxis:{textStyle:{fontSize:13}}});
  applyPresentationChartFont(lines,{fontFamily:'Arial'});
  const vals=[['Exact with both natural questions','Earlier','Direct','Context'],...['statements','exam'].map((t,i)=>[[ 'English (96 pages)','Chinese (96 pages)'][i],...modes.map(m=>String(data[m][t].both_exact))])];
  const tb=slide.tables.add({rows:3,columns:4,left:40,top:373,width:997,height:124,columnWidths:[400,190,190,217],values:vals});
  tb.styleOptions={headerRow:true,bandedRows:false};tb.borders.assign({style:'solid',fill:'#D4DFD9',width:1});
  for(let i=0;i<3;i++)for(let j=0;j<4;j++){const c=tb.getCell(i,j);c.fill=i?'#FFFFFF':'#E9F1ED';c.text.style={typeface:'Arial',fontSize:16,bold:i===0,color:ink};}
 }else if(ranking){
  const data=JSON.parse(await fs.readFile(path.join(folder,'scores.json'),'utf8'));
  const chosen=JSON.parse(await fs.readFile(path.join(folder,'reader.json'),'utf8'));
  const modes=['frozen','endpoints','whole'],groups=['statements','exam'];
  const labels=['Original','Endpoints','Whole span'],colors=['#83766C','#5879A3',teal];
  label(slide,'Which span answers the question?',22,18,1030,44,30,true,ink);
  label(slide,'Accepted fact and exact answer image (192 questions per language)',22,73,1040,36,18,true,ink);
  const chart=slide.charts.add('bar',{barOptions:{direction:'column',grouping:'clustered'},
    position:{left:18,top:124,width:596,height:295},categories:['English fact','English reply','Chinese fact','Chinese reply'],
    series:modes.map((m,i)=>({name:labels[i],fill:colors[i],values:groups.flatMap(g=>[chosen[m][g].selected,data[m][g].bitmap_exact])})),
    hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:14}},yAxis:{min:0,max:50,majorUnit:10,textStyle:{fontSize:13}},xAxis:{textStyle:{fontSize:13}}});
  applyPresentationChartFont(chart,{fontFamily:'Arial'});
  label(slide,'Both replies correct (96 pages)',632,129,432,40,20,true,ink);
  const values=[['Span score','English','Chinese'],...modes.map((m,i)=>[labels[i],...groups.map(g=>String(data[m][g].both_exact))])];
  const t=slide.tables.add({rows:4,columns:3,left:632,top:184,width:427,height:180,columnWidths:[193,117,117],values});
  t.styleOptions={headerRow:true,bandedRows:false};t.borders.assign({style:'solid',fill:'#D4DFD9',width:1});
  for(let i=0;i<4;i++)for(let j=0;j<3;j++){const c=t.getCell(i,j);c.fill=i?'#FFFFFF':'#E9F1ED';c.text.style={typeface:'Arial',fontSize:17,bold:i===0,color:ink};}
  label(slide,'All three readers use the same 32 candidate positions and the same sentence writer.',23,452,1030,51,18,false,ink);
}else if(marginal){
  const data=JSON.parse(await fs.readFile(path.join(folder,'trajectory.json')));
  const scores=JSON.parse(await fs.readFile(path.join(folder,'answers.json'))).new.scores;
  const steps=Object.keys(data).filter(k=>data[k]).sort((a,b)=>Number(a)-Number(b));
  label(slide,'Learning to write a complete answer',25,18,1040,43,31,true,ink);
  label(slide,'English development questions (192)',25,85,522,40,20,true,ink);
  const chart=slide.charts.add('bar',{barOptions:{direction:'column',grouping:'clustered'},position:{left:20,top:140,width:512,height:290},
    categories:steps.map(k=>Number(k).toLocaleString('en-US')),series:[['fact','Fact present',teal],['bitmap_exact','Exact answer','#AD6A48']].map(([key,name,fill])=>({name,fill,values:steps.map(s=>data[s].scores.statements[key])})),
    hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:13}},yAxis:{min:0,max:50,textStyle:{fontSize:14}},xAxis:{textStyle:{fontSize:13}}});
  applyPresentationChartFont(chart,{fontFamily:'Arial'});
  label(slide,'Final English confirmation (192 questions)',555,85,505,48,19,true,ink);
  const values=[['Training','Exact','Fact present'],...['single','longest','marginal'].map((a,i)=>[['Single glyph','Longest run','All paths'][i],String(scores[a].statements.bitmap_exact),String(scores[a].statements.fact)])];
  const table=slide.tables.add({rows:4,columns:3,left:555,top:143,width:486,height:204,columnWidths:[220,100,166],values});
  table.styleOptions={headerRow:true,bandedRows:false};table.borders.assign({style:'solid',fill:'#D4DFD9',width:1});
  for(let i=0;i<4;i++)for(let j=0;j<3;j++){const c=table.getCell(i,j);c.fill=i?'#FFFFFF':'#E9F1ED';c.text.style={typeface:'Arial',fontSize:17,bold:i===0,color:ink};}
  label(slide,'Columns show recorded updates. The planned 2,400-update checkpoint was unavailable.',25,453,1030,58,18,false,ink);
}else if(phrase){
  const results=JSON.parse(await fs.readFile(path.join(folder,'answers.json'))).confirmation.scores;
  label(slide,'Writing two answers from one pictured passage',25,20,1038,45,31,true,ink);
  label(slide,'English questions answered (192)',25,85,515,43,21,true,ink);
  const arms=['single','phrase','restricted'], names=['Single glyph','Variable run','Runs disabled'];
  const chart=slide.charts.add('bar',{barOptions:{direction:'column',grouping:'clustered'},position:{left:23,top:132,width:510,height:300},
    categories:['Exact answer','Fact present'],series:arms.map((a,i)=>({name:names[i],values:[results[a].statements.bitmap_exact,results[a].statements.fact],fill:['#897B72',teal,'#91ABA6'][i]})),
    hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:13}},yAxis:{min:0,max:80,textStyle:{fontSize:14}},xAxis:{textStyle:{fontSize:14}}});
  applyPresentationChartFont(chart,{fontFamily:'Arial'});
  label(slide,'Both requested facts present (96 passages)',544,85,528,52,19,true,ink);
  const values=[['Writer','English','Chinese'],...arms.map((a,i)=>[names[i],String(results[a].statements.both_fact),String(results[a].exam.both_fact)])];
  const table=slide.tables.add({rows:4,columns:3,left:548,top:148,width:498,height:208,columnWidths:[220,139,139],values});
  table.styleOptions={headerRow:true,bandedRows:false};table.borders.assign({style:'solid',fill:'#D4DFD9',width:1});
  for(let i=0;i<4;i++)for(let j=0;j<3;j++){const c=table.getCell(i,j);c.fill=i?'#FFFFFF':'#E9F1ED';c.text.style={typeface:'Arial',fontSize:17,bold:i===0,color:ink};}
  label(slide,'English targets are human statements. Chinese reading targets are short extractive answers.',25,464,1038,46,18,false,ink);
}else if(region){
  const results=JSON.parse(await fs.readFile(path.join(folder,'pixel-audit.json'))).measurements;
  label(slide,'Writing the requested fact',24,18,1040,48,31,true,ink);
  label(slide,'Replies containing the requested fact',25,84,540,40,21,true,ink);
  const arms=['direct','selected','oracle'],names=['Full page','Selected region','Human region'];
  const chart=slide.charts.add('bar',{barOptions:{direction:'column',grouping:'clustered'},position:{left:24,top:133,width:506,height:300},
    categories:['English statements','Chinese catalogue'],series:arms.map((a,i)=>({name:names[i],values:[results[a].statements.fact,results[a].catalogue.fact],fill:['#897B72',teal,'#91ABA6'][i]})),
    hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:13}},yAxis:{min:0,max:96,textStyle:{fontSize:14}},xAxis:{textStyle:{fontSize:14}}});
  applyPresentationChartFont(chart,{fontFamily:'Arial'});
  label(slide,'Exact stopped answer images',554,84,498,40,21,true,ink);
  const values=[['Input','English','Chinese'],...arms.map((a,i)=>[names[i],String(results[a].statements.bitmap_exact),String(results[a].catalogue.bitmap_exact)])];
  const table=slide.tables.add({rows:4,columns:3,left:554,top:143,width:490,height:204,columnWidths:[223,129,138],values});
  table.styleOptions={headerRow:true,bandedRows:false};table.borders.assign({style:'solid',fill:'#D4DFD9',width:1});
  for(let i=0;i<4;i++)for(let j=0;j<3;j++){let c=table.getCell(i,j);c.fill=i?'#FFFFFF':'#E9F1ED';c.text.style={typeface:'Arial',fontSize:17,bold:i===0,color:ink};}
  label(slide,'Each domain has 96 held families. A fact can appear in a sentence with an incorrect subject or predicate.',25,458,1038,64,17,false,ink);
  slide.speakerNotes.textFrame.setText('The chart counts stopped answers containing the requested short fact. The table requires the complete human reference bitmap. The two writing models were fitted on the same complete human targets.');
}else if(views){
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
const total=plates.length+1+Number(fact);
for(const d of ['draft','output'])await fs.mkdir(path.join(stage,d),{recursive:true});
const candidatePath=path.join(stage,'draft/plates.pptx'),finalPath=path.join(stage,'output',context?'Writing_the_sentence_in_context.pptx':ranking?'Choosing_the_answer_span.pptx':fact?'Reading_the_fact_then_writing.pptx':marginal?'Learning_to_write_complete_answers.pptx':phrase?'Writing_with_visible_phrases.pptx':region?'Writing_the_requested_fact.pptx':views?'Whole_stories_and_evidence.pptx':branch?'Distinguishing_the_requested_fact.pptx':exam?'Visual_question_conditioning.pptx':sets?'Binding_visual_answers.pptx':runs?'Visual_lists_from_pictured_pages.pptx':competing?'Competing_facts_and_quantized_readers.pptx':'Visual_evidence_and_statements.pptx');
await(await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:total,
  materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:fact?[total-1,total]:[total],requiredNativeTableOwnerSlides:[total],verifyArtifactToolImport:true,
  integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','10363200,5124450','--require-native-table-slide',String(total)],
  receiptPath:path.join(stage,'presentation_validation.json')});
