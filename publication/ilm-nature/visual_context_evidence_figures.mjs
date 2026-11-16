import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {Presentation,PresentationFile,FileBlob} from '@oai/artifact-tool';
const [root,stage,skill,python]=process.argv.slice(2);
const hash=b=>createHash('sha256').update(b).digest('hex');
const raw=await fs.readFile(path.join(root,'publication/ilm-nature/evidence/visual_context_completed_figure_data.json'));
const d=JSON.parse(raw);
for(const [n,h] of Object.entries(d.source_sha256))if(hash(await fs.readFile(path.join(root,n)))!==h)throw Error('Evidence changed: '+n);
const {applyPresentationChartFont,finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const deck=Presentation.create({slideSize:{width:1600,height:1700}});
const font='Arial',ink='#192936',muted='#43535C',blue='#386985',grey='#82919B',red='#A65B50',grid='#DDE4E8';
function text(s,value,x,y,w,h,size=27,bold=false){const q=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});q.text=value;q.text.style={typeface:font,fontSize:size,bold,color:bold?ink:muted,autoFit:'none'};return q;}
function table(s,values,x,y,w,h,widths,size=26){const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:h,columnWidths:widths,values});t.styleOptions={headerRow:true,bandedRows:false};t.borders.assign({style:'solid',fill:grid,width:1});for(let r=0;r<values.length;r++)for(let c=0;c<values[0].length;c++){const q=t.getCell(r,c);q.fill=r===0?'#EFF4F6':'#FFFFFF';q.text.style={typeface:font,fontSize:size,bold:r===0,color:r===0?ink:muted};}return t;}
async function picture(s,a,x,y,w,h){const b=await fs.readFile(path.join(root,a.file));if(hash(b)!==a.sha256)throw Error('Original PNG changed');s.images.add({blob:b,contentType:'image/png',fit:'contain',position:{left:x,top:y,width:w,height:h},alt:a.role??a.expected});}
const s=deck.slides.add();s.background.fill='#FFFFFF';
text(s,'Blank-target training misses its physical prerequisite',48,28,1504,70,42,true);
text(s,'12,597 supplied-history positions; autonomous language is unmeasured',50,105,1500,50,27);
text(s,'a',42,180,40,45,34,true);
const categories=['False ink at 252 white targets','False blanks at nonwhite targets'];
const series=[{name:'Control',values:[25,37],fill:grey},{name:'White-target model',values:[12,54],fill:blue},{name:'Fixed maximum',values:[11,37],fill:red}];
const chartTitle='Complete physical errors';
const axis={textStyle:{typeface:font,fontSize:26,fill:muted},line:{fill:muted,width:1}};
const chart=s.charts.add('bar',{name:chartTitle,title:chartTitle,position:{left:85,top:220,width:1460,height:410},
 categories,series,titleTextStyle:{typeface:font,fontSize:30,bold:true,fill:ink},barOptions:{direction:'column',grouping:'clustered',gapWidth:100},
 hasLegend:true,legend:{position:'bottom',overlay:false,textStyle:{typeface:font,fontSize:26,fill:muted}},
 xAxis:axis,yAxis:{...axis,min:0,max:65,majorUnit:10,numberFormatCode:'0',majorGridlines:{fill:grid,width:1}},dataLabels:{showValue:true,position:'outEnd',textStyle:{typeface:font,fontSize:26,fill:ink}}});
applyPresentationChartFont(chart,{fontFamily:font});
text(s,'b',42,680,40,45,34,true);text(s,'Every measured source group',85,680,1460,45,30,true);
const groupValues=[['Source group','Positions','False ink','False blanks: model / control'],...d.white.groups.map(r=>[r.label.replace(' qa ',' QA '),String(r.positions),String(r.false_ink),`${r.false_blanks} / ${r.control_false_blanks}`])];
table(s,groupValues,85,745,1460,365,[590,230,220,420]);
text(s,'All seven readability groups pass: 1,016/1,040 candidate forms credited.\nAll 17 additional false blanks occur in books. Exact bitmaps: 4,543 versus 4,609.',85,1135,1460,85,26);
text(s,'c',42,1250,40,45,34,true);text(s,'First qualifying example of each outcome, in original case order',85,1250,1460,45,30,true);
const sampleNames={cleared_white:'White target cleared',residual_false_ink:'Residual false ink',introduced_false_blank:'New false blank'};
for(let i=0;i<d.white.samples.length;i++){
 const a=d.white.samples[i],x=85+i*500;
 text(s,sampleNames[a.rule],x,1320,470,42,27,true);
 for(let j=0;j<3;j++){await picture(s,a.images[j],x+j*145,1380,96,96);text(s,['Source','Control','Model'][j],x+j*145-8,1490,130,32,22);}
 text(s,`${a.case}\nposition ${a.position}`,x,1540,470,60,22);
}
text(s,'Original tile pixels; no replacement reading. Both fixed blank conditions fail.\nFresh control review 1,010/1,040 does not replace its original 1,002/1,040.',50,1610,1500,80,26);
s.speakerNotes.textFrame.setText('Source: references/evidence/white_target_physical_result_20261116.json and publication/ilm-nature/evidence/visual_context_completed_figure_data.json. All124 cases and12597 positions are retained. The prospective limits are at most11false-ink outputs and37falseblanks. Readability credits are recognizable nonempty forms, not source-character identity or complete-language accuracy. All33boards and2080candidate/control images were reviewed before the frozen reference scoring, with18uncertain readings and no independent human adjudication. Fresh control readability is1010/1040; its original1002is unchanged. Examples follow first-in-original-qualification-order rules, not visual selection: first cleared white target, first residual false ink, first newly introduced false blank. Source/control/candidate32x32binary tiles are embedded from exact arrays; there is no reconstruction, font or corrected transcription. Complete autonomous language remains unmeasured because the physical gate fails.');
const q=deck.slides.add();q.background.fill='#FFFFFF';
text(q,'Source supervision and full-context execution',48,28,1504,65,42,true);
text(q,'Source fidelity, reader qualification and language learning are distinct claims',50,105,1500,50,27);
text(q,'a',42,185,40,45,34,true);text(q,'Completed source and reader checks',85,185,1460,45,30,true);
const checkValues=[['Check','Measured evidence','Decision'],
 ['Original source renderer','7,255,223 / 7,255,223 exact cells','Fidelity passes'],
 ['Pointwise Unicode labels','11 nonwhite aliases; 34 invisible cells','Not unique'],
 ['Fixed vision reader','203 / 273 observed reference masks fail','Rejected; incomplete'],
 ['Fixed image-encoder metric','6 / 9 correct answers fully accepted','Rejected']];
table(q,checkValues,85,250,1460,360,[485,650,325],26);
text(q,'Reader: 402/526 queries executed; 102 reference masks remain unobserved.\nMetric: 0/9 readable-wrong and 0/46 uncertain answers accepted; the correct-answer gate fails.',85,635,1460,85,26);
text(q,'b',42,755,40,45,34,true);text(q,'Preserved clean-reference reader failures',85,755,1460,45,30,true);
for(let i=0;i<2;i++){
 const a=d.reader_samples[i],x=130+i*750;await picture(q,a,x,820,145,145);
 text(q,`Expected ${a.expected}\nReader output: ${a.observed==='U+03C0'?'π (U+03C0)':a.observed}`,x+185,855,510,80,28);
}
text(q,'These are teacher inputs, not generated student answers. Original PNGs are unchanged.',85,980,1460,45,26);
text(q,'c',42,1050,40,45,34,true);text(q,'Complete-context learning: measured execution support',85,1050,1460,45,30,true);
const labels=['Visible glyphs\n32 × 32 tiles','Continuous\nimage reader','Causal context\n1.409B inherited','Image writer\nand pixels'];
for(let i=0;i<4;i++){
 const x=85+i*380;
 q.shapes.add({geometry:'rect',position:{left:x,top:1130,width:300,height:100},fill:'none',line:{fill:blue,width:2}});
 text(q,labels[i],x+12,1145,276,82,29,true);
 if(i<3)text(q,'→',x+315,1150,55,65,38,true);
}
text(q,'←  Each emitted span is decoded and reread as actual pixels  ←',270,1260,1220,45,27);
const executionValues=[['Completed reset check','Measured result'],['Complete training-record updates','6 / 6 reset exactly'],
 ['Changed context tensors per trial','309 treatment / 0 frozen control'],['Sampled PyTorch AdamW update error','0'],['Peak allocated GPU memory','11.252 GiB']];
table(q,executionValues,85,1335,1460,250,[950,510],26);
text(q,'No autonomous language score follows these reset updates.\nThe complete 13,785-sequence matched training comparison is pending.',50,1610,1500,80,26);
q.speakerNotes.textFrame.setText('Sources: the six completed public evidence JSON files bound in visual_context_completed_figure_data.json. The fullsource audit proves renderer fidelity, not semantic labels: eleven nonwhite physical alias groups and34zero-width U+200B occurrences reject exact pointwise Unicode recovery. The fixed localQwen3.5-9B Q8 reader has402/526saved responses;203of273observed distinct clean-reference masks fail, with102reference masks unobserved and124queries unexecuted. This one-sided necessary-condition failure rejects the fixed setup without completing calibration or generalizing to other vision models. The first two distinct failures in original case order are expected U+5B9E→A and U+7BA1→U+03C0. The source/preprocessing audit verifies402actual image encodings at81vision plus98text positions and does not establish a causal repair. The fixed image-encoder verifier accepts6/9knowncorrect,0/9readablewrong and0/46uncertain complete answers; unchanged allcorrect/separation gates fail. The final panel is an editable architecture diagram, not a trained output. Six real complete-record trials compare fresh FP32 frozen/trainablecontext arms with the same BF16forward, encoder, fullrankwriter and six objectives. Every one of309context tensors receives finite nonzero gradients and changes only in treatment; all actual fullstates reset byte exactly. Up to17sampled entries per active tensor match GPU PyTorchAdamW exactly; no independent fullgradient derivation is claimed.149.509seconds,0waiting,11.252GiBpeakallocation,42.713GiBminimumavailablehostRAM. CPU moments:11730771112bytes treatment. Whole deployedmodel1465381142parameters; sourcecontext1409410048and trainingonlyprojection8388608. Inherited textpretraining remains. No trained checkpoint retained, fromscratch visionlanguage proof, matchedquality efficiency or1000fold claim.');
await fs.mkdir(path.join(stage,'output'),{recursive:true});
await(await PresentationFile.exportPptx(deck)).save(path.join(stage,'candidate.pptx'));
const bindings={data_sha256:hash(raw),charts:[{title:chartTitle,categories,series:series.map(({name,values})=>({name,values}))}],tables:[groupValues,checkValues,executionValues],original_png_sha256:[...d.white.samples.flatMap(x=>x.images.map(a=>a.sha256)),...d.reader_samples.map(a=>a.sha256)],native_charts:1,native_tables:3,editable_diagram_nodes:4};
await fs.writeFile(path.join(stage,'figure_bindings.json'),JSON.stringify(bindings,null,2)+'\n');
const reference=path.join(root,'publication/ilm-nature/figures/Reward_calibration_figures.pptx');
await finalizePresentation({workspaceDir:stage,candidatePath:path.join(stage,'candidate.pptx'),finalPath:path.join(stage,'output/Visual_context_evidence_figures.pptx'),explicitTotalSlideCount:2,
 requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1,2],materializeLiteralChartWorkbooks:true,
 fontPolicy:{basis:'reference',families:[font],referencePath:reference,referenceSha256:hash(await fs.readFile(reference))},pythonExecutable:python,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','15240000,16192500','--validate-heading-fit','--require-native-table-slide','1','--require-native-table-slide','2'],
 verifyArtifactToolImport:true,receiptPath:path.join(stage,'presentation_validation.json')});
const final=await PresentationFile.importPptx(await FileBlob.load(path.join(stage,'output/Visual_context_evidence_figures.pptx')));
for(let i=0;i<2;i++){
 await fs.writeFile(path.join(stage,`figure-${30+i}.png`),new Uint8Array(await(await final.slides.items[i].export({format:'png',scale:1})).arrayBuffer()));
 await fs.writeFile(path.join(stage,`figure-${30+i}.layout.json`),await(await final.slides.items[i].export({format:'layout'})).text());
}
