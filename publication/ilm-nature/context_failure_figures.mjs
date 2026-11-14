import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root,stage,skill,python]=process.argv.slice(2);
if(![root,stage,skill,python].every(x=>path.isAbsolute(x??'')))throw Error('Absolute paths required');
const {applyPresentationChartFont,finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const hash=raw=>createHash('sha256').update(raw).digest('hex');
const raw=await fs.readFile(path.join(root,'publication/ilm-nature/evidence/context_failure_figure_data.json')),d=JSON.parse(raw);
if(d.physical_gate_passed||d.complete_language!=='unmeasured'||d.new_model_calls!==0||d.samples.length!==6)throw Error('Completed measured failure required');
for(const [name,digest]of Object.entries(d.source_sha256))if(hash(await fs.readFile(path.join(root,name)))!==digest)throw Error('Source changed: '+name);
await fs.mkdir(path.join(stage,'output'),{recursive:true});
const deck=Presentation.create({slideSize:{width:1600,height:1800}}),s=deck.slides.add();s.background.fill='#FFFFFF';
const font='Arial',ink='#192936',muted='#43535C',blue='#386985',brown='#9B7457',red='#A74337',grid='#DDE4E8';
function text(value,x,y,w,h,size=26,bold=false){const q=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});q.text=value;q.text.style={typeface:font,fontSize:size,bold,color:bold?ink:muted,autoFit:'none'};}
const axis={textStyle:{typeface:font,fontSize:23,fill:muted},line:{fill:muted,width:1}},charts=[];
function bar(title,x,y,w,h,categories,series,max,major,format){
 const config={name:title,title,position:{left:x,top:y,width:w,height:h},categories,series,
 titleTextStyle:{typeface:font,fontSize:29,bold:true,fill:ink},
 barOptions:{direction:'column',grouping:'clustered',gapWidth:90},hasLegend:true,
 legend:{position:'bottom',overlay:false,textStyle:{typeface:font,fontSize:23,fill:muted}},
 xAxis:axis,yAxis:{...axis,min:0,max,majorUnit:major,numberFormatCode:format,majorGridlines:{fill:grid,width:1}},
 dataLabels:{showValue:true,position:'outEnd',textStyle:{typeface:font,fontSize:24,fill:ink}}};
 const chart=s.charts.add('bar',config);applyPresentationChartFont(chart,{fontFamily:font});
 charts.push({title,categories,series:series.map(({name,values})=>({name,values})),max,major,format});
}
text('Lower image loss coexists with more blank output',48,28,1504,65,43,true);
text('a',42,122,40,45,34,true);text('b',827,122,40,45,34,true);
const labels=d.book_groups.map(g=>g.label);
bar('Source-image NLL per physical mask',85,125,700,455,labels,[
 {name:'Head only',values:d.book_groups.map(g=>Number(g.control_nll.toFixed(3))),fill:brown},
 {name:'Learned context',values:d.book_groups.map(g=>Number(g.context_nll.toFixed(3))),fill:blue}],.13,.03,'0.000');
bar('False blank outputs',865,125,680,455,labels,[
 {name:'Head only',values:d.book_groups.map(g=>g.control_false_blank),fill:brown},
 {name:'Learned context',values:d.book_groups.map(g=>g.context_false_blank),fill:blue}],360,100,'0');
text('Complete supplied-history positions: 1,161 development; 4,586 in each training layout. True whitespace excluded from false blanks.',85,610,1460,70,25);
const tableValues=[['Group','Dev books fixed','Train books fixed','Train books proportional'],
 ['Source whitespace','22','110','110'],['False blanks, head / context','42 / 93','137 / 307','145 / 289']];
const table=s.tables.add({rows:3,columns:4,left:85,top:690,width:1460,height:140,columnWidths:[390,340,340,390],values:tableValues});
table.styleOptions={headerRow:true,bandedRows:false};table.borders.assign({style:'solid',fill:grid,width:1});
for(let r=0;r<3;r++)for(let c=0;c<4;c++){const cell=table.getCell(r,c);cell.fill=r===0?'#EFF4F6':'#FFFFFF';cell.text.style={typeface:font,fontSize:23,bold:r===0,color:r===0?ink:muted};}
text('c',42,878,40,45,34,true);
bar('Source-preferred false blanks',85,880,760,365,labels,[
 {name:'Observed',values:d.book_groups.map(g=>g.source_preferred),fill:blue},
 {name:'Required',values:d.book_groups.map(g=>g.required),fill:red}],180,50,'0');
text('108 certified empty modes',925,895,620,50,32,true);
text('Of 711 false blanks, only 57 prefer the source.\nAll three prospective search criteria fail.\n\nThe 108-mode count includes 102 book positions.\nAll saved physical argmax choices reproduce.\n\nBounded numerical test; finite precision limits\nits interpretation. No new glyphs generated.',925,965,620,270,25);
text('d',42,1308,40,45,34,true);
text('First six certified empty-mode development-book failures',85,1308,1460,50,30,true);
text('Each triplet: source, head-only control, learned context. Original case and zero-based position order.',85,1360,1460,45,25);
for(let i=0;i<d.samples.length;i++){
 const item=d.samples[i],x=90+i*239,y=1418;
 text(item.case.split('__')[1].replace('_normal','').replace('book_','B')+':'+item.position,x,y,224,38,24);
 for(let j=0;j<3;j++){
  const role=['source','head','context'][j],a=item.images[role],bytes=await fs.readFile(path.join(root,a.file));
  if(hash(bytes)!==a.sha256)throw Error('Original sample changed');
  s.shapes.add({geometry:'rect',position:{left:x+j*73-1,top:y+43,width:71,height:71},fill:'none',line:{fill:grid,width:1}});
  s.images.add({blob:bytes,contentType:'image/png',fit:'contain',position:{left:x+j*73,top:y+44,width:69,height:69},alt:`${item.case} position${item.position} ${role}; original32x32 expanded3x nearest`});
 }
}
text('All 21 eligible development positions remain in the probability record; six shown by the fixed ordering.',85,1575,1460,45,25);
text('28 review blank-flag errors include four candidate and two control credited empty images; frozen grades retained.',50,1650,1500,65,25);
text('Complete language remains unmeasured after failed qualification. The 7.553B deployment retains inherited weights.',50,1730,1500,55,25);
s.speakerNotes.textFrame.setText('Sources: publication/ilm-nature/evidence/context_failure_figure_data.json and its hash-bound completed physical/probability exports. '
 +'The context-gradient actor completed13766sources/41298updates/6108741targetpositions. All124 supplied-history cases use the actual learned131-tensor continuation state. '
 +'Panels a,b retain all three book groups; NLL plot/workbook values round to three decimals; full precision remains in the source JSON. False blanks exclude truly white source images. Counts are exact. '
 +'Panel c uses source-over-blank margin0.001nat and the prospective47/93,154/307,145/289 required counts. Observed4,26,27 all fail. Bound: every nonempty raster has a first nonzero physical mask, and all later log probabilities are <=0.108falseblankstates,102bookstates, satisfy its numerical strict blank preference; this is not a formal floating-point proof. '
 +'Panel d shows the first6 of21 certified development-book empty modes in originalcase/position order. Threefold nearest-neighbor source expansions preserve all original binary pixels. Every source contains ink; every context output is empty. No inferred or reference-repaired reading enters this figure. '
 +'Original frozen model review used prior cohort familiarity and no independent human adjudication.28blankflags contradictpixels, including4candidate+2controlcreditedempty; scores remain unchanged and are not corrected physical accuracy. '
 +'The new whole-glyph ranking objective passed a reset real-source preflight, but its full trial is separately registered/running and unmeasured. No new model call or optimizer update was used to make this figure. Whole-model compactness, useful language, historical transfer and matched-quality efficiency remain unproved.');
const candidate=path.join(stage,'candidate.pptx');await(await PresentationFile.exportPptx(deck)).save(candidate);
await fs.writeFile(path.join(stage,'figure_bindings.json'),JSON.stringify({measured_data_sha256:hash(raw),source_sha256:d.source_sha256,charts,tableValues,native_charts:3,native_tables:1,embedded_pixel_images:18,plotted_nll_decimal_places:3,original_readings_unchanged:true,new_model_calls:0},null,2)+'\n');
await finalizePresentation({workspaceDir:stage,candidatePath:candidate,finalPath:path.join(stage,'output/Context_likelihood_failure_figures.pptx'),explicitTotalSlideCount:1,
 requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],materializeLiteralChartWorkbooks:true,
 fontPolicy:{basis:'reference',families:[font],referencePath:path.join(root,'publication/ilm-nature/figures/Joint_writer_physical_figures.pptx'),referenceSha256:hash(await fs.readFile(path.join(root,'publication/ilm-nature/figures/Joint_writer_physical_figures.pptx')))},
 pythonExecutable:python,integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','15240000,17145000','--validate-heading-fit','--require-native-table-slide','1'],
 verifyArtifactToolImport:true,receiptPath:path.join(stage,'presentation_validation.json')});
const final=await PresentationFile.importPptx(await FileBlob.load(path.join(stage,'output/Context_likelihood_failure_figures.pptx')));
const slide=final.slides.items[0];await fs.writeFile(path.join(stage,'final-reimport.png'),new Uint8Array(await(await slide.export({format:'png',scale:1})).arrayBuffer()));
await fs.writeFile(path.join(stage,'final-reimport.layout.json'),await(await slide.export({format:'layout'})).text());
