import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root,stage,skill,python]=process.argv.slice(2);
if (![root,stage,skill,python].every(x=>path.isAbsolute(x??''))) throw Error('Absolute paths required');
const {applyPresentationChartFont,finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const hash=raw=>createHash('sha256').update(raw).digest('hex');
const source=path.join(root,'publication/ilm-nature/evidence/joint_writer_figure_data.json');
const raw=await fs.readFile(source),d=JSON.parse(raw);
if(d.physical_gate_passed||d.complete_language!=='unmeasured'||d.new_model_calls!==0||d.blank_book_samples.length!==17)throw Error('Completed measured trial required');
for(const [name,digest] of Object.entries(d.source_sha256))if(hash(await fs.readFile(path.join(root,name)))!==digest)throw Error('Source changed: '+name);
await fs.mkdir(path.join(stage,'output'),{recursive:true});
const deck=Presentation.create({slideSize:{width:1600,height:1800}}),s=deck.slides.add();s.background.fill='#FFFFFF';
const font='Arial',ink='#192936',muted='#43535C',blue='#386985',brown='#9B7457',red='#A74337',grid='#DDE4E8';
function text(value,x,y,w,h,size=26,bold=false){const q=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});q.text=value;q.text.style={typeface:font,fontSize:size,bold,color:bold?ink:muted,autoFit:'none'};}
const axis={textStyle:{typeface:font,fontSize:23,fill:muted},line:{fill:muted,width:1}};
const charts=[];
function bar(title,x,y,w,h,categories,series,max,major,format){
 const config={position:{left:x,top:y,width:w,height:h},title,titlePlacement:'aboveChart',titleTextStyle:{typeface:font,fontSize:29,bold:true,fill:ink},
 categories,series:series.map(v=>({...v,valuesFormatCode:format})),barOptions:{direction:'column',grouping:'clustered',gapWidth:90},
 hasLegend:true,legend:{position:'bottom',overlay:false,textStyle:{typeface:font,fontSize:24,fill:muted}},xAxis:axis,
 yAxis:{...axis,min:0,max,majorUnit:major,numberFormatCode:format,majorGridlines:{fill:grid,width:1}},
 dataLabels:{showValue:true,position:'outEnd',textStyle:{typeface:font,fontSize:23,fill:ink}}};
 const c=s.charts.add('bar',config);applyPresentationChartFont(c,{fontFamily:font});charts.push({title,categories,series:config.series.map(({name,values})=>({name,values})),max,major,format});
}
text('Joint image likelihood and the remaining blank outputs',48,28,1504,70,43,true);
text('a',42,123,40,45,34,true);
bar('Frozen review credits with supplied image history',80,125,1470,505,d.groups.map(g=>g.label),[
 {name:'Point control',values:d.groups.map(g=>Number(g.control_fraction.toFixed(6))),fill:brown},
 {name:'Joint writer',values:d.groups.map(g=>Number(g.candidate_fraction.toFixed(6))),fill:blue,
  dataLabelOverrides:d.groups.map((_,idx)=>({idx,position:'center',showValue:true,textStyle:{typeface:font,fontSize:23,fill:'#FFFFFF'}}))},
 {name:'Required in every group',values:d.groups.map(()=>.9),fill:red}],1.08,.25,'0.0%');
const tableValues=[['Count',...d.groups.map(g=>g.label.replaceAll('\n',' '))],['Joint',...d.groups.map(g=>`${g.candidate}/${g.count}`)],['Point',...d.groups.map(g=>`${g.control}/${g.count}`)]];
const table=s.tables.add({rows:3,columns:8,left:85,top:650,width:1460,height:140,columnWidths:[95,195,195,195,195,195,195,195],values:tableValues});
table.styleOptions={headerRow:true,bandedRows:false};table.borders.assign({style:'solid',fill:grid,width:1});
for(let r=0;r<3;r++)for(let c=0;c<8;c++){const cell=table.getCell(r,c);cell.fill=r===0?'#EFF4F6':'#FFFFFF';cell.text.style={typeface:font,fontSize:22,bold:r===0,color:r===0?ink:muted};}
text('Original credits retained; pixel audit finds 17 blank-flag errors, including four credited empty images.',85,805,1460,48,26);

text('b',42,880,40,45,34,true);text('All 17 uncredited selected development-book outputs',85,878,1460,48,30,true);
text('Each triplet: source image, point control, joint writer. Case:position uses the original zero-based position.',85,932,1460,45,24);
for(let i=0;i<d.blank_book_samples.length;i++){
 const item=d.blank_book_samples[i],x=92+(i%9)*163,y=995+Math.floor(i/9)*140;
 text(item.case.split('__')[1].replace('_normal','').replace('book_','B')+':'+item.position,x,y,148,36,22);
 for(let j=0;j<3;j++){
  const role=['source','point','joint'][j],a=item.images[role],bytes=await fs.readFile(path.join(root,a.file));if(hash(bytes)!==a.sha256)throw Error('Original pixels changed');
  s.shapes.add({geometry:'rect',position:{left:x+j*49-1,top:y+41,width:46,height:46},fill:'none',line:{fill:grid,width:1}});
  s.images.add({blob:bytes,contentType:'image/png',fit:'contain',position:{left:x+j*49,top:y+42,width:44,height:44},alt:`${item.case}, position${item.position}, ${role}; original32x32 expanded3x nearest`});
 }
}
text('15 of these outputs are empty; two other empty book outputs were credited. Frozen grades are retained.',85,1270,1460,45,25);

text('c',42,1340,40,45,34,true);
bar('Source-preferred selected false blanks',85,1340,760,330,d.selected_search_categories,[
 {name:'Observed',values:d.selected_source_preferred,fill:blue},
 {name:'Required',values:d.selected_search_required,fill:red}],8,2,'0');
text('Search-only criterion fails',910,1348,620,50,30,true);
text('Source preferred: 1/14 selected book false blanks;\n0/4 selected training QA false blanks.\n\nAcross all positions: 16/349 false blanks prefer\nthe source over the empty image. The prefix bound\nsupports an empty mode for 15/42 book false blanks.\n\nNumerical bound; no formal floating-point certificate.',910,1410,630,250,24);
text('124 complete cases · 33 opaque boards · 2,080 frozen readings · 41,298 retained head updates',50,1700,1500,43,25);
text('Supplied-history readability is not complete language. Autonomous quality remains unmeasured.',50,1745,1500,43,25);
s.speakerNotes.textFrame.setText('Sources: publication/ilm-nature/evidence/joint_writer_figure_data.json and its SHA-bound completed physical and probability exports. '
 +'All seven groups retain their original selected positions, whitespace and90%rule. Review used one model reviewer with prior cohort familiarity; no independent human adjudication. '
 +'Panel b includes every uncredited selected development-book output, in case/position order. Each96x96PNG is the original32x32binary image expanded exactly3x with nearest neighbor. No repair or new generation. '
 +'In all108selected book positions, four sources are blank but only three overlap17blank outputs, hence14falseblanks. The public JSON already had14; earlier prose13was a marginal-subtraction error. '
 +'Pixel-based blank-flag audit finds17disagreements among2080frozen readings, including4creditedemptyimages (3candidate,1control). Among17uncreditedbookimages,15areemptyand2arenonempty;2otheremptybookimageswerecredited. Original readings and verdict unchanged; original credit counts are not corrected physical accuracy. '
 +'Panel c uses the unchanged minimum7book and1QA source-preferred cases. Complete-image log-likelihood margin0.001nat; a source is only a diagnostic comparator. '
 +'The2,642,368new head parameters retain over7.522Binherited visual/contextual parameters; head-only allocation is not whole-model footprint or matched-quality efficiency. '
 +'The subsequent context-gradient trial is separate and unmeasured; no interim language scores enter this figure.');
const candidate=path.join(stage,'candidate.pptx');await(await PresentationFile.exportPptx(deck)).save(candidate);
execFileSync(python,[path.join(root,'publication/ilm-nature/format_joint_writer_chart.py'),candidate]);
await fs.writeFile(path.join(stage,'figure_bindings.json'),JSON.stringify({measured_data_sha256:hash(raw),source_sha256:d.source_sha256,charts,tableValues,native_charts:2,native_tables:1,embedded_pixel_images:51,plotted_fraction_decimal_places:6,exact_numerators_and_denominators_in_table:true,original_readings_unchanged:true,new_model_calls:0},null,2)+'\n');
await finalizePresentation({workspaceDir:stage,candidatePath:candidate,finalPath:path.join(stage,'output/Joint_writer_physical_figures.pptx'),explicitTotalSlideCount:1,
 requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],materializeLiteralChartWorkbooks:true,
 fontPolicy:{basis:'reference',families:[font],referencePath:path.join(root,'publication/ilm-nature/figures/Complete_continuation_figures.pptx'),referenceSha256:hash(await fs.readFile(path.join(root,'publication/ilm-nature/figures/Complete_continuation_figures.pptx')))},
 pythonExecutable:python,integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','15240000,17145000','--validate-heading-fit','--require-native-table-slide','1'],
 verifyArtifactToolImport:true,receiptPath:path.join(stage,'presentation_validation.json')});
const final=await PresentationFile.importPptx(await FileBlob.load(path.join(stage,'output/Joint_writer_physical_figures.pptx')));
const slide=final.slides.items[0],preview=await slide.export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'final-reimport.png'),new Uint8Array(await preview.arrayBuffer()));
await fs.writeFile(path.join(stage,'final-reimport.layout.json'),await(await slide.export({format:'layout'})).text());
