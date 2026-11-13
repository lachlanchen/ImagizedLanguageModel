import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {Presentation, PresentationFile} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
if (![root, stage, skill, python].every(x => path.isAbsolute(x ?? ''))) throw Error('Absolute paths required');
const {applyPresentationChartFont, finalizePresentation} = await import(pathToFileURL(
  path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const hash = raw => createHash('sha256').update(raw).digest('hex');
const source = path.join(root,'publication/ilm-nature/evidence/complete_continuation_measured_data.json');
const bytes = await fs.readFile(source), data = JSON.parse(bytes);
if (!data.all_values_measured || data.original_pilot_passed || data.new_model_calls !== 0) throw Error('Completed measured results required');
for (const [name,digest] of Object.entries(data.source_sha256)) {
  if (hash(await fs.readFile(path.join(root,name))) !== digest) throw Error('Source changed: '+name);
}
await fs.mkdir(stage,{recursive:true});await fs.mkdir(path.join(stage,'output'),{recursive:true});
const deck=Presentation.create({slideSize:{width:1600,height:1600}});
const slide=deck.slides.add();slide.background.fill='#FFFFFF';
const font='Arial',ink='#192936',muted='#43535C',blue='#386985',brown='#9B7457',grid='#DDE4E8';
const charts=[];
function text(value,x,y,w,h,size=26,bold=false) {
  const shape=slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},
    fill:'none',line:{fill:'none',width:0}});
  shape.text=value;shape.text.style={typeface:font,fontSize:size,bold,color:bold?ink:muted,autoFit:'none'};
}
const axis={textStyle:{typeface:font,fontSize:23,fill:muted},line:{fill:muted,width:1}};
function chart(title,x,y,categories,series,maximum,major,format='0',legend=true) {
  const config={position:{left:x,top:y,width:700,height:470},title,titlePlacement:'aboveChart',
    titleTextStyle:{typeface:font,fontSize:29,bold:true,fill:ink},categories,
    series:series.map(s=>({...s,valuesFormatCode:format})),
    barOptions:{direction:'column',grouping:'clustered',gapWidth:90},hasLegend:legend,
    legend:{position:'bottom',overlay:false,textStyle:{fontSize:23,fill:muted}},xAxis:axis,
    yAxis:{...axis,min:0,max:maximum,majorUnit:major,numberFormatCode:format,majorGridlines:{fill:grid,width:1}},
    dataLabels:{showValue:true,position:'outEnd',textStyle:{typeface:font,fontSize:23,fill:ink}}};
  const item=slide.charts.add('bar',config);applyPresentationChartFont(item,{fontFamily:font});
  charts.push({title,categories,series:config.series.map(({name,values})=>({name,values})),maximum,format});
}
const labels=data.steps.map(x=>x.toLocaleString('en-US'));
text('Complete visual continuation: answers and writing',48,28,1504,70,43,true);
text('a',42,137,40,50,34,true);
chart('Complete exact answers / 32',80,140,labels,[
  {name:'Fixed cells',fill:brown,values:data.qa_series.fixed_cells.complete_exact},
  {name:'Proportional',fill:blue,values:data.qa_series.proportional.complete_exact}],32,8);
text('Optimizer updates; all five checkpoints are fixed.\nThe final model remains the 41,298-update snapshot.',85,635,700,85,25);
text('b',832,137,40,50,34,true);
chart('Readable complete answers / 32',870,140,labels,[
  {name:'Fixed cells',fill:brown,values:data.qa_series.fixed_cells.readable},
  {name:'Proportional',fill:blue,values:data.qa_series.proportional.readable}],32,8);
text('Whole-response readings are frozen before scoring.\nReadable writing alone does not establish correctness.',875,635,685,85,25);

text('c',42,802,40,50,34,true);
chart('Book tiles above clean ink threshold',80,805,['Initial free','Final free','Final supplied'],[
  {name:'Fraction of tiles',fill:blue,values:data.book_density.map(x=>Number(x.fraction.toFixed(6)))}],1.05,.25,'0.0%',false);
text('Clean-book 99th percentile: '+data.clean_book_ink_99+' black pixels / 1,024.\n'
  +'All 12 cases; tiles: '+data.book_density.map(x=>x.glyphs.toLocaleString('en-US')).join(' / ')+'.\n'
  +'Supplied history and length are diagnostic.',85,1310,700,120,25);

text('d',832,802,40,50,34,true);
text('Final complete-language outcomes',875,802,680,55,30,true);
const values=data.final_table;
const table=slide.tables.add({rows:values.length,columns:3,left:880,top:885,width:660,height:340,
  columnWidths:[300,180,180],values});
table.styleOptions={headerRow:true,bandedRows:false};table.borders.assign({style:'solid',fill:grid,width:1});
for(let r=0;r<values.length;r++)for(let c=0;c<3;c++) {
  const cell=table.getCell(r,c);cell.fill=r===0?'#EFF4F6':'#FFFFFF';
  cell.text.style={typeface:font,fontSize:25,bold:r===0,color:r===0?ink:muted};
}
text('Long books: '+data.final_long_book_stops+'/12 stops; 0/12 coherent complete.\n'
  +'Original and four-erasure audits reject the pilot.',885,1250,660,85,25);
const fraction=data.later_proportional_QA.fraction_at_least_095;
text('With supplied history: '+Math.round(fraction*data.later_proportional_QA.positions)+'/713 later QA tiles\n'
  +'reach F1 ≥0.95. This is not an answer score.',885,1355,660,85,25);
text('13,766 complete sources · 41,298 updates · '+data.training.peak_allocated_vram_gib.toFixed(3)+' GiB peak allocated VRAM',50,1450,1500,45,25);
text((data.whole_model_parameters.total_unique_parameters/1e9).toFixed(3)+'B whole model; '
  +(data.whole_model_parameters.trainable_parameters/1e6).toFixed(3)+'M trainables. Single-device execution is not matched-quality efficiency.',50,1500,1500,45,24);
text('Reused development; one model reviewer; no independent human adjudication. Joint-writer quality is unmeasured.',50,1550,1500,43,23);
slide.speakerNotes.textFrame.setText(
  'Sources: publication/ilm-nature/evidence/complete_continuation_measured_data.json and its three SHA-bound public results. '
  +'Panels a/b retain 32 questions and 16 pairs per layout at all five fixed snapshots; no intermediate checkpoint is selected. '
  +'Counts assess whole answers, not glyph accuracy. Canonical mismatch alone is not semantic falsity; frozen uncertainty remains. '
  +'Panel c pools tiles across all twelve book cases. Its 2,122/6,647/1,161 denominators differ. '
  +'Supplied history provides answer images and length with stopping bypassed; this is a physical diagnosis, not autonomous language. '
  +'Excess ink does not prove a stable attractor or identify the loss as the sole cause. '
  +'Panel d gives final pair and retention counts. Books use only the fixed prompt layout. '
  +'No normal book arm has a credited coherent complete continuation; the final 640-cap candidate stops on 8/12 cases. '
  +'656/713 later proportional QA tiles reach F1 .95 only under supplied history. '
  +'The complete body has 7,578,390,003 parameters, including 7,522,463,984 inherited parameters; 20,212,225 trainables do not establish a compact foundation. '
  +'Training costs 75,209.326 cumulative child seconds; preparation, evaluation, review and inherited training remain separate. '
  +'Density chart values round to six fraction decimals for portable workbook storage; source counts and full precision remain in the bound JSON. '
  +'The new joint writer is unmeasured and is not plotted as zero. Charts and tables contain literal measured data.');
const candidate=path.join(stage,'candidate.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidate);
execFileSync(python,[path.join(root,'publication/ilm-nature/format_complete_continuation_labels.py'),candidate]);
const preview=await slide.export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'authoring.png'),new Uint8Array(await preview.arrayBuffer()));
const layout=await slide.export({format:'layout'});
await fs.writeFile(path.join(stage,'authoring.layout.json'),await layout.text());
await fs.writeFile(path.join(stage,'figure_bindings.json'),JSON.stringify({
  measured_data_sha256:hash(bytes),source_sha256:data.source_sha256,
  builder_sha256:hash(await fs.readFile(path.join(root,'publication/ilm-nature/complete_continuation_figures.mjs'))),
  formatter_sha256:hash(await fs.readFile(path.join(root,'publication/ilm-nature/format_complete_continuation_labels.py'))),
  charts,tableValues:values,native_charts:3,native_tables:1,plotted_fraction_decimal_places:6,new_model_calls:0,
  supplied_history_not_language:true,joint_writer_quality_unmeasured:true},null,2)+'\n');
await finalizePresentation({workspaceDir:stage,candidatePath:candidate,
  finalPath:path.join(stage,'output/Complete_continuation_figures.pptx'),explicitTotalSlideCount:1,
  requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],materializeLiteralChartWorkbooks:true,
  fontPolicy:{basis:'reference',families:[font],
    referencePath:path.join(root,'publication/ilm-nature/figures/Continuation_history_figures.pptx'),
    referenceSha256:hash(await fs.readFile(path.join(root,'publication/ilm-nature/figures/Continuation_history_figures.pptx')))},
  pythonExecutable:python,
  integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','15240000,15240000','--validate-heading-fit','--require-native-table-slide','1'],
  verifyArtifactToolImport:true,receiptPath:path.join(stage,'presentation_validation.json')});
