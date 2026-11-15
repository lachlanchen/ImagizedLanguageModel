import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root,stage,skill,python]=process.argv.slice(2);
if(![root,stage,skill,python].every(x=>path.isAbsolute(x??'')))throw Error('Absolute paths required');
const {applyPresentationChartFont,finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const hash=b=>createHash('sha256').update(b).digest('hex');
const raw=await fs.readFile(path.join(root,'publication/ilm-nature/evidence/ranking_completed_figure_data.json')),d=JSON.parse(raw);
if(d.physical_gate_passed||d.complete_language!=='unmeasured'||d.samples.length!==6||d.whitespace_failures.length!==3)throw Error('Completed failed ranking comparison required');
for(const [name,digest] of Object.entries(d.source_sha256))if(hash(await fs.readFile(path.join(root,name)))!==digest)throw Error('Evidence changed: '+name);
await fs.mkdir(path.join(stage,'output'),{recursive:true});
const deck=Presentation.create({slideSize:{width:1600,height:1900}}),s=deck.slides.add();s.background.fill='#FFFFFF';
const font='Arial',ink='#192936',muted='#43535C',blue='#386985',brown='#9B7457',red='#A74337',grid='#DDE4E8';
function text(value,x,y,w,h,size=26,bold=false,color=undefined){
 const q=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 q.text=value;q.text.style={typeface:font,fontSize:size,bold,color:color??(bold?ink:muted),autoFit:'none'};
}
text('Conditional glyph readability and whitespace',48,28,1504,62,42,true);
text('Completed nonempty-ranking comparison with supplied reference-image history',50,101,1500,55,27);
text('a',42,190,40,45,34,true);
const tableValues=[['Readable forms / selected','Control re-review','Nonempty ranking'],
 ...d.groups.map(g=>[g.label,`${g.control}/${g.count}`,`${g.ranking}/${g.count}`])];
const table=s.tables.add({rows:8,columns:3,left:85,top:190,width:1460,height:384,
 columnWidths:[780,340,340],values:tableValues});
table.styleOptions={headerRow:true,bandedRows:false};table.borders.assign({style:'solid',fill:grid,width:1});
for(let r=0;r<8;r++)for(let c=0;c<3;c++){
 const cell=table.getCell(r,c);cell.fill=r===0?'#EFF4F6':'#FFFFFF';
 cell.text.style={typeface:font,fontSize:26,bold:r===0,color:r===0?ink:muted};
}
text('Totals: control 981/1,040; ranking 1,002/1,040. Every ranking group exceeds 90%.\nReadability does not require source identity. The control pixels are unchanged.',85,594,1460,72,26);
const axis={textStyle:{typeface:font,fontSize:24,fill:muted},line:{fill:muted,width:1}},charts=[];
function chart(title,x,y,w,h,control,ranking,max,major){
 const categories=['Complete cohort'],series=[{name:'Likelihood control',values:[control],fill:brown},{name:'Nonempty ranking',values:[ranking],fill:blue}];
 const q=s.charts.add('bar',{name:title,title,position:{left:x,top:y,width:w,height:h},categories,series,
 titleTextStyle:{typeface:font,fontSize:28,bold:true,fill:ink},barOptions:{direction:'column',grouping:'clustered',gapWidth:110},
 hasLegend:true,legend:{position:'bottom',overlay:false,textStyle:{typeface:font,fontSize:24,fill:muted}},
 xAxis:axis,yAxis:{...axis,min:0,max,majorUnit:major,numberFormatCode:'0',majorGridlines:{fill:grid,width:1}},
 dataLabels:{showValue:true,position:'outEnd',textStyle:{typeface:font,fontSize:26,fill:ink}}});
 applyPresentationChartFont(q,{fontFamily:font});charts.push({title,categories,series:series.map(({name,values})=>({name,values})),max,major});
}
text('b',42,695,40,45,34,true);text('c',827,695,40,45,34,true);
chart('False blanks / 12,597 targets',85,690,710,390,d.totals.parent.false_blanks,d.totals.candidate.false_blanks,800,200);
chart('False ink / 252 white targets',865,690,680,390,d.totals.parent.false_ink,d.totals.candidate.false_ink,30,10);
text('False blanks exclude true-white sources.',85,1090,710,40,25);
text('Whitespace gate fails: 25 exceeds the limit of 11.',865,1090,680,40,25,true,red);
text('d',42,1150,40,45,34,true);
text('Original glyphs at fixed source positions',85,1150,1460,45,30,true);
text('Each triplet: source, likelihood control, nonempty ranking',85,1198,1460,40,25);
async function triplet(item,x,y,size,step){
 for(let j=0;j<3;j++){
  const role=['source','control','ranking'][j],a=item.images[role],bytes=await fs.readFile(path.join(root,a.file));
  if(hash(bytes)!==a.sha256)throw Error('Original glyph changed');
  s.shapes.add({geometry:'rect',position:{left:x+j*step-1,top:y-1,width:size+2,height:size+2},fill:'none',line:{fill:grid,width:1}});
  s.images.add({blob:bytes,contentType:'image/png',fit:'contain',position:{left:x+j*step,top:y,width:size,height:size},
   alt:`Original ${item.case} position ${item.position} ${role}; binary pixels expanded3x nearest`});
 }
}
for(let i=0;i<d.samples.length;i++){
 const item=d.samples[i],x=90+i*239;
 const number=item.case.match(/book_(\d+)/)?.[1];if(!number)throw Error('Development-book sample required');
 text(`D${number}:${item.position}`,x,1244,224,36,24);
 await triplet(item,x,1290,69,73);
}
text('First six source-ink images in original case/selected-position order. Candidate success does not select the samples.',85,1380,1460,65,25);
text('e',42,1470,40,45,34,true);
text('First three whitespace failures',85,1470,1460,45,30,true);
for(let i=0;i<d.whitespace_failures.length;i++){
 const item=d.whitespace_failures[i],x=130+i*470;
 const label=item.case.replace('train_','Train ').replace('_fixed_cells',' fixed').replace('_proportional',' proportional');
 text(`${label}:${item.position}`,x,1520,440,40,24);
 await triplet(item,x,1570,88,92);
}
text(`Exact source bitmaps: control ${d.totals.parent.source_pixel_exact.toLocaleString('en-US')}/12,597; ranking ${d.totals.candidate.source_pixel_exact.toLocaleString('en-US')}/12,597.\nBlank choice fixes 3 whitespace errors, leaves 22 and adds 1 false blank.`,85,1695,1460,75,26);
text('Conditional glyph quality only. Autonomous language remains unmeasured.\nModel review has prior familiarity and no independent human adjudication.',50,1790,1500,90,26);
s.speakerNotes.textFrame.setText(`Completed scientific evidence: publication/ilm-nature/evidence/ranking_completed_figure_data.json and the three bound physical, whitespace-density and preflight exports listed there.

Panel a reports all seven original groups from one frozen opaque review. A credit means readable nonempty writing without uncertainty; source identity and complete semantics are not required. Control pixels are unchanged. Their fresh review gives 981/1,040 credits. The original 970/1,040 credits and detected review errors remain preserved.

All 33 boards and 2,080 images were inspected before the sole freeze and new owner/reference scoring. There are 28 uncertain readings and 69 actually empty reviewed images. No credited candidate is empty. Initial unfrozen entry alignment corrections were preserved; frozen readings were not repaired. The reviewer was a model with prior cohort familiarity, without independent human adjudication.

Panels b,c recount all 12,597 positions: candidate false blanks 37 versus 711 control; candidate false ink 25 of 252 truly white sources versus 11 control. All readable groups pass but the prospective whitespace maximum of eleven fails. No autonomous language was generated for this actor.

Panel d follows the first six source-ink selected development-book positions in original case/position order, without selecting candidate success. Panel e deliberately shows the first three false-ink source-white failures in original complete-case/position order. The original binary 32-by-32 pixels receive only threefold nearest-neighbour expansion. Wrong or uncertain forms are retained. Exact source-bitmap identity is 4,603/12,597 control versus 4,609/12,597 ranking. Several shown outputs are simple marks different from their source. Identity is a fidelity diagnostic, not itself a complete-language error judgment.

The complete-density rule selects blank only when its complete log probability exceeds the saved image by 0.001 nat. It would fix three white errors but add one nonempty erasure, leaving 22 false-ink outputs and 38 false blanks. This is counterfactual eligibility, not new physical review or a changed gate.

A separately registered objective for blank targets passes a six-case reset full-body preflight. Its full quality comparison remains running and unmeasured. The deployment retains 7.553 billion parameters including 7.522 billion inherited parameters. Useful independent language, compactness, historical transfer and matched-quality efficiency remain unproved.`);
const candidate=path.join(stage,'candidate.pptx');await(await PresentationFile.exportPptx(deck)).save(candidate);
await fs.writeFile(path.join(stage,'figure_bindings.json'),JSON.stringify({data_sha256:hash(raw),source_sha256:d.source_sha256,charts,tableValues,
 native_charts:2,native_tables:1,embedded_pixel_image_instances:27,new_model_calls:0,original_readings_unchanged:true},null,2)+'\n');
const reference=path.join(root,'publication/ilm-nature/figures/Context_likelihood_failure_figures.pptx');
await finalizePresentation({workspaceDir:stage,candidatePath:candidate,finalPath:path.join(stage,'output/Ranking_completed_figures.pptx'),explicitTotalSlideCount:1,
 requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],materializeLiteralChartWorkbooks:true,
 fontPolicy:{basis:'reference',families:[font],referencePath:reference,referenceSha256:hash(await fs.readFile(reference))},
 pythonExecutable:python,integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','15240000,18097500','--validate-heading-fit','--require-native-table-slide','1'],
 verifyArtifactToolImport:true,receiptPath:path.join(stage,'presentation_validation.json')});
const final=await PresentationFile.importPptx(await FileBlob.load(path.join(stage,'output/Ranking_completed_figures.pptx')));
const slide=final.slides.items[0];await fs.writeFile(path.join(stage,'final-reimport.png'),new Uint8Array(await(await slide.export({format:'png',scale:1})).arrayBuffer()));
await fs.writeFile(path.join(stage,'final-reimport.layout.json'),await(await slide.export({format:'layout'})).text());
