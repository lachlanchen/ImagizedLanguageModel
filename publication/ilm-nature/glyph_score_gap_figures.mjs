import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const [root,stage,skill,python]=process.argv.slice(2);
if(![root,stage,skill,python].every(x=>path.isAbsolute(x??'')))throw Error('Absolute paths required');
const {applyPresentationChartFont,finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const hash=b=>createHash('sha256').update(b).digest('hex');
const raw=await fs.readFile(path.join(root,'publication/ilm-nature/evidence/glyph_score_gap_figure_data.json')),d=JSON.parse(raw);
if(d.groups.length!==7||d.books.length!==3||d.semantic_accuracy_measured||d.complete_language!=='unmeasured')throw Error('Completed descriptive comparison required');
for(const [name,digest] of Object.entries(d.source_sha256))if(hash(await fs.readFile(path.join(root,name)))!==digest)throw Error('Evidence changed: '+name);
await fs.mkdir(path.join(stage,'output'),{recursive:true});
const deck=Presentation.create({slideSize:{width:1600,height:1700}}),s=deck.slides.add();s.background.fill='#FFFFFF';
const font='Arial',ink='#192936',muted='#43535C',blue='#386985',grey='#7A8791',grid='#DDE4E8';
function text(value,x,y,w,h,size=27,bold=false){
 const q=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 q.text=value;q.text.style={typeface:font,fontSize:size,bold,color:bold?ink:muted,autoFit:'none'};
}
text('Source and emitted image probabilities',48,28,1504,65,42,true);
text('Completed ranked writer with supplied reference-image history',50,105,1500,50,27);
text('a',42,180,40,45,34,true);
const categories=d.books.map(x=>x.label),series=[
 {name:'Source preferred',values:d.books.map(x=>x.observed_fraction),fill:blue},
 {name:'Required by book rule',values:d.books.map(x=>x.required_fraction),fill:grey}];
const title='Source preference among nonempty book mismatches';
const axis={textStyle:{typeface:font,fontSize:26,fill:muted},line:{fill:muted,width:1}};
const chart=s.charts.add('bar',{name:title,title,position:{left:85,top:180,width:1460,height:510},
 categories,series,titleTextStyle:{typeface:font,fontSize:30,bold:true,fill:ink},
 barOptions:{direction:'column',grouping:'clustered',gapWidth:100},hasLegend:true,
 legend:{position:'bottom',overlay:false,textStyle:{typeface:font,fontSize:26,fill:muted}},
 xAxis:axis,yAxis:{...axis,min:0,max:.6,majorUnit:.2,numberFormatCode:'0%',majorGridlines:{fill:grid,width:1}},
 dataLabels:{showValue:true,position:'outEnd',textStyle:{typeface:font,fontSize:27,fill:ink}}});
applyPresentationChartFont(chart,{fontFamily:font});
text('Source counts / required counts: 51 / 466, 287 / 1,632 and 298 / 1,627.\nAll three rules fail. The rule requires at least half in each book group.',85,710,1460,85,27);
text('b',42,830,40,45,34,true);
text('All seven groups at nonempty bitmap mismatches',85,830,1460,45,30,true);
text('Preference requires a complete-image score advantage above 0.001 nat',85,884,1460,44,26);
const tableValues=[['Group','Mismatches','Source','Emitted','Ties'],
 ...d.groups.map(x=>[x.label,...['nonempty_mismatches','source_preferred','emitted_preferred','tied'].map(k=>x[k].toLocaleString('en-US'))])];
const table=s.tables.add({rows:8,columns:5,left:85,top:940,width:1460,height:368,
 columnWidths:[660,230,180,230,160],values:tableValues});
table.styleOptions={headerRow:true,bandedRows:false};table.borders.assign({style:'solid',fill:grid,width:1});
for(let r=0;r<8;r++)for(let c=0;c<5;c++){
 const cell=table.getCell(r,c);cell.fill=r===0?'#EFF4F6':'#FFFFFF';
 cell.text.style={typeface:font,fontSize:26,bold:r===0,color:r===0?ink:muted};
}
text('Totals: 7,926 mismatches, 717 source preferences, 7,206 emitted preferences and 3 ties.',85,1330,1460,65,27);
text('c',42,1420,40,45,34,true);
text('Measured image scores and language quality',85,1420,1460,45,30,true);
text('S = log q(source | history)     E = log q(emitted | history)\nAt 7,206 mismatches, E > S + 0.001. The source is below an exhibited alternative.',85,1475,1460,95,27);
text('A valid continuation can differ from the reference book. These counts measure canonical\nimage fidelity. Complete language and the running white-target trial remain unmeasured.',50,1590,1500,95,27);
s.speakerNotes.textFrame.setText(`Completed source evidence: references/evidence/ranked_glyph_score_gap_result_20261116.json and ranked_glyph_score_gap_verification_20261116.json, bound in publication/ilm-nature/evidence/glyph_score_gap_figure_data.json.

All 124 cases and 12,597 positions use the original saved probabilities and binary source/output arrays. Both images contain ink and their bitmaps differ at 7,926 positions. The source has greater complete log probability by more than 0.001 nat at 717 positions, the emitted image at 7,206, with three ties. Original exact bitmaps 4,609, false blanks 37, false ink 25 and genuine white sources 252 remain separate counts. No image or frozen reading was repaired.

Panel a uses only the three prospectively specified book strata. Required source-preference counts are the ceiling of half the mismatch denominator. Chart fractions round to six decimals before workbook materialization; displayed percentages round further. The exact numerator/denominator counts remain in the table and source data. No QA readability gate is inferred from this diagnostic rule.

Panel b retains all seven group denominators and every source/emitted/tie count as native table cells. A bitmap difference is not a semantic error. The cohort was previously examined, so this descriptive calculation is not untouched confirmation or independent human adjudication.

An emitted alternative above the source excludes the source from the highest score in the measured arithmetic. This neither proves the emitted image globally optimal nor measures prefix logits. Greater source score at other positions does not establish a practical search. A useful language model need not reproduce every reference-book glyph. Complete prompt-dependent language remains the goal.

The vector and independent scalar calculations agree. The CPU-only calculation performs no neural replay, optimizer update or new glyph generation. The separate white-target fit retains all 115 sources and its original schedule, gates and owner. Its quality remains unmeasured. The active model retains 7.553 billion parameters, including inherited weights. No compactness or matched-quality efficiency result follows.`);
const candidate=path.join(stage,'candidate.pptx');await(await PresentationFile.exportPptx(deck)).save(candidate);
await fs.writeFile(path.join(stage,'figure_bindings.json'),JSON.stringify({data_sha256:hash(raw),source_sha256:d.source_sha256,
 charts:[{title,categories,series:series.map(({name,values})=>({name,values}))}],tableValues,
 native_charts:1,native_tables:1,embedded_pixel_image_instances:0,new_model_calls:0},null,2)+'\n');
const reference=path.join(root,'publication/ilm-nature/figures/Ranking_completed_figures.pptx');
await finalizePresentation({workspaceDir:stage,candidatePath:candidate,finalPath:path.join(stage,'output/Glyph_score_gap_figures.pptx'),explicitTotalSlideCount:1,
 requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],materializeLiteralChartWorkbooks:true,
 fontPolicy:{basis:'reference',families:[font],referencePath:reference,referenceSha256:hash(await fs.readFile(reference))},
 pythonExecutable:python,integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','15240000,16192500','--validate-heading-fit','--require-native-table-slide','1'],
 verifyArtifactToolImport:true,receiptPath:path.join(stage,'presentation_validation.json')});
const final=await PresentationFile.importPptx(await FileBlob.load(path.join(stage,'output/Glyph_score_gap_figures.pptx')));
const slide=final.slides.items[0];await fs.writeFile(path.join(stage,'final-reimport.png'),new Uint8Array(await(await slide.export({format:'png',scale:1})).arrayBuffer()));
await fs.writeFile(path.join(stage,'final-reimport.layout.json'),await(await slide.export({format:'layout'})).text());
