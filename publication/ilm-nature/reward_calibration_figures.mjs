import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';
const [root,stage,skill,python]=process.argv.slice(2);
if(![root,stage,skill,python].every(x=>path.isAbsolute(x??'')))throw Error('Absolute paths required');
const {applyPresentationChartFont,finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const hash=b=>createHash('sha256').update(b).digest('hex');
const raw=await fs.readFile(path.join(root,'publication/ilm-nature/evidence/reward_calibration_figure_data.json')),d=JSON.parse(raw);
if(!d.both_calibrations_failed||d.student_training||d.samples.length!==3)throw Error('Completed negative controls required');
for(const [name,digest] of Object.entries(d.source_sha256))if(hash(await fs.readFile(path.join(root,name)))!==digest)throw Error('Evidence changed: '+name);
await fs.mkdir(path.join(stage,'output'),{recursive:true});
const deck=Presentation.create({slideSize:{width:1600,height:1700}}),s=deck.slides.add();s.background.fill='#FFFFFF';
const font='Arial',ink='#192936',muted='#43535C',blue='#386985',grey='#7A8791',grid='#DDE4E8';
function text(value,x,y,w,h,size=27,bold=false){const q=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});q.text=value;q.text.style={typeface:font,fontSize:size,bold,color:bold?ink:muted,autoFit:'none'};}
text('Training rewards fail conservative verification',48,28,1504,65,42,true);
text('64 frozen answer images from the completed continuation model',50,105,1500,50,27);
const title='Reward extrema in frozen review groups';
const series=d.extrema.map((r,i)=>({name:r.name,values:r.values,fill:i===0?blue:grey}));
const axis={textStyle:{typeface:font,fontSize:26,fill:muted},line:{fill:muted,width:1}};
const chart=s.charts.add('bar',{name:title,title,position:{left:85,top:225,width:1460,height:425},
 categories:d.categories,series,titleTextStyle:{typeface:font,fontSize:30,bold:true,fill:ink},
 barOptions:{direction:'column',grouping:'clustered',gapWidth:100},hasLegend:true,
 legend:{position:'bottom',overlay:false,textStyle:{typeface:font,fontSize:26,fill:muted}},
 xAxis:axis,yAxis:{...axis,min:0,max:1.1,majorUnit:.25,numberFormatCode:'0.00',majorGridlines:{fill:grid,width:1}},
 dataLabels:{showValue:false}});applyPresentationChartFont(chart,{fontFamily:font});
text('a',42,180,40,45,34,true);
text('Image edit: minimum correct 0.485744, maximum uncredited 0.492818.\nSelective verifier: minimum correct 0.428571, maximum uncredited 1.000000.',85,680,1460,85,27);
text('b',42,805,40,45,34,true);text('Selective verifier decisions on complete answers',85,805,1460,45,30,true);
const tableValues=[['Frozen review group','Cases','Accepted'],...d.groups.map(r=>[r.label,String(r.cases),String(r.selective_accepted)])];
const table=s.tables.add({rows:4,columns:3,left:85,top:870,width:1460,height:184,columnWidths:[820,320,320],values:tableValues});
table.styleOptions={headerRow:true,bandedRows:false};table.borders.assign({style:'solid',fill:grid,width:1});
for(let r=0;r<4;r++)for(let c=0;c<3;c++){const cell=table.getCell(r,c);cell.fill=r===0?'#EFF4F6':'#FFFFFF';cell.text.style={typeface:font,fontSize:26,bold:r===0,color:r===0?ink:muted};}
text('The training-only inventory contains 9,581 forms. Recognition can abstain on each cell.',85,1070,1460,45,26);
text('c',42,1150,40,45,34,true);text('Every selective verifier disagreement with the frozen review',85,1150,1460,45,30,true);
for(let i=0;i<d.samples.length;i++){
 const a=d.samples[i],bytes=await fs.readFile(path.join(root,a.file));if(hash(bytes)!==a.sha256)throw Error('Original answer changed');
 const y=1215+i*112;
 s.images.add({blob:bytes,contentType:'image/png',fit:'contain',position:{left:85,top:y,width:a.width*2,height:a.height*2},alt:`Original ${a.case} ${a.layout}; frozen reading unchanged`});
 text(`${a.case.replace('_normal','')} / ${a.layout==='fixed_cells'?'fixed':'proportional'}`,85,y+67,770,30,22);
 const label=a.frozen_correct?'Known correct\nOne cell abstained':'Frozen reading uncertain\nAll 10 cells accepted';text(label,900,y,640,80,26);
}
text('Original answer pixels and frozen readings remain unchanged. Both calibrations fail.\nThe inventory stays outside deployment. No student training or language gain is measured.',50,1585,1500,95,26);
s.speakerNotes.textFrame.setText(`Sources: references/evidence/whole_answer_reward_calibration_20261116.json and selective_visual_verifier_calibration_20261116.json, with both preregistrations. Figure data and original PNG hashes: publication/ilm-nature/evidence/reward_calibration_figure_data.json.
All 64 final normal QA answers remain: 32 per presentation,9strict complete-correct,9readable-wrong and46without a credited clear reading. Existing uncertainty is not proof of semantic error. The original model reviewer had cohort familiarity and there is no independent human adjudication. No original reading or score is repaired.
Panel a plots the minimum correct and maximum other-group rewards. Values round to six decimal places in the literal chart workbook. Image-edit AUC is1.0 and selective AUC is0.975309 among only the18known-readable outcomes. Neither pooled ranking value establishes a trustworthy training reward. A fixed-font image-edit reward can rank an uncredited output above a known-correct one.
Panel b retains all selective full-content acceptance counts. The finite bank has9581unique binary forms from7255223training glyph positions in13785training records. No development/reference prototype or Unicode label enters it. Disjoint Hamming balls ensure unique bounded-corruption decoding, not arbitrary-image legibility. The component is forbidden in deployed student generation.
Panel c follows the complete predetermined disagreement rule: every known-correct answer not fully accepted and every uncredited answer fully accepted. These are the only three rows. All original PNG bytes are embedded unchanged; there is no source prototype or reference transcription. Two known-correct outputs each have one abstained cell. The uncertain output's fourth glyph remains uncertain in the frozen reading even though the verifier accepts its sequence. Preserve this contradiction.
Both studies make zero neural calls and zero optimizer updates. Image scoring takes0.538744seconds at0.630943GiB peak processRSS; selective calibration takes7.457499seconds at0.095661GiB, including6.404933seconds for inventory construction. Imports/source hashing precede timers. CPU contract tests and the generic policy helper's explicit-reward guard are implementation evidence only. No policy fit, model improvement, historical transfer or matched-quality efficiency is established. The separate white-target training schedule,115sources and owner are unchanged and its quality remains unmeasured.`);
await(await PresentationFile.exportPptx(deck)).save(path.join(stage,'candidate.pptx'));
await fs.writeFile(path.join(stage,'figure_bindings.json'),JSON.stringify({data_sha256:hash(raw),source_sha256:d.source_sha256,
 charts:[{title,categories:d.categories,series:series.map(({name,values})=>({name,values}))}],tableValues,
 native_charts:1,native_tables:1,embedded_pixel_image_instances:3,new_model_calls:0},null,2)+'\n');
const reference=path.join(root,'publication/ilm-nature/figures/Glyph_score_gap_figures.pptx');
await finalizePresentation({workspaceDir:stage,candidatePath:path.join(stage,'candidate.pptx'),finalPath:path.join(stage,'output/Reward_calibration_figures.pptx'),explicitTotalSlideCount:1,
 requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],materializeLiteralChartWorkbooks:true,
 fontPolicy:{basis:'reference',families:[font],referencePath:reference,referenceSha256:hash(await fs.readFile(reference))},
 pythonExecutable:python,integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','15240000,16192500','--validate-heading-fit','--require-native-table-slide','1'],
 verifyArtifactToolImport:true,receiptPath:path.join(stage,'presentation_validation.json')});
const final=await PresentationFile.importPptx(await FileBlob.load(path.join(stage,'output/Reward_calibration_figures.pptx')));
await fs.writeFile(path.join(stage,'final-reimport.png'),new Uint8Array(await(await final.slides.items[0].export({format:'png',scale:1})).arrayBuffer()));
await fs.writeFile(path.join(stage,'final-reimport.layout.json'),await(await final.slides.items[0].export({format:'layout'})).text());
