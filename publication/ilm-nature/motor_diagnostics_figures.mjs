import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import sharp from 'sharp';
import {Presentation,PresentationFile} from '@oai/artifact-tool';

const [root,stage,skill,python]=process.argv.slice(2);
const {applyPresentationChartFont,finalizePresentation}=await import(pathToFileURL(
  path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const hash=x=>createHash('sha256').update(x).digest('hex');
const data=JSON.parse(await fs.readFile(path.join(stage,'measured_data.json'),'utf8'));
const deck=Presentation.create({slideSize:{width:1600,height:1600}});
const s=deck.slides.add();s.background.fill='#FFFFFF';
const font='Arial',ink='#192936',muted='#43535C',blue='#386985',brown='#9B7457',grid='#DDE4E8';
const charts=[],pictures=[];
function text(value,x,y,w,h,size=27,bold=false){
  const z=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  z.text=value;z.text.style={typeface:font,fontSize:size,bold,color:bold?ink:muted,autoFit:'none'};
}
const axis={textStyle:{typeface:font,fontSize:23,fill:muted},line:{fill:muted,width:1}};
const titleTextStyle={typeface:font,fontSize:29,bold:true,fill:ink};
function chart(config){
  const c=s.charts.add('bar',config);applyPresentationChartFont(c,{fontFamily:font});
  charts.push({categories:config.categories,series:config.series.map(x=>({name:x.name,values:x.values})),position:config.position});
}
text('Conditional motor repair and its limits',48,28,1504,70,46,true);
const layouts=['fixed_cells','proportional'];
for(const [i,[field,title,max]] of [['exact','Complete exact QA (32)',32],['pairs','Both correct (16 pairs)',16]].entries()){
  text('ab'[i],42+i*790,124,40,50,34,true);
  chart({position:{left:75+i*790,top:125,width:710,height:380},title,titlePlacement:'aboveChart',titleTextStyle,
    categories:['Fixed cells','Proportional'],series:[
      {name:'Parent',values:layouts.map(l=>data.qa.parent[l][field]),fill:blue},
      {name:'Flow',values:layouts.map(l=>data.qa.flow[l][field]),fill:brown}],
    barOptions:{direction:'column',grouping:'clustered',gapWidth:100},hasLegend:true,
    legend:{position:'bottom',overlay:false,textStyle:{fontSize:22,fill:muted}},xAxis:axis,
    yAxis:{...axis,min:0,max,majorUnit:max/4,numberFormatCode:'0',majorGridlines:{fill:grid,width:1}},
    dataLabels:{showValue:true,position:'outEnd',textStyle:{typeface:font,fontSize:29,fill:ink}}});
}
text('Flow retains 5/6 proportional exact answers. The registered language pilot fails.\nProportional exact-QA difference: +3.125 points (95% interval −6.25 to 12.5).',80,522,1470,83,27);
const routes=['parent','flow','decoded','codec'];
for(const [i,[field,title,max,unit,format]] of [
 ['median_foreground_f1','Training median foreground F1',1.2,.25,'0.00'],
 ['at_least_095','Training tiles with F1 ≥ 0.95 (32)',36,8,'0']].entries()){
  text('cd'[i],42+i*790,625,40,50,34,true);
  chart({position:{left:75+i*790,top:625,width:710,height:350},title,titlePlacement:'aboveChart',titleTextStyle,
    categories:['Parent','Flow','Pixel loss','Codec'],series:[
      {name:'First',valuesFormatCode:i===0?'0.000000':'0',values:routes.map(r=>field==='median_foreground_f1'?Number(data.physical.train_first[r][field].toFixed(6)):data.physical.train_first[r][field]),fill:blue},
      {name:'Later',valuesFormatCode:i===0?'0.000000':'0',values:routes.map(r=>field==='median_foreground_f1'?Number(data.physical.train_later[r][field].toFixed(6)):data.physical.train_later[r][field]),fill:brown}],
    barOptions:{direction:i===0?'bar':'column',grouping:'clustered',gapWidth:65},hasLegend:true,
    legend:{position:'bottom',overlay:false,textStyle:{fontSize:22,fill:muted}},xAxis:axis,
    yAxis:{...axis,min:0,max,majorUnit:unit,numberFormatCode:format,majorGridlines:{fill:grid,width:1}},
    dataLabels:{showValue:true,position:'outEnd',textStyle:{typeface:font,fontSize:i===0?20:26,fill:ink}}});
}
text('Each pool requires median ≥ 0.99 and ≥ 30/32 tiles at F1 ≥ 0.95. Both new writers fail.\nPixel-loss whole-answer scores are unmeasured. Codec receives the target image.',80,990,1470,80,26);
text('e   Original tile excerpts',48,1090,750,50,31,true);
for(const [i,title] of ['Target','Parent','Flow','Pixel loss','Codec'].entries())text(title,555+195*i,1145,190,40,26,true);
for(const [j,[key,title]] of [
 ['first_02321','Training: recovered first form'],['later_15112','Training: damaged later form'],
 ['proportional__qa_02_b_normal','QA: native prefix agrees'],['proportional__qa_09_a_normal','QA: recovered area initial']].entries()){
  text(title,60,1200+j*78,480,58,25);
  for(const [i,kind] of ['target','parent','flow','decoded','codec'].entries()){
    const binding=data.pictures.find(p=>p.case===key&&p.kind===kind);
    const raw=await fs.readFile(path.join(stage,binding.file));
    if(hash(raw)!==binding.sha256)throw Error('Original tile bytes changed');
    if(hash(await sharp(raw).removeAlpha().greyscale().raw().toBuffer())!==binding.gray_pixels_sha256)throw Error('Original pixels changed');
    const position={left:595+195*i,top:1180+j*78,width:76,height:76};
    s.images.add({blob:raw,contentType:'image/png',fit:'contain',position,
      alt:`${key} ${kind} original32-square tile, unchanged pixels`});
    pictures.push({...binding,position,original_bytes_supplied:true,raster_resampled_before_embedding:false});
  }
}
text('Retrospective excerpts. All 216 flow streams and 16 diagnostic boards remain available.\nReused development cohort, one model reviewer, no independent human adjudication.',50,1520,1500,72,25);
s.speakerNotes.textFrame.setText('Sources: references/evidence/glyph_context_motor_learning_20261111.json; glyph_context_motor_capacity_learning_20261111.json; decoder_aware_motor_learning_20261111.json; glyph_tail_first_writing_learning_20261111.json; both public128-position pixel archives. Charts a,b compare the unchanged parent and original12000-update direct-context motor flow, not the newer pixel-loss model. All216 initial states match the parent bit-for-bit. Complete strict answers require natural stopping and the frozen reference rule. Proportional gain loses one prior exact answer; pilot requires8exact,2both-correct pairs,11readable and all earlier6proportional/2fixed exact retained. Pilot fails. Passage-group pointwise95percent bootstrap uses10000draws/seed202611154; no untouched confirmation or independent training seed. Charts c,d recount32first and32later authentic training positions for each route. The original flow and paired pixel-loss writer both fail median>=.99 and at least30/32F1>=.95 in each pool. Actual pixel-loss whole-language generation is withheld by that gate and is unmeasured, never zero. Codec receives target images. Full precision remains in bound JSON. Native caches, workbook cells and labels explicitly round F1 to six decimal places. These are model/control comparisons, not a measured learning curve. Four rows retrospectively illustrate recovery, residual damage, native-prefix agreement with damaged writing and a recovered area initial. QA rows are first tiles only. All20PNGobjects preserve32x32original pixel arrays without stroke repair. Both complete128-position archives and16labelled diagnostic boards remain. Native helper and targets are evaluator-only; actual answer path has no lexical decoder, OCR, font, answer bank or reference access. Figure authoring makes no optimizer update. No usable general visual-language or efficiency claim.');
await fs.mkdir(path.join(stage,'output'),{recursive:true});
const candidate=path.join(stage,'candidate.pptx');await(await PresentationFile.exportPptx(deck)).save(candidate);
execFileSync(python,[path.join(root,'publication/ilm-nature/format_motor_chart_labels.py'),candidate]);
const preview=await s.export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'artifact-slide-1.png'),new Uint8Array(await preview.arrayBuffer()));
await fs.writeFile(path.join(stage,'figure_bindings.json'),JSON.stringify({measured_data_sha256:hash(await fs.readFile(path.join(stage,'measured_data.json'))),
  source_sha256:data.source_sha256,charts,pictures,native_charts:4,original_first_tile_images:20,
  full_precision_training_F1:data.physical,chart_F1_decimal_places:6,
  first_tile_selection_retrospective:true,language_score_for_decoded_model:null,
  unmeasured_language_score_not_zero:true,model_comparison_not_learning_curve:true,new_optimizer_updates:0},null,2)+'\n');
await finalizePresentation({workspaceDir:stage,candidatePath:candidate,finalPath:path.join(stage,'output/Motor_diagnostics_figures.pptx'),
  explicitTotalSlideCount:1,requiredNativeChartOwnerSlides:[1],materializeLiteralChartWorkbooks:true,
  fontPolicy:{basis:'reference',families:[font],referencePath:path.join(root,'publication/ilm-nature/figures/Native_reader_writing_figures.pptx'),
    referenceSha256:hash(await fs.readFile(path.join(root,'publication/ilm-nature/figures/Native_reader_writing_figures.pptx')))},
  pythonExecutable:python,integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','15240000,15240000','--validate-heading-fit'],
  verifyArtifactToolImport:true,receiptPath:path.join(stage,'presentation_validation.json')});
