import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';
import sharp from 'sharp';
import {Presentation,PresentationFile} from '@oai/artifact-tool';

const [root,stage,skill,python]=process.argv.slice(2);
const {applyPresentationChartFont,finalizePresentation}=await import(pathToFileURL(
  path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const hash=x=>createHash('sha256').update(x).digest('hex');
const data=JSON.parse(await fs.readFile(path.join(stage,'measured_data.json'),'utf8'));
const deck=Presentation.create({slideSize:{width:1600,height:1600}});
const s=deck.slides.add();s.background.fill='#FFFFFF';
const font='Arial',ink='#192936',muted='#43535C',blue='#386985',brown='#9B7457',grey='#89969D',grid='#DDE4E8';
const charts=[],pictures=[];
function text(value,x,y,w,h,size=27,bold=false){
  const z=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},
    fill:'none',line:{fill:'none',width:0}});z.text=value;
  z.text.style={typeface:font,fontSize:size,bold,color:bold?ink:muted,autoFit:'none'};
}
const axis={textStyle:{typeface:font,fontSize:24,fill:muted},line:{fill:muted,width:1}};
const titleTextStyle={typeface:font,fontSize:30,bold:true,fill:ink};
function chart(config){
  const c=s.charts.add('bar',config);applyPresentationChartFont(c,{fontFamily:font});
  charts.push({categories:config.categories,series:config.series.map(x=>({name:x.name,values:x.values})),position:config.position});
}
text('Native answer access and first-image writing',48,28,1504,70,46,true);
const layouts=['fixed_cells','proportional'];
for(const [i,[field,title,max]] of [['exact','Complete exact QA (32)',32],['pairs','Both correct (16 pairs)',16]].entries()){
  text('ab'[i],42+i*790,128,40,50,34,true);
  chart({position:{left:75+i*790,top:130,width:710,height:430},title,titlePlacement:'aboveChart',titleTextStyle,
    categories:['Fixed cells','Proportional'],series:[
      {name:'Native adapted',values:layouts.map(l=>data.native.adapted[l][field]),fill:blue},
      {name:'Native source',values:layouts.map(l=>data.native.source_zero_output[l][field]),fill:brown},
      {name:'Glyph adapted',values:layouts.map(l=>data.glyph[l][field]),fill:grey}],
    barOptions:{direction:'column',grouping:'clustered',gapWidth:100},hasLegend:true,
    legend:{position:'bottom',overlay:false,textStyle:{fontSize:22,fill:muted}},xAxis:axis,
    yAxis:{...axis,min:0,max,majorUnit:max/4,numberFormatCode:'0',majorGridlines:{fill:grid,width:1}},
    dataLabels:{showValue:true,position:'outEnd',textStyle:{typeface:font,fontSize:29,fill:ink}}});
}
text('Native decoding is an external diagnostic. All 192 adapted image prefills match actual glyph states.\nProportional native minus glyph exact QA: +12.5 points (95% interval 0 to 25).',80,580,1470,88,28);
text('c',42,700,40,50,34,true);
chart({position:{left:80,top:700,width:700,height:335},title:'Mean first-tile foreground F1',titlePlacement:'aboveChart',titleTextStyle,
  categories:['Supplied AE\n0.999934','Fixed original\n0.492173','Proportional\n0.632918'],
  series:[{name:'Canonical pixel agreement',values:data.mean_first_F1,fill:blue}],
  barOptions:{direction:'column',grouping:'clustered',gapWidth:95},hasLegend:false,xAxis:axis,
  yAxis:{...axis,min:0,max:1,majorUnit:.25,numberFormatCode:'0.00',majorGridlines:{fill:grid,width:1}},
  dataLabels:{showValue:false}});
text('d',832,700,40,50,34,true);
chart({position:{left:870,top:700,width:680,height:335},title:'Proportional strict outcomes (32)',titlePlacement:'aboveChart',titleTextStyle,
  categories:Object.keys(data.proportional_outcomes),series:[{name:'Cases',values:Object.values(data.proportional_outcomes),fill:brown}],
  barOptions:{direction:'column',grouping:'clustered',gapWidth:85},hasLegend:false,xAxis:{...axis,textStyle:{...axis.textStyle,fontSize:23}},
  yAxis:{...axis,min:0,max:32,majorUnit:8,numberFormatCode:'0',majorGridlines:{fill:grid,width:1}},
  dataLabels:{showValue:true,position:'outEnd',textStyle:{typeface:font,fontSize:29,fill:ink}}});
text('Supplied images reveal physical support, not language.\nFirst pixel-exact: 31/32 supplied, 0/32 fixed, 3/32 proportional.',80,1042,710,75,25);
text('Native-only: six uncertain glyph readings and one alias omission.\nGlyph-only: native wording or punctuation differs from the reference.',870,1042,680,75,25);
text('e   Original first tiles',48,1140,750,55,32,true);
for(const [x,title] of [[500,'Canonical'],[760,'Supplied AE'],[1020,'Fixed original'],[1280,'Proportional']])text(title,x-35,1200,235,45,26,true);
for(const [j,[key,title]] of [['qa_09_a','Area answer: damaged form'],['qa_05_b','Exact first; alias omitted'],['qa_13_a','Different answer prefix']].entries()){
  text(title,60,1250+j*87,400,65,26);
  for(const [i,kind] of ['canonical','supplied','fixed','proportional'].entries()){
    const binding=data.pictures.find(p=>p.case===key&&p.kind===kind);
    const raw=await fs.readFile(path.join(stage,binding.file));
    if(hash(raw)!==binding.sha256)throw Error('Original first tile changed');
    const pixels=await sharp(raw).removeAlpha().greyscale().raw().toBuffer();
    if(hash(pixels)!==binding.gray_pixels_sha256)throw Error('Original pixel array changed');
    const position={left:510+260*i,top:1238+j*87,width:80,height:80};
    s.images.add({blob:raw,contentType:'image/png',fit:'contain',position,
      alt:`${key} ${kind} original first32-square tile; exact pixels; no stroke repair`});
    pictures.push({...binding,position,original_bytes_supplied:true,raster_resampled_before_embedding:false});
  }
}
text('First-tile excerpts selected retrospectively. All 32 comparisons and 192 original streams remain available.\nPixel differences do not grade meaning. One model reviewer; no independent human adjudication.',50,1520,1500,72,25);
s.speakerNotes.textFrame.setText('Sources: references/evidence/glyph_tail_native_reader_learning_20261111.json; glyph_tail_first_writing_learning_20261111.json; glyph_tail_first_writing_registration_20261111.json; glyph_tail_first_writing_20261111/pixels.npz. Four hundred forty-eight helper calls, including image and complete-text controls; all192 adapted image prefills reproduce actual glyph-generation states bit-for-bit. Native source disables contextual adapter outputs. Native decoder tokenizer, lexical projection and previous-symbol embeddings remain outside the image-output actor. Native source-unit cap128 and image glyph cap128 have different units. Image-erasure arms score0/32 exact throughout. Text native exact scores11/32 adapted and10/32 source. Mixed adapter effects do not justify replacing the deployed body. Native-glyph contrast is a reused-development pointwise group-bootstrap interval with10000draws/seed202611154, not an untouched or independent-seed result. Physical F1 is per-image foreground mean against canonical first tiles, not character or semantic accuracy. All32 first forms have primitive coverage; supplied reconstruction31/32pixel-exact. Full references743/745covered; two later absent forms remain in denominators. Examples qa09a,qa05b,qa13a selected retrospectively after completed diagnostics to expose early damage, alias omission despite an exact first image and a different answer prefix. These are explicitly first-tile excerpts, not complete answers. No pixels repaired, cropped or resampled before embedding. The12originalPNGtileobjects remain editable. No optimizer update, new ILM image or usable-language claim follows from these controls.');
await fs.mkdir(path.join(stage,'output'),{recursive:true});
const candidate=path.join(stage,'candidate.pptx');await(await PresentationFile.exportPptx(deck)).save(candidate);
const preview=await s.export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'artifact-slide-1.png'),new Uint8Array(await preview.arrayBuffer()));
await fs.writeFile(path.join(stage,'figure_bindings.json'),JSON.stringify({measured_data_sha256:hash(await fs.readFile(path.join(stage,'measured_data.json'))),
  source_sha256:data.source_sha256,charts,pictures,native_charts:4,original_first_tile_images:12,
  first_tile_selection_retrospective:true,first_pixel_error_not_semantic_accuracy:true,new_optimizer_updates:0},null,2)+'\n');
await finalizePresentation({workspaceDir:stage,candidatePath:candidate,finalPath:path.join(stage,'output/Native_reader_writing_figures.pptx'),
  explicitTotalSlideCount:1,requiredNativeChartOwnerSlides:[1],materializeLiteralChartWorkbooks:true,
  fontPolicy:{basis:'reference',families:[font],referencePath:path.join(root,'publication/ilm-nature/figures/Glyph_input_transfer_figures.pptx'),
    referenceSha256:hash(await fs.readFile(path.join(root,'publication/ilm-nature/figures/Glyph_input_transfer_figures.pptx')))},
  pythonExecutable:python,integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','15240000,15240000','--validate-heading-fit'],
  verifyArtifactToolImport:true,receiptPath:path.join(stage,'presentation_validation.json')});
