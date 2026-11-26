import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile} from '@oai/artifact-tool';
const [root,stage,skill,python,edition]=process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')));
const deck=Presentation.create({slideSize:{width:1280,height:720}});
const pub=path.join(root,'references/evidence/pictured_path_translation_v1');
const read=async x=>JSON.parse(await fs.readFile(path.join(pub,x)));
function label(s,text,x,y,w,size,bold=false,color='#19332f'){
 const shape=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:size*1.48},fill:'none',line:{fill:'none',width:0}});
 shape.text=text;
 shape.text.style={typeface:'Arial',fontSize:size,bold,color,autoFit:'none',insets:{left:0,right:0,top:0,bottom:0}};
}
function slide(heading){
 const s=deck.slides.add();s.background.fill='#ffffff';label(s,heading,36,22,1208,32,true);return s;
}
async function plate(name,title,notes,assetFolder="pictured_path"){
 const v=await read(name),s=slide(title);
 const sc=Math.min(1170/v.width,586/v.height);
 const x0=42,y0=95;
 for(const i of v.items){
  const y=y0+i.y*sc;
  if(i.type==='heading'){
   label(s,i.text,x0,y,sc*300,11*sc,true);
   label(s,i.meta,x0+sc*300,y+sc,sc*192,7.8*sc,false,'#49615b');
  }else{
   label(s,i.label,x0+8*sc,y+sc,sc*87,8*sc);
   s.images.add({blob:await fs.readFile(path.join(root,'publication/ilm-nature/figures',assetFolder,i.name)),contentType:'image/png',alt:i.name,fit:'contain',position:{left:x0+i.x*sc,top:y,width:i.w*sc,height:i.h*sc}});
  }
 }
 s.speakerNotes.textFrame.setText(notes);
}
const breadthEdition = edition === 'breadth';
if (breadthEdition) {
 await plate('../visual_training_breadth_v1/plate.json','Writing after exposure to more distinct translations','First credited English broad answer, first broad gain against the repeated arm or first changed answer, and next uncredited output in the opposite direction. All pretrained outputs use the same pictured sources.', 'visual_training_breadth');
 const data=await read('../visual_training_breadth_v1/curve.json');
 const s=slide('Development translations over six passes');
 const colors=['#126b62','#975733','#3476ae','#705286'];
 const chart=s.charts.add('line',{position:{left:60,top:88,width:1150,height:535},categories:data.labels.map(String),lineOptions:{smooth:false},
  yAxis:{visible:true,min:0,max:.5,majorUnit:.1,numberFormatCode:'0.0',tickLabelPosition:'nextTo'},
  series:Object.entries(data.series).map(([name,values],i)=>({name,values:values.map(v=>Number(v.toFixed(12))),line:{fill:colors[i],width:3}})),
  hasLegend:true,legend:{position:'bottom'}});
 applyPresentationChartFont(chart,{fontFamily:'Arial'});
 label(s,'Mean glyph 1–4-gram F2 on 72 translations per direction. Pass zero is the shared starting writer.',65,636,1170,18);
 s.speakerNotes.textFrame.setText('Each retained training pass has the same number of supervised positions. Complete meaning was graded on fresh books after the final pass.');
}
const retrievedEdition = edition === 'retrieved' || breadthEdition;
const sequenceEdition = edition === 'sequence-preference' || retrievedEdition;
const editorEdition = edition === 'editor' || sequenceEdition;
if (sequenceEdition) {
 await plate('../visual_sequence_preference_v1/plate.json','Learning a correction and writing on a new page','First repeated training example credited only after preference learning, followed by original held-out source and answer PNGs. First preference-only gain relative to likelihood or the first differing output when no such gain occurs. Second example is the subsequent uncredited answer in the opposite direction.', 'visual_sequence_preference');
}
const phraseEdition = edition === 'phrases' || editorEdition;
if (retrievedEdition) {
 await plate('../retrieved_glyph_writer_v1/plate.json','Writing with related bilingual passages','First gain over the parent if present, otherwise the first changed answer. Second example is the next uncredited answer in the opposite direction. Images are original PNGs.', 'retrieved_glyphs');
}
if (editorEdition && !sequenceEdition) {
 await plate('../glyph_draft_editor_v1/plate.json','Can a visual writer use a phrase draft?','First credited editor answer over the matched writer, followed by the first uncredited editor answer in the opposite direction. All rows reproduce the archived PNGs.', 'glyph_draft_editor');
}
const alignmentEdition = edition === 'alignment' || phraseEdition;
if (phraseEdition && !editorEdition) {
 await plate('../covered_phrases_v1/plate.json','Source coverage and complete meaning','First book answer credited only to the neural writer relative to covered phrases. First negative object cleft exact for the neural writer and missed by the phrase writer. Emitted PNGs are unaltered.', 'covered_phrases');
}
const fullImageEdition = edition === 'full-image' || alignmentEdition;
if (alignmentEdition && !phraseEdition) {
 await plate('../ordered_alignment_v1/plate.json','What source order changes in visual translation','Original evaluation images. Top: the first object cleft made exact by source movement. Bottom: the first fresh English-to-Chinese book answer changed by movement. Both conditions use the same physical repertoire.', 'ordered_alignment');
}
const referentEdition = edition === 'referents' || fullImageEdition;
if (fullImageEdition && !alignmentEdition) {
 await plate('../full_image_roles_v1/plate.json','Writing from a complete pictured sentence',
   'Controlled polarity contrast and an untrained book passage. Every pictured source and response is an original evaluation image. The language writer chose its own output length.', 'full_image_roles');
}
const roleEdition = edition === 'roles' || referentEdition;
if (referentEdition && !fullImageEdition) {
 await plate('../visual_referents_v1/plate.json','Can a pictured reader resolve a relative expression?',
   'First newly correct controlled answer and first external failure. The English event strips are selected source cells in a fixed print face. Both Chinese answers were emitted by the frozen writer.', 'visual_referents');
}
if (roleEdition && !referentEdition) {
 await plate('../natural_pictured_roles_v1/plate.json','Finding the participants in a pictured clause',
   'Original pictured test sentences and both crops emitted by the model. Only the predicate was marked. Cases selected by the fixed display rule.', 'natural_roles');
}
const restorationEdition = edition === 'restoration' || roleEdition;
const supportEdition = edition === 'support' || restorationEdition;
if (restorationEdition) {
 await plate('../visual_strip_restoration_v1/Free_strip_restoration.json','Can a visual writer supply the words of a translation?',
   'First Chinese-directed request in the reserved book panel. The source alone determines the number and content of output strips.', 'strip_restoration');
 await plate('../visual_strip_restoration_v1/Supplied_strip_restoration.json','Can the source resolve order when the content is given?',
   'The supplied multiset is identical across the role reversal. First both-exact pair in fixed order if available, otherwise the first pair.', 'strip_restoration');
}
const actionsEdition = edition === 'actions' || supportEdition;
if(supportEdition) {
 await plate('../source_supported_writing_v1/plate.json','Writing with source-predicted content',
   'Two fixed-order fresh book cases; additional whole-answer gain if available. All answer strips are original pixels.','source_supported');
}
const draftEdition = edition === 'drafts' || actionsEdition;
if(actionsEdition) {
 await plate('../independent_optical_actions_v1/plate.json','Free writing with single glyphs or learned strips',
   'First pair of the first two held books and the first credited strip response, selected after scoring. All images are the original archived bitmaps.','independent_optical_actions');
}
if(draftEdition) {
 await plate('../emitted_history_v1/plate.json','Can a writer recover meaning from its own translation drafts?',
   'Raw source and output bitmaps from the held book panel. The control and draft-supervised writer share the same starting weights and training exposures. Selection is fixed by corpus order.',
   'sentence_reconstruction');
}
await plate('answers.json','Reading the same source through two visual interfaces','The first newly credited answer in each direction and the first uncredited answer under the frozen selection rule. All images come from the model archive.');
{
 const s=slide('How often did each writer preserve the whole meaning?');
 const data=await read('chart.json');
 const chart=s.charts.add('bar',{position:{left:55,top:85,width:1170,height:525},categories:data.labels,
   series:[{name:'To English',values:data.series.en,fill:'#136b62'},{name:'To Chinese',values:data.series.zh,fill:'#966442'}],
   dataLabels:{showValue:true,position:'outEnd',textStyle:{typeface:'Arial',fontSize:17,fill:'#19332f'}},
   hasLegend:true,legend:{position:'bottom'},barOptions:{direction:'column',grouping:'clustered'}});
 applyPresentationChartFont(chart,{fontFamily:'Arial'});
 label(s,'Complete meanings among 96 requests in each direction. Both masked readings and a natural stop were required.',75,630,1160,18);
 s.speakerNotes.textFrame.setText('The two optical readers share their target writer. Supplied input paths use native source text addresses. The Qwen comparator receives text.');
}
await plate('reading.json','Even an exact source sequence can yield the wrong answer','A translation rejected in both-order scoring despite exact recovery of its source addresses. The two complete visual replies can be compared pixel by pixel.');
await plate('origins.json','Can a small reader answer from a historical dictionary page?','The first held request and first reworded failure in fixed order. The writer selects and copies original source pixels. Both the gloss and the form were supplied on the page.');
if(draftEdition) await plate('../visual_order_v1/plate.json','How does the requested order change a historical answer?',
 'Both pictured questions refer to the same labelled entries. The two models share the same selected values and stop. All four answers preserve the original source pixels.',
 'visual_order');
if(actionsEdition) await plate('../independent_optical_actions_v1/other-works.json','Free writing in the other two held books',
 'Same preselected first-pair rule. Images reproduce the exact raster answers from the physical motor.','independent_optical_actions');
if(supportEdition) await plate('../source_supported_writing_v1/opus-plate.json','Free translation within the OPUS test domain',
 'First credited aligned response in fixed order if available, otherwise the first request. Compare the unchanged parent and guided output.','source_supported');
if (breadthEdition) await plate('../visual_training_breadth_v1/phase-plate.json','When the same pixels have two embeddings','First credit lost by phase-conditioned feedback relative to original pooled feedback. Source and all outputs are archived PNGs.', 'visual_training_breadth');
for(const name of ['draft','output']) await fs.mkdir(path.join(stage,name),{recursive:true});
const candidatePath=path.join(stage,'draft/figures.pptx'),finalPath=path.join(stage,'output',breadthEdition?'Visual_training_breadth.pptx':retrievedEdition?'Retrieved_glyphs.pptx':sequenceEdition?'Visual_sequence_preference.pptx':editorEdition?'Glyph_draft_editor.pptx':phraseEdition?'Covered_glyph_phrases.pptx':alignmentEdition?'Ordered_glyph_alignment.pptx':fullImageEdition?'Full_image_role_writing.pptx':referentEdition?'Visual_referent_answers.pptx':roleEdition?'Visual_role_reading.pptx':restorationEdition?'Visual_strip_restoration.pptx':supportEdition?'Source_supported_writing.pptx':actionsEdition?'Independent_optical_actions.pptx':draftEdition?'Sentence_reconstruction.pptx':'Pictured_source_paths.pptx');
await(await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:breadthEdition?17:retrievedEdition?14:roleEdition?13:restorationEdition?12:supportEdition?10:actionsEdition?8:draftEdition?6:4,
 materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:breadthEdition?[2,11]:[retrievedEdition?9:roleEdition?8:restorationEdition?7:supportEdition?5:actionsEdition?4:draftEdition?3:2],verifyArtifactToolImport:true,
 integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000'],receiptPath:path.join(stage,'presentation_validation.json')});
