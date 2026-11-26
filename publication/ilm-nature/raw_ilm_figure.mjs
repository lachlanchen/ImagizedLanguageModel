import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { Presentation, PresentationFile } from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const { finalizePresentation, applyPresentationChartFont } = await import(
  pathToFileURL(path.join(skill, 'container_tools/artifact_tool_utils.mjs'))
);
const evidence = path.join(root, 'references/evidence/paired_ink_v1');
const raw = path.join(root, 'publication/ilm-nature/figures/raw_ilm');
const read = async name => JSON.parse(await fs.readFile(path.join(evidence, name)));
const deck = Presentation.create({ slideSize: {width:1060, height:652} });
function newSlide() {
  const slide = deck.slides.add();
  slide.background.fill = '#FFFFFF';
  return slide;
}
function label(slide, text, x, y, width, size, bold=false, color='#172927') {
  const box = slide.shapes.add({geometry:'textbox',
    position:{left:x, top:y, width, height:size*1.5}, fill:'none', line:{fill:'none',width:0}});
  box.text = text;
  box.text.style = {typeface:'Arial', fontSize:size, bold, color, autoFit:'none',
    insets:{left:0, right:0, top:0, bottom:0}};
}
async function picture(slide, name, left, top, width, height) {
  slide.images.add({blob:await fs.readFile(path.join(raw,name)), contentType:'image/png',
    alt:name, fit:'contain', position:{left,top,width,height}});
}
const cover = newSlide();
label(cover,'Who invited whom?',38,38,970,40,true);
label(cover,'Two pictured requests put the same people in opposite roles.',38,110,970,26);
for (const [i,id] of [54,55].entries()) {
  const y = 196 + i * 192;
  label(cover,'English input',40,y,190,23,true);
  await picture(cover,'role-'+id+'-input.png',238,y,800,50);
  label(cover,'Chinese output',40,y+83,190,23,true);
  await picture(cover,'role-'+id+'-output.png',238,y+83,200,25);
}
const rawSlide = newSlide();
for (const p of (await read('scene.json')).items) {
  if (p.type === 'text') label(rawSlide,p.text,2*p.x,2*p.y,1060-2*p.x,2*p.size,p.bold,p.color);
  else await picture(rawSlide,p.name,2*p.x,2*p.y,2*p.w,2*p.h);
}
rawSlide.speakerNotes.textFrame.setText('Full source and emitted bitmaps. The six individual cells use supplied visual history and come from a separate pixel writer.');
const results = newSlide();
label(results,'Reading order and free translation',30,34,1000,38,true);
const chartData = await read('charts.json');
chartData.panels.forEach((panel,i) => {
  const left = 22 + 528*i;
  label(results,panel.title,left,142,504,23,true);
  label(results,panel.den,left,192,504,19);
  const chart = results.charts.add('bar',{position:{left,top:242,width:494,height:327},
    categories:panel.labels,series:Object.entries(panel.series).map(([name,values]) =>
      ({name,values,fill:chartData.colors[name]})),
    hasLegend:true, legend:{position:'bottom',textStyle:{fontSize:12}},
    barOptions:{direction:'column',grouping:'clustered'},
    xAxis:{textStyle:{fontSize:13}}, yAxis:{min:0,max:96,majorUnit:24,textStyle:{fontSize:13}}});
  applyPresentationChartFont(chart,{fontFamily:'Arial'});
});
results.speakerNotes.textFrame.setText('48 bilingual test families in both directions. Meaning required acceptance in both grading orders. Counts and original charts are in the accompanying evidence.');
for (const folder of ['draft','output']) await fs.mkdir(path.join(stage,folder),{recursive:true});
const candidatePath=path.join(stage,'draft/raw_ilm.pptx');
const finalPath=path.join(stage,'output/Raw_ILM_images.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,
  explicitTotalSlideCount:3,materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[3],
  verifyArtifactToolImport:true,integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','10096500,6210300'],receiptPath:path.join(stage,'presentation_validation.json')});
