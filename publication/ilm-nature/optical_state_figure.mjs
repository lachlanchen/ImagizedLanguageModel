import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation,PresentationFile} from '@oai/artifact-tool';
const [root,stage,skill,python]=process.argv.slice(2);
const {finalizePresentation,applyPresentationChartFont}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')));
const deck=Presentation.create({slideSize:{width:1060,height:652}});
const evidence=path.join(root,'references/evidence/optical_state_v1');
function label(slide,s,x,y,w,size,bold=false){
  const b=slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:size*1.65},fill:'none',line:{fill:'none',width:0}});
  b.text=s;b.text.style={typeface:'Arial',fontSize:size,bold,color:'#213634',autoFit:'none',insets:{left:0,right:0,top:0,bottom:0}};
}
for (const [stem,title] of [['Figure_213_glyph_attention_measure','One word across several glyph positions'],
                            ['Figure_214_optical_state_reader','Learning a bilingual representation from ink']]){
  const {panels,colors}=JSON.parse(await fs.readFile(path.join(evidence,stem+'.json')));
  const slide=deck.slides.add();slide.background.fill='#ffffff';
  label(slide,title,26,22,1010,32,true);
  for(const [i,p] of panels.entries()){
    const x=20+526*i;
    label(slide,p.title,x,105,512,18,true);
    label(slide,p.unit,x,149,512,15);
    const chart=slide.charts.add(p.kind==='line'?'line':'bar',{
      position:{left:x,top:195,width:506,height:390},categories:p.labels,
      series:Object.entries(p.series).map(([name,values])=>({name,values,fill:colors[name],line:{fill:colors[name],width:2.5}})),
      hasLegend:true,legend:{position:'bottom',textStyle:{fontSize:13}},
      xAxis:{textStyle:{fontSize:13}},yAxis:{min:0,max:p.upper},
      ...(p.kind==='line'?{}:{barOptions:{direction:'column',grouping:'clustered'}})
    });
    applyPresentationChartFont(chart,{fontFamily:'Arial'});
  }
}
for(const d of ['draft','output'])await fs.mkdir(path.join(stage,d),{recursive:true});
const candidatePath=path.join(stage,'draft/measure.pptx'),finalPath=path.join(stage,'output/Glyph_attention_and_reading.pptx');
await(await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,explicitTotalSlideCount:2,
  materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1,2],verifyArtifactToolImport:true,
  integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','10096500,6210300'],receiptPath:path.join(stage,'presentation_validation.json')});
