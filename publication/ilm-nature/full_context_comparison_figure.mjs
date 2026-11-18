import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { pathToFileURL } from 'node:url';
import { Presentation, PresentationFile, FileBlob } from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {applyPresentationChartFont, finalizePresentation} = await import(
  pathToFileURL(path.join(skill, 'container_tools/artifact_tool_utils.mjs')).href);
const digest = bytes => createHash('sha256').update(bytes).digest('hex');
const source = 'references/evidence/trainable_visual_context_result_20261117.json';
const evidence = JSON.parse(await fs.readFile(path.join(root, source)));
const review = 'artifacts/trainable_visual_context_comparison_v1/review';
const mapPath = review + '/mapping.json';
const mapBytes = await fs.readFile(path.join(root, mapPath));
if (digest(mapBytes) !== evidence.bindings[mapPath]) throw Error('Archived mapping changed');
const mapping = JSON.parse(mapBytes).rows;
const deck = Presentation.create({slideSize: {width:1600, height:1550}});
const slide = deck.slides.add();
slide.background.fill = '#FFFFFF';
const font = 'Arial', navy = '#193849', teal = '#197478', rust = '#AA503B';
const arms = ['trainable','frozen'];
const names = ['Learned context','Frozen context'];
function label(value, left, top, width, height, fontSize, bold=false) {
  const shape = slide.shapes.add({geometry:'textbox', position:{left,top,width,height},
    fill:'none',line:{fill:'none',width:0}});
  shape.text = value;
  shape.text.style = {typeface:font,fontSize,bold,color:navy,autoFit:'none'};
}
label('Question answering and book continuation after contextual training',48,18,1504,70,41,true);
label('a   Strict answers across training',48,117,1504,52,34,true);
const trajectory = arms.map(a => evidence.all_snapshot_summaries.filter(s=>s.arm===a));
const categories = trajectory[0].map(s=>s.step.toLocaleString('en-US'));
const chart = slide.charts.add('line',{
  position:{left:56,top:179,width:860,height:335},title:'Correct answers per 32 questions',
  categories,series:trajectory.map((values,i)=>({name:names[i],values:values.map(s=>s.exact),
    line:{fill:i===0?teal:rust,width:i===0?7:2}})),hasLegend:false,
  xAxis:{textStyle:{fontSize:23,typeface:font}},yAxis:{min:0,max:32,majorUnit:8,
    textStyle:{fontSize:23,typeface:font},majorGridlines:{fill:'#D9E1E4',width:1}}
});
applyPresentationChartFont(chart,{fontFamily:font});
label('Both curves coincide at zero across all five checkpoints.',964,221,572,100,30,true);
label('At every checkpoint, both models also scored 0/16 on paired questions and 0/32 on readable complete answers.',964,334,574,134,26);
label('Optimizer updates',330,519,450,37,24);
label('b   Final continuations from 12 normal book prefixes',48,600,1504,54,34,true);
const rows = [['Model','Natural stops','Reached 640 glyphs','Coherent continuations'],
  ...arms.map((a,i)=>{const r=evidence.arms[a].final_summary;return [names[i],String(r.long_book_stops),
    String(r.long_book_caps),String(r.coherent_complete_long_books)];})];
const table = slide.tables.add({rows:3,columns:4,left:54,top:680,width:1484,height:182,
  columnWidths:[345,300,383,456],values:rows});
table.styleOptions = {headerRow:true,bandedRows:false};
table.borders.assign({style:'solid',fill:'#DBE3E6',width:1});
for(let r=0;r<3;r++) for(let c=0;c<4;c++) {
  const cell=table.getCell(r,c);
  cell.fill=r===0?'#E9F1F2':'#FFFFFF';
  cell.text.style={typeface:font,fontSize:27,bold:r===0,color:navy};
}
label('c   Complete output images for two questions about the same passage',48,915,1510,55,34,true);
const samples=[];
for (const [j,slot] of ['a','b'].entries()) {
  const x=55+759*j;
  label('Question '+slot.toUpperCase(),x,989,690,50,29,true);
  for(const [i,a] of arms.entries()){
    const matches=Object.entries(mapping).filter(([,r])=>r.case==='qa_01_'+slot+'_normal'
      &&r.step===8172&&r.profile==='all'&&r.arm===a);
    if(matches.length!==1) throw Error('Missing prespecified complete output');
    const [id,r]=matches[0], p=path.join(root,review,'pages',id+'.png');
    const bytes=await fs.readFile(p);
    if(digest(bytes)!==r.image_sha256) throw Error('Original pixels changed');
    const w=bytes.readUInt32BE(16), h=bytes.readUInt32BE(20);
    const y=1056+i*166;
    label(names[i]+'   '+r.emitted_glyphs+' glyphs, '+(r.stopped?'stopped':'capped'),x,y,710,40,24);
    slide.images.add({blob:bytes,contentType:'image/png',fit:'contain',alt:names[i]+' question '+slot,
      position:{left:x,top:y+50,width:704,height:704*h/w}});
    samples.push({arm:a,case:r.case,png_sha256:r.image_sha256,emitted:r.emitted_glyphs,width:w,height:h});
  }
}
label('First paired passage in evaluation order. All four outputs received uncertain readings.',50,1469,1506,62,26);
slide.speakerNotes.textFrame.setText(source+' | Opaque review of 792 streams. One model reviewer had prior familiarity with this development cohort. Original answer images retain all emitted pixels.');
await fs.mkdir(path.join(stage,'draft'),{recursive:true});
await fs.mkdir(path.join(stage,'output'),{recursive:true});
const candidatePath=path.join(stage,'draft/comparison.pptx');
const finalPath=path.join(stage,'output/Full_context_comparison.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
await finalizePresentation({workspaceDir:stage,candidatePath,finalPath,pythonExecutable:python,
  explicitTotalSlideCount:1,materializeLiteralChartWorkbooks:true,requiredNativeChartOwnerSlides:[1],requiredNativeTableOwnerSlides:[1],verifyArtifactToolImport:true,
  integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','15240000,14763750','--validate-heading-fit','--require-native-table-slide','1'],
  receiptPath:path.join(stage,'presentation_validation.json')});
const final=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const preview=await final.slides.items[0].export({format:'png',scale:1});
await fs.writeFile(path.join(stage,'Figure_32_full_context_comparison.png'),new Uint8Array(await preview.arrayBuffer()));
await fs.writeFile(path.join(stage,'samples.json'),JSON.stringify(samples,null,2));
