// Update only completed-trial status labels; preserve the original native evidence.
import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {pathToFileURL} from 'node:url';
import {FileBlob, PresentationFile} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
if (![root, stage, skill, python].every(v => path.isAbsolute(v ?? ''))) {
  throw Error('Absolute project, new private stage, skill and runtime Python paths required');
}
if (!path.isAbsolute(process.env.RUNTIME_NODE_MODULES ?? '')) {
  throw Error('Set the bundled RUNTIME_NODE_MODULES before native finalization');
}
const here = path.join(root, 'publication/ilm-nature');
const hash = raw => createHash('sha256').update(raw).digest('hex');
const source = path.join(here, 'figures/Continuation_history_figures.pptx');
const measured = path.join(root, 'references/evidence/visual_continuation_learning_20261113.json');
const sourceHash = hash(await fs.readFile(source));
const measuredHash = hash(await fs.readFile(measured));
if (sourceHash !== 'eeb04bef5e154532c4f8d0eca9a49947ac7208e57b600031fd93575aa1c83862'
    || measuredHash !== '94c4f77615372f37da9454be2e6cf810dd87d96143148644d1b8a54697db68de') {
  throw Error('Preserve the original figure and authenticated completed comparison');
}
const data = JSON.parse(await fs.readFile(measured, 'utf8'));
if (data.training.updates !== 41298 || data.generated_streams !== 732
    || data.original_pilot_passed !== false || data.registered_protocol_pilot_passed !== false) {
  throw Error('Completed failed final trial required');
}
await fs.mkdir(path.join(stage, 'output'), {recursive: true});
const candidate = path.join(stage, 'candidate.pptx');
try { await fs.access(candidate); throw Error('Preserve previous candidate'); }
catch (error) { if (error.code !== 'ENOENT') throw error; }
const replacements = [
  ['ppt/slides/slide1.xml', 'Autonomous language is unmeasured.', 'Full trial failed (Fig. 24).'],
  ['ppt/slides/slide1.xml',
    'Reference history is diagnostic. The ongoing complete fit has no language score yet.',
    'Reference history is diagnostic. The completed language trial fails its final pilot (Figure 24).'],
  ['ppt/notesSlides/notesSlide1.xml',
    'No preflight output or in-progress fit is a new language score.',
    'The preflight establishes feasibility only. The completed full trial fails its final language pilot; see Figure 24 and references/evidence/visual_continuation_learning_20261113.json. The subsequent joint writer remains unmeasured.'],
];
// Surgical XML changes retain every native chart, literal workbook, shape and
// relationship byte. Do not round-trip chart data through an imported model.
execFileSync(python, ['-c', `
import json, sys
from zipfile import ZipFile
v=json.load(sys.stdin)
with ZipFile(v['source']) as original, ZipFile(v['candidate'],'x') as output:
 changed=set()
 for item in original.infolist():
  raw=original.read(item.filename)
  for part,before,after in v['replacements']:
   if item.filename==part:
    assert raw.count(before.encode())==1, (part,before)
    raw=raw.replace(before.encode(),after.encode());changed.add(part)
  output.writestr(item,raw)
 assert changed=={'ppt/slides/slide1.xml','ppt/notesSlides/notesSlide1.xml'}
`], {input: JSON.stringify({source, candidate, replacements}), encoding: 'utf8'});
const {finalizePresentation} = await import(pathToFileURL(
  path.join(skill, 'container_tools/artifact_tool_utils.mjs')).href);
const finalPath = path.join(stage, 'output/Continuation_history_completed_figures.pptx');
await finalizePresentation({
  workspaceDir: stage, candidatePath: candidate, finalPath, pythonExecutable: python,
  integrityValidatorPath: path.join(skill, 'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath: path.join(skill, 'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs: ['--expected-slide-size-emu', '15240000,15240000',
    '--validate-bullet-geometry', '--validate-heading-fit', '--require-native-table-slide', '1'],
  explicitTotalSlideCount: 1, requiredNativeTableOwnerSlides: [1],
  requiredNativeChartOwnerSlides: [1], requiredEmbeddedWorkbookChartOwnerSlides: [1],
  materializeLiteralChartWorkbooks: false, verifyArtifactToolImport: true,
  fontPolicy: {basis: 'reference', families: ['Arial'], referencePath: source, referenceSha256: sourceHash},
  receiptPath: path.join(stage, 'presentation_validation.json'),
});
const presentation = await PresentationFile.importPptx(await FileBlob.load(finalPath));
const snapshot = await presentation.inspect({kind: 'textbox,shape,chart,table,notes', maxChars: 100000});
for (const [, before] of replacements) {
  if (snapshot.ndjson.includes(before)) throw Error('Obsolete status survived final-file import');
}
const slide = presentation.slides.items[0];
const image = await slide.export({format: 'png', scale: 1});
await fs.writeFile(path.join(stage, 'final-reimport.png'), new Uint8Array(await image.arrayBuffer()));
await fs.writeFile(path.join(stage, 'final-reimport.layout.json'), await (await slide.export({format: 'layout'})).text());
await fs.writeFile(path.join(stage, 'status_bindings.json'), JSON.stringify({
  original_pptx_sha256: sourceHash, completed_result_sha256: measuredHash,
  editor_sha256: hash(await fs.readFile(path.join(here, 'update_continuation_history_status.mjs'))),
  final_pptx_sha256: hash(await fs.readFile(finalPath)),
  final_reimport_sha256: hash(await fs.readFile(path.join(stage, 'final-reimport.png'))),
  replacements, new_model_calls: 0, actual_visual_inspection_pending: true,
  Office_export_verified: false, article_activation_pending: true,
}, null, 2) + '\n');
console.log(JSON.stringify({finalized: true, actual_visual_inspection_pending: true, new_model_calls: 0}));
