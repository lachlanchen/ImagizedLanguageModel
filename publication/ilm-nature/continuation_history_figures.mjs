import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {Presentation, PresentationFile} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
if (![root, stage, skill, python].every(value => path.isAbsolute(value ?? ''))) {
  throw Error('Absolute project, private stage, skill and runtime Python paths required');
}
const {applyPresentationChartFont, finalizePresentation} = await import(pathToFileURL(
  path.join(skill, 'container_tools/artifact_tool_utils.mjs')).href);
const hash = raw => createHash('sha256').update(raw).digest('hex');
const source = path.join(root, 'publication/ilm-nature/evidence/continuation_history_measured_data.json');
const raw = await fs.readFile(source); const data = JSON.parse(raw);
if (!data.all_literal_position_metrics_recounted || data.new_model_calls !== 0
    || !data.original_complete_language_scores_unchanged || !data.full_fit.language_quality_unmeasured
    || Object.values(data.later_physical_gates).some(Boolean)) throw Error('Completed diagnostic evidence required');
for (const [name, digest] of Object.entries(data.source_sha256)) {
  if (hash(await fs.readFile(path.join(root, name))) !== digest) throw Error('Bound evidence changed: ' + name);
}
await fs.mkdir(stage, {recursive: true});
await fs.mkdir(path.join(stage, 'output'), {recursive: true});
const deck = Presentation.create({slideSize: {width: 1600, height: 1600}});
const s = deck.slides.add(); s.background.fill = '#FFFFFF';
const font = 'Arial', ink = '#192936', muted = '#43535C', blue = '#386985';
const brown = '#9B7457', green = '#527B70', grid = '#DDE4E8';
const charts = [], nodes = [], links = [];
function text(value, x, y, width, height, size = 26, bold = false) {
  const shape = s.shapes.add({geometry: 'textbox',
    position: {left: x, top: y, width, height}, fill: 'none', line: {fill: 'none', width: 0}});
  shape.text = value;
  shape.text.style = {typeface: font, fontSize: size, bold, color: bold ? ink : muted, autoFit: 'none'};
  return shape;
}
const axis = {textStyle: {typeface: font, fontSize: 24, fill: muted}, line: {fill: muted, width: 1}};
const titleTextStyle = {typeface: font, fontSize: 29, bold: true, fill: ink};
const legend = {position: 'bottom', overlay: false, textStyle: {fontSize: 23, fill: muted}};
const categories = ['Fixed cells', 'Proportional'];
const layouts = ['fixed_cells', 'proportional'];
const rounded = value => Number(value.toFixed(6));
function chart(title, x, series) {
  const config = {position: {left: x, top: 140, width: 700, height: 475},
    title, titlePlacement: 'aboveChart', titleTextStyle, categories,
    series: series.map(series => ({...series, valuesFormatCode: '0.000000'})),
    barOptions: {direction: 'column', grouping: 'clustered', gapWidth: 90},
    hasLegend: true, legend, xAxis: axis,
    yAxis: {...axis, min: 0, max: 1.1, majorUnit: .25, numberFormatCode: '0.00',
      majorGridlines: {fill: grid, width: 1}},
    dataLabels: {showValue: true, position: 'outEnd', textStyle: {typeface: font, fontSize: 23, fill: ink}}};
  const item = s.charts.add('bar', config); applyPresentationChartFont(item, {fontFamily: font});
  charts.push({title, categories, series: config.series.map(({name, values}) => ({name, values}))});
}
text('Image history and adaptable visual continuation', 48, 28, 1504, 75, 43, true);
text('a', 42, 137, 40, 50, 34, true);
chart('Mean F1 on shared indices', 80, [
  {name: 'Generated history', fill: brown, values: layouts.map(k => rounded(data.history_groups[k].autonomous_shared.mean_f1))},
  {name: 'Reference history', fill: blue, values: layouts.map(k => rounded(data.history_groups[k].forced_shared.mean_f1))}]);
text('Shared positions: 738 fixed; 598 proportional.\nMissing targets: 7 fixed; 147 proportional.\nShared indices are not semantic alignment.', 85, 640, 700, 112, 25);

text('b', 832, 137, 40, 50, 34, true);
chart('Later writing with reference history', 870, [
  {name: 'Median F1', fill: blue, values: layouts.map(k => rounded(data.history_groups[k].later.median_f1))},
  {name: 'Fraction F1 ≥0.95', fill: green, values: layouts.map(k => {
    const v = data.history_groups[k].later; return rounded(v.at_least_095 / v.positions);
  })}]);
text('Gate: median ≥0.99 and fraction ≥0.90.\nBoth fail, with 713 later positions per layout.\nReference length and prior images are supplied.', 875, 640, 685, 112, 25);

text('c', 42, 802, 40, 50, 34, true);
text('Frozen prompt, adapted continuation', 80, 802, 710, 55, 30, true);
function box(id, label, x, y, width, height, trainable = false, size = 25) {
  const node = s.shapes.add({name: id, geometry: 'rect',
    position: {left: x, top: y, width, height},
    fill: trainable ? '#F5EBE3' : '#EFF4F6', line: {fill: trainable ? brown : blue, width: 1.5}});
  text(label, x + 10, y + 9, width - 20, height - 16, size);
  nodes.push({id, shape_id: node.id, label, trainable, position: {left: x, top: y, width, height}});
  return node;
}
function link(a, b, fromSide, toSide) {
  s.shapes.connect(a, b, {kind: 'straight', fromSide, toSide,
    line: {fill: muted, width: 2}, tail: {type: 'triangle', width: 'med', length: 'med'}});
  links.push({from: a.id, to: b.id, fromSide, toSide});
}
const prompt = box('prompt_pixels', 'Prompt pixels X', 80, 885, 195, 80);
const prefill = box('frozen_prefill', 'Frozen image\nprompt prefill', 325, 885, 230, 80);
const memory = box('immutable_KV', 'Prompt K,V\nfixed for each X', 325, 1015, 230, 90);
const query = box('initial_query', 'Learned\nstart query q', 80, 1155, 195, 110, true);
const answer = box('continuation', '36 frozen layers\n+ rank-8\nresiduals', 325, 1155, 230, 110, true, 24);
const writer = box('writer_stop', 'Readout R\nDecoder D\nStop', 615, 1155, 170, 110, true, 23);
const image = box('emitted_pixels', 'Emitted\nbitmap I_t', 615, 1300, 170, 80);
const encoder = box('image_feedback', 'Frozen image\nencoder + S', 325, 1300, 230, 80);
link(prompt, prefill, 'right', 'left'); link(prefill, memory, 'bottom', 'top');
link(memory, answer, 'bottom', 'top'); link(query, answer, 'right', 'left');
link(answer, writer, 'right', 'left'); link(writer, image, 'bottom', 'top');
link(image, encoder, 'left', 'right'); link(encoder, answer, 'top', 'bottom');
text('Actual pixels\nfeed the next\nstep', 80, 1300, 210, 110, 24);
text('Train q, residuals, R and stop.\nThe inherited body, E, D and prompt memory stay frozen.', 80, 1420, 710, 75, 24);

text('d', 832, 802, 40, 50, 34, true);
text('Real complete-target preflight', 875, 802, 680, 55, 30, true);
const names = ['Shortest QA', 'Largest prompt', 'Longest QA', 'Longest book'];
const tableValues = [['Case', 'Prompt', 'Target'], ...data.preflight_cases.map((c, i) =>
  [names[i], c.prompt_glyphs.toLocaleString('en-US'), c.complete_target_glyphs.toLocaleString('en-US')])];
const table = s.tables.add({rows: 5, columns: 3, left: 880, top: 885,
  width: 660, height: 305, columnWidths: [310, 175, 175], values: tableValues});
table.styleOptions = {headerRow: true, bandedRows: false};
table.borders.assign({style: 'solid', fill: grid, width: 1});
for (let row = 0; row < 5; row++) for (let col = 0; col < 3; col++) {
  const cell = table.getCell(row, col); cell.fill = row === 0 ? '#EFF4F6' : '#FFFFFF';
  cell.text.style = {typeface: font, fontSize: 26, bold: row === 0, color: row === 0 ? ink : muted};
}
text('Prompt and target sizes are glyph counts.', 885, 1207, 660, 45, 24);
text('Peak VRAM: 18.122546 GiB', 885, 1260, 660, 48, 30, true);
text('4 actual updates; 85 Adam states\n20,212,225 trainable parameters\n7.522B inherited body + frozen interfaces', 885, 1325, 660, 100, 25);
text('Prefill and causality checks pass.\nAutonomous language is unmeasured.', 885, 1435, 660, 68, 26, true);
text('Reference history is diagnostic. The ongoing complete fit has no language score yet.', 50, 1520, 1500, 38, 24);
text('Reused development cases; one model reviewer; no independent human adjudication.', 50, 1558, 1500, 38, 24);
s.speakerNotes.textFrame.setText(
  'Source: publication/ilm-nature/evidence/continuation_history_measured_data.json and its complete bound sources. '
  + 'Panel a shows canonical foreground F1 on shared numeric target indices only; different answer wording/lengths prevent semantic alignment. '
  + 'All missing/extra original positions remain recorded; reference history supplies answer information and target length, with stopping bypassed. '
  + 'Panel b uses all713later positions per layout. Both conditions fail the joint median>=.99 and90%>=.95 gate. '
  + 'These are conditional image diagnostics, never autonomous answer accuracies or a measured language-learning curve. '
  + 'Panel c is the actual continuation-only architecture: source prompt prefill bypasses new residuals; all36answerlayers recompute during fitting. '
  + 'Query/residuals/readout/stop train; source body, image encoder/decoder and affine statistics stay frozen. '
  + 'Native answer cache is separate from immutable source prompt memory at inference. Only emitted binary pixels feed back. '
  + 'Panel d binds four actual complete training cases,772target positions, four real optimizer steps and18.122546GiB allocated peak. '
  + 'Prompt/frozen-body checks are assertions from the registered GPU execution; optimizer, targets and numerical boundaries are independently audited. '
  + 'Initial residual-down tensors were not archived in that preflight. Different execution shapes use tolerance.002, not bit-exact equivalence. '
  + 'The7.522B inherited body and frozen interfaces remain deployed costs;20.212Mtrainable parameters do not define a compact foundation. '
  + 'No preflight output or in-progress fit is a new language score. No generated example or illustrative glyph is used as evidence. '
  + 'F1chart cells/labels round explicitly to six decimals; full precision remains in the bound JSON. Native chart workbooks are literal snapshots.');

const candidate = path.join(stage, 'candidate.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidate);
execFileSync(python, [path.join(root, 'publication/ilm-nature/format_continuation_history_chart_labels.py'), candidate]);
const preview = await s.export({format: 'png', scale: 1});
await fs.writeFile(path.join(stage, 'artifact-slide-1.png'), new Uint8Array(await preview.arrayBuffer()));
const layout = await s.export({format: 'layout'});
await fs.writeFile(path.join(stage, 'artifact-slide-1.layout.json'), await layout.text());
await fs.writeFile(path.join(stage, 'figure_bindings.json'), JSON.stringify({
  measured_data_sha256: hash(raw), source_sha256: data.source_sha256,
  builder_sha256: hash(await fs.readFile(path.join(root, 'publication/ilm-nature/continuation_history_figures.mjs'))),
  label_formatter_sha256: hash(await fs.readFile(path.join(root, 'publication/ilm-nature/format_continuation_history_chart_labels.py'))),
  charts, tableValues, nodes, links, native_charts: 2, native_tables: 1,
  chart_decimal_places: 6, full_precision_history: data.history_groups,
  new_model_calls: 0, unmeasured_language_score_not_zero: true,
  semantic_alignment_claimed: false, preflight_not_language_evidence: true}, null, 2) + '\n');
await finalizePresentation({workspaceDir: stage, candidatePath: candidate,
  finalPath: path.join(stage, 'output/Continuation_history_figures.pptx'), explicitTotalSlideCount: 1,
  requiredNativeChartOwnerSlides: [1], requiredNativeTableOwnerSlides: [1], materializeLiteralChartWorkbooks: true,
  fontPolicy: {basis: 'reference', families: [font],
    referencePath: path.join(root, 'publication/ilm-nature/figures/Realized_writer_figures.pptx'),
    referenceSha256: hash(await fs.readFile(path.join(root, 'publication/ilm-nature/figures/Realized_writer_figures.pptx')))},
  pythonExecutable: python,
  integrityValidatorPath: path.join(skill, 'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath: path.join(skill, 'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs: ['--expected-slide-size-emu', '15240000,15240000', '--validate-heading-fit', '--require-native-table-slide', '1'],
  verifyArtifactToolImport: true, receiptPath: path.join(stage, 'presentation_validation.json')});
