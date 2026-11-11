import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';
import {Presentation, PresentationFile} from '@oai/artifact-tool';

const [root, stage, skill, python] = process.argv.slice(2);
const {applyPresentationChartFont, finalizePresentation} = await import(pathToFileURL(
  path.join(skill, 'container_tools/artifact_tool_utils.mjs')).href);
const hash = value => createHash('sha256').update(value).digest('hex');
const data = JSON.parse(await fs.readFile(path.join(stage, 'measured_data.json'), 'utf8'));
if (!data.all256_physical_positions_recounted || !data.language_independently_recounted_in_bound_source
    || !data.unmeasured_language_not_zero) throw Error('Complete bound evidence required');
const deck = Presentation.create({slideSize: {width: 1600, height: 1600}});
const s = deck.slides.add(); s.background.fill = '#FFFFFF';
const font = 'Arial', ink = '#192936', muted = '#43535C';
const blue = '#386985', brown = '#9B7457', green = '#527B70', grid = '#DDE4E8';
const charts = [];
function text(value, x, y, width, height, size = 27, bold = false) {
  const box = s.shapes.add({geometry: 'textbox', position: {left: x, top: y, width, height},
    fill: 'none', line: {fill: 'none', width: 0}});
  box.text = value;
  box.text.style = {typeface: font, fontSize: size, bold, color: bold ? ink : muted, autoFit: 'none'};
}
const axis = {textStyle: {typeface: font, fontSize: 24, fill: muted}, line: {fill: muted, width: 1}};
const titleTextStyle = {typeface: font, fontSize: 29, bold: true, fill: ink};
const legend = {position: 'bottom', overlay: false, textStyle: {fontSize: 23, fill: muted}};
function chart(config) {
  const item = s.charts.add('bar', config);
  applyPresentationChartFont(item, {fontFamily: font});
  charts.push({categories: config.categories,
    series: config.series.map(series => ({name: series.name, values: series.values})),
    position: config.position});
}
text('Physical writing and complete-answer transfer', 48, 28, 1504, 75, 44, true);
text('a', 42, 135, 40, 50, 34, true);
const rolloutRoutes = ['original_flow', 'control', 'candidate'];
chart({position: {left: 75, top: 135, width: 710, height: 495},
  title: 'Training median foreground F1', titlePlacement: 'aboveChart', titleTextStyle,
  categories: ['Original flow', 'Extra FM', 'Full rollout'],
  series: [['train_first', 'First', blue], ['train_later', 'Later', brown]].map(([group, name, fill]) =>
    ({name, fill, valuesFormatCode: '0.000000', values: rolloutRoutes.map(route =>
      Number(data.rollout_physical[group][route].median_foreground_f1.toFixed(6)))})),
  barOptions: {direction: 'bar', grouping: 'clustered', gapWidth: 90},
  hasLegend: true, legend, xAxis: axis,
  yAxis: {...axis, min: 0, max: 1.2, majorUnit: .25, numberFormatCode: '0.00',
    majorGridlines: {fill: grid, width: 1}},
  dataLabels: {showValue: true, position: 'outEnd', textStyle: {typeface: font, fontSize: 23, fill: ink}}});
text('Median gate 0.99 and ≥30/32 at F1 ≥0.95\nFull rollout passes first only', 90, 650, 680, 95, 27);

text('b', 832, 135, 40, 50, 34, true);
const physicalGroups = ['train_first', 'train_later', 'fixed_cells', 'proportional'];
chart({position: {left: 865, top: 135, width: 685, height: 495},
  title: 'Balanced writer: exact tiles (32)', titlePlacement: 'aboveChart', titleTextStyle,
  categories: ['Train first', 'Train later', 'Dev fixed first', 'Dev prop. first'],
  series: [{name: 'Exact pixels', fill: green,
    values: physicalGroups.map(group => data.balanced_physical[group].candidate.exact)}],
  barOptions: {direction: 'bar', grouping: 'clustered', gapWidth: 100}, hasLegend: false,
  xAxis: axis, yAxis: {...axis, min: 0, max: 36, majorUnit: 8, numberFormatCode: '0',
    majorGridlines: {fill: grid, width: 1}},
  dataLabels: {showValue: true, position: 'outEnd', textStyle: {typeface: font, fontSize: 29, fill: ink}}});
text('Training states and development first states\nCanonical pixels do not grade meaning', 875, 650, 680, 95, 27);

const layouts = ['fixed_cells', 'proportional'];
const models = [['parent', 'Parent', blue], ['flow', 'Flow', brown], ['balanced', 'Balanced', green]];
for (const [i, [metric, title, maximum]] of [
  ['exact', 'Complete exact QA (32)', 32], ['pairs', 'Both correct (16 pairs)', 16]].entries()) {
  text('cd'[i], 42 + i * 790, 795, 40, 50, 34, true);
  chart({position: {left: 75 + i * 790, top: 795, width: 710, height: 475},
    title, titlePlacement: 'aboveChart', titleTextStyle, categories: ['Fixed cells', 'Proportional'],
    series: models.map(([model, name, fill]) => ({name, fill,
      values: layouts.map(layout => data.qa[model][layout][metric])})),
    barOptions: {direction: 'column', grouping: 'clustered', gapWidth: 90},
    hasLegend: true, legend, xAxis: axis,
    yAxis: {...axis, min: 0, max: maximum, majorUnit: maximum / 4, numberFormatCode: '0',
      majorGridlines: {fill: grid, width: 1}},
    dataLabels: {showValue: true, position: 'outEnd', textStyle: {typeface: font, fontSize: 29, fill: ink}}});
}
const retained = data.balanced_union_retention.proportional.retained.length;
text(`Balanced retains ${retained}/8 required proportional exact answers. Prospective pilot: ${data.balanced_pilot_passed ? 'pass' : 'fail'}.`,
  80, 1310, 1460, 60, 29, true);
text('Readout and stopping change together. The reader and image codec remain frozen.\nRollout whole-answer scores are unmeasured and have no language bars.',
  80, 1380, 1460, 100, 28);
text('Model comparisons, not a language-learning curve. Reused development cohort.\nOne model reviewer, no independent human adjudication or untouched confirmation.',
  50, 1510, 1500, 80, 26);
s.speakerNotes.textFrame.setText(
  'Sources: references/evidence/rollout_endpoint_motor_learning_20261111.json; '
  + 'balanced_writer_matched_diagnosis_learning_20261111.json; balanced_writer_matched_learning_20261111.json; '
  + 'glyph_context_motor_learning_20261111.json; glyph_tail_first_writing_learning_20261111.json. '
  + 'Panel a compares original12000-update EMA with two matched2000-update continuations from that EMA. '
  + 'Only the full-rollout arm adds balanced frozen-decoder BCE through all32actual sampling intervals. '
  + 'The median threshold alone is insufficient: each32-position pool also requires at least30tiles atF1>=.95. '
  + 'Neither continuation passes both gates; both full language trials remain unmeasured, never zero. '
  + 'Panel b uses32positions in each of four unchanged physical diagnostic groups. '
  + 'Exact canonical first-image agreement is distinct from semantic accuracy. '
  + 'Panels c,d retain every32normal QA question and16paired passages in both layouts. '
  + 'All216balanced streams,54opaque boards and12prefixes are reviewed before reference scoring, '
  + 'with independent score recount and initial-state comparison in the bound evidence. '
  + 'The balanced readout and stop predictor differ as one bundle. There are zero new fitting updates in that comparison. '
  + 'The inherited7.522Bvisual/contextual body, prior fitting and offline preparation remain costs. '
  + 'One model reviewer has prior cohort familiarity. No independent human adjudication or untouched confirmation. '
  + 'These fixed model comparisons are not measured language-learning trajectories or a general efficiency result. '
  + 'No glyph images are repaired or illustrated by this quantitative figure. Full precision remains in bound JSON, '
  + 'with F1chart caches,workbooks and labels rounded explicitly to six decimals.');
await fs.mkdir(path.join(stage, 'output'), {recursive: true});
const candidate = path.join(stage, 'candidate.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidate);
execFileSync(python, [path.join(root, 'publication/ilm-nature/format_motor_chart_labels.py'), candidate]);
const preview = await s.export({format: 'png', scale: 1});
await fs.writeFile(path.join(stage, 'artifact-slide-1.png'), new Uint8Array(await preview.arrayBuffer()));
await fs.writeFile(path.join(stage, 'figure_bindings.json'), JSON.stringify({
  measured_data_sha256: hash(await fs.readFile(path.join(stage, 'measured_data.json'))),
  source_sha256: data.source_sha256, charts, native_charts: 4,
  full_precision_rollout: data.rollout_physical, full_precision_balanced: data.balanced_physical,
  chart_F1_decimal_places: 6, new_optimizer_updates: 0,
  unmeasured_language_score_not_zero: true, model_comparison_not_learning_curve: true}, null, 2) + '\n');
await finalizePresentation({workspaceDir: stage, candidatePath: candidate,
  finalPath: path.join(stage, 'output/Realized_writer_figures.pptx'), explicitTotalSlideCount: 1,
  requiredNativeChartOwnerSlides: [1], materializeLiteralChartWorkbooks: true,
  fontPolicy: {basis: 'reference', families: [font],
    referencePath: path.join(root, 'publication/ilm-nature/figures/Motor_diagnostics_figures.pptx'),
    referenceSha256: hash(await fs.readFile(path.join(root, 'publication/ilm-nature/figures/Motor_diagnostics_figures.pptx')))},
  pythonExecutable: python,
  integrityValidatorPath: path.join(skill, 'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath: path.join(skill, 'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs: ['--expected-slide-size-emu', '15240000,15240000', '--validate-heading-fit'],
  verifyArtifactToolImport: true, receiptPath: path.join(stage, 'presentation_validation.json')});
