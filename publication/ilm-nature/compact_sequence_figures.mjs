import fs from "node:fs/promises";
import path from "node:path";
import { createHash } from "node:crypto";
import { pathToFileURL } from "node:url";
import sharp from "sharp";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const [root, stage, skill, python] = process.argv.slice(2);
const { applyPresentationChartFont, finalizePresentation } = await import(pathToFileURL(
  path.join(skill, "container_tools/artifact_tool_utils.mjs"),
).href);
const evidenceDir = path.join(root, "references/evidence");
const sourcePath = path.join(evidenceDir, "compact_visual_span_learning_20261111.json");
const evidence = JSON.parse(await fs.readFile(sourcePath, "utf8"));
const baselinePath = path.join(evidenceDir, "compact_visual_span_qwen_uncertainty_20261111.json");
const baseline = JSON.parse(await fs.readFile(baselinePath, "utf8"));
const rawDir = path.join(evidenceDir, evidence.raw_examples_directory);
const hash = bytes => createHash("sha256").update(bytes).digest("hex");
const steps = evidence.checkpoint_bindings.map(row => row.step);
if (steps.join() !== "0,1956,4049,6152,8172" || evidence.review_summary.images_reviewed !== 372) {
  throw new Error("Require the completed fixed V1 trajectory and all opaque readings");
}
const csv = (await fs.readFile(path.join(rawDir, "language_metrics.csv"), "utf8")).trim().split(/\r?\n/);
const csvRows = csv.slice(1).map(line => {
  const [step, condition, metric, count, cases, rate, low, high] = line.split(",");
  return {step: +step, condition, metric, count: +count, cases: +cases, rate: +rate, low: +low, high: +high};
});
for (const row of evidence.metrics) {
  const found = csvRows.find(x => x.step === row.step && x.condition === row.condition && x.metric === row.metric);
  if (!found || found.count !== row.count || found.cases !== row.cases || found.rate !== row.mean
      || found.low !== row.ci95[0] || found.high !== row.ci95[1]) throw new Error("CSV/JSON mismatch");
}

const font = "Arial";
const ink = "#192936", muted = "#43535C", blue = "#386985", reference = "#9B7457", grid = "#DDE4E8";
const deck = Presentation.create({slideSize: {width: 1600, height: 1600}});
function label(slide, text, left, top, width, height, size = 26, bold = false) {
  const shape = slide.shapes.add({geometry: "textbox", position: {left, top, width, height},
    fill: "none", line: {fill: "none", width: 0}});
  shape.text = text;
  shape.text.style = {typeface: font, fontSize: size, bold, color: bold ? ink : muted, autoFit: "none"};
  return shape;
}
function slideTitle(slide, text) {label(slide, text, 48, 28, 1504, 70, 48, true);}
const axes = {textStyle: {typeface: font, fontSize: 25, fill: muted}, line: {fill: muted, width: 1}};
const measurements = deck.slides.add(); measurements.background.fill = "#FFFFFF";
const authoredSlides = [measurements];
slideTitle(measurements, "Compact V1: complete-answer evaluation");
label(measurements, "a", 40, 135, 40, 50, 34, true);
label(measurements, "b", 820, 135, 40, 50, 34, true);
const chartRows = [];
for (const [index, [metric, title, denominator]] of [
  ["qa_complete_exact", "Complete exact QA (32 questions)", 32],
  ["qa_both_correct", "Both questions correct (16 pairs)", 16],
].entries()) {
  const values = steps.map(step => evidence.metrics.find(row => row.step === step && row.condition === "normal" && row.metric === metric));
  const textReference = baseline.metrics.find(row => row.metric === metric);
  if (values.some(row => row.cases !== denominator || row.count !== 0 || row.ci95.some(x => x !== 0))) throw new Error("Unexpected trajectory");
  const chart = measurements.charts.add("line", {
    position: {left: 70 + index * 780, top: 135, width: 705, height: 540},
    title, titlePlacement: "aboveChart", titleTextStyle: {typeface: font, fontSize: 30, bold: true, fill: ink},
    categories: steps.map(String),
    series: [
      {name: "Visual model", values: values.map(x => x.mean), line: {fill: blue, width: 3}, marker: {symbol: "circle", size: 10}},
      {name: "Text reference (CPU)", values: steps.map(() => textReference.mean), line: {fill: reference, width: 2}, marker: {symbol: "none"}},
    ],
    lineOptions: {smooth: false}, hasLegend: true,
    legend: {position: "bottom", overlay: false, textStyle: {fontSize: 24, fill: muted}},
    xAxis: {...axes, title: {text: "Optimizer update", textStyle: {fontSize: 24, fill: muted}}},
    yAxis: {...axes, min: 0, max: 1, majorUnit: .25, numberFormatCode: "0%", majorGridlines: {fill: grid, width: 1}},
  });
  applyPresentationChartFont(chart, {fontFamily: font});
  label(measurements,
    `Visual model: 0/${denominator} at every checkpoint\nText reference: ${textReference.count}/${denominator}; 95% interval ` +
    `${(textReference.ci95[0] * 100).toFixed(3).replace(/0+$/, "").replace(/\.$/, "")}% to ` +
    `${(textReference.ci95[1] * 100).toFixed(3).replace(/0+$/, "").replace(/\.$/, "")}%`,
    80 + index * 780, 690, 690, 90, 26);
  chartRows.push({metric, values: values.map(x => x.mean), reference: textReference.mean, reference_ci95: textReference.ci95});
}
label(measurements, "Pointwise passage-group intervals are [0, 0] for every visual-model point.\nConstant observations do not imply a zero population success probability.",
  80, 795, 1440, 80, 27);
label(measurements, "c", 40, 925, 40, 50, 34, true);
const normal = evidence.normal_output_counts;
if (normal.some((row, i) => row.step !== steps[i] || row.cases !== 44 || row.stopped + row.capped !== 44)) throw new Error("Termination counts differ");
const stopping = measurements.charts.add("bar", {
  position: {left: 75, top: 920, width: 1450, height: 475},
  title: "Normal-prompt termination (44 inputs)", titlePlacement: "aboveChart",
  titleTextStyle: {typeface: font, fontSize: 30, bold: true, fill: ink},
  categories: steps.map(String), series: [
    {name: "Predicted stop", values: normal.map(x => x.stopped), fill: blue},
    {name: "Generation budget reached", values: normal.map(x => x.capped), fill: reference,
      dataLabelOverrides: normal.map((x, idx) => ({idx, showValue: x.capped > 0,
        showSeriesName: false, showCategoryName: false, position: "center",
        textStyle: {typeface: font, fontSize: 25, fill: "#FFFFFF"}}))},
  ], barOptions: {direction: "column", grouping: "stacked", gapWidth: 90}, hasLegend: true,
  legend: {position: "bottom", overlay: false, textStyle: {fontSize: 25, fill: muted}},
  xAxis: {...axes, title: {text: "Optimizer update", textStyle: {fontSize: 25, fill: muted}}},
  yAxis: {...axes, min: 0, max: 44, majorUnit: 11, numberFormatCode: "0", majorGridlines: {fill: grid, width: 1}},
  dataLabels: {showValue: true, position: "center", textStyle: {fontSize: 25, fill: "#FFFFFF"}},
});
applyPresentationChartFont(stopping, {fontFamily: font});
label(measurements, `Empty responses, included above: ${normal.map(x => x.empty).join(", ")} of 44 at successive checkpoints.`, 80, 1410, 1440, 50, 27);
label(measurements, "All 372 outputs retained. One model-based visual review; no independent human adjudication.\nTraining completion and stopping do not establish readable language generation.",
  80, 1495, 1440, 80, 26);
measurements.speakerNotes.textFrame.setText(`Sources: ${sourcePath}; ${baselinePath}. All values authenticate against the analysis CSV. Bootstrap resamples sixteen passage groups, 10000 replicates, seed 202611154. Book coherence is limited by unreadable whole-output transcriptions. The reference ran on CPU; no matched-quality efficiency claim.`);

const cases = ["qa_01_a_normal", "qa_01_b_normal", "book_01_normal"];
const titles = ["Fixed QA question A: raw outputs", "Fixed QA question B: raw outputs", "Fixed book continuation: raw outputs"];
const pictures = [];
for (const [index, caseName] of cases.entries()) {
  const slide = deck.slides.add(); slide.background.fill = "#FFFFFF"; slideTitle(slide, titles[index]);
  authoredSlides.push(slide);
  const inputPath = path.join(rawDir, "raw_examples", caseName, "input.png");
  const input = await fs.readFile(inputPath);
  const matching = evidence.raw_examples.filter(row => row.case === caseName);
  if (matching.length !== 5 || matching.some(row => row.input_sha256 !== hash(input))) throw new Error("Fixed input changed");
  const inputMeta = await sharp(input).metadata();
  const crop = {left: 0, top: Math.max(0, inputMeta.height - 96), width: inputMeta.width, height: Math.min(96, inputMeta.height)};
  const preview = await sharp(input).extract(crop).png().toBuffer();
  label(slide, "Visible input: last three rows; the complete source PNG is retained in the evidence", 32, 100, 1536, 36, 26);
  slide.images.add({blob: preview, contentType: "image/png", fit: "contain", alt: `${caseName} last three input rows`,
    position: {left: 32, top: 140, width: crop.width * 1.5, height: crop.height * 1.5}});
  pictures.push({slide: index + 2, case: caseName, input_source_sha256: hash(input), input_preview_crop: crop});
  for (const [i, step] of steps.entries()) {
    const item = matching.find(row => row.step === step);
    const bytes = await fs.readFile(path.join(rawDir, item.file));
    if (hash(bytes) !== item.image_sha256) throw new Error("An original answer image changed");
    const meta = await sharp(bytes).metadata();
    if (meta.width > 1024 || meta.height > 128) throw new Error("Unexpected raw output grid");
    const top = 300 + i * 240;
    label(slide, `Update ${step.toLocaleString("en-US")}     ${item.emitted_glyphs} glyphs     ${item.stopped ? "predicted stop" : "generation budget reached"}${item.emitted_glyphs === 0 ? " (blank output)" : ""}`,
      32, top, 1536, 36, 28, true);
    slide.images.add({blob: bytes, contentType: "image/png", fit: "contain", alt: `${caseName}, step ${step}, original unrepaired answer`,
      position: {left: 32, top: top + 44, width: meta.width * 1.5, height: meta.height * 1.5}});
    pictures.push({slide: index + 2, case: caseName, step, source: path.relative(root, path.join(rawDir, item.file)), sha256: hash(bytes),
      glyphs: item.emitted_glyphs, stopped: item.stopped});
  }
  label(slide, "The first QA pair and first book prefix were fixed before fitting. All five outputs are shown.\nOriginal answer PNGs are embedded without cropping, reconstruction, OCR replacement or repair.",
    32, 1520, 1536, 74, 25);
  slide.speakerNotes.textFrame.setText(`Source: ${sourcePath}. Case ${caseName}. This figure contains the exact original answer PNGs bound by the source receipt. Only the identified input preview is cropped; its full PNG remains in the evidence directory. Model outputs failed complete-response review; recognizable individual glyphs do not supply a complete answer.`);
}

await fs.mkdir(path.join(stage, "output"), {recursive: true});
const candidate = path.join(stage, "candidate.pptx");
await (await PresentationFile.exportPptx(deck)).save(candidate);
for (let i = 0; i < 4; i++) {
  const preview = await authoredSlides[i].export({format: "png", scale: 1});
  await fs.writeFile(path.join(stage, `artifact-slide-${i + 1}.png`), new Uint8Array(await preview.arrayBuffer()));
}
await fs.writeFile(path.join(stage, "source_bindings.json"), JSON.stringify({
  source_sha256: hash(await fs.readFile(sourcePath)), reference_sha256: hash(await fs.readFile(baselinePath)),
  metrics: chartRows, stop_counts: normal.map(x => x.stopped), cap_counts: normal.map(x => x.capped),
  pictures, original_answer_pngs: 15, native_charts: 3,
}, null, 2) + "\n");
await finalizePresentation({workspaceDir: stage, candidatePath: candidate,
  finalPath: path.join(stage, "output/Compact_sequence_figures.pptx"), explicitTotalSlideCount: 4,
  requiredNativeChartOwnerSlides: [1], materializeLiteralChartWorkbooks: true,
  fontPolicy: {basis: "design", families: [font]}, pythonExecutable: python,
  integrityValidatorPath: path.join(skill, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(skill, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: ["--expected-slide-size-emu", "15240000,15240000", "--validate-heading-fit"],
  verifyArtifactToolImport: true, receiptPath: path.join(stage, "presentation_validation.json"),
});
