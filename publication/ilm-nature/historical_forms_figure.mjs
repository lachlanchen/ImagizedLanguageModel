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
const source = path.join(root, "references/evidence");
const evidence = JSON.parse(await fs.readFile(path.join(source, "historical_visual_evaluation_20261111.json"), "utf8"));
const hash = bytes => createHash("sha256").update(bytes).digest("hex");
const figures = new Map();
for (const row of evidence.boards) {
  const bytes = await fs.readFile(path.join(source, row.file));
  if (hash(bytes) !== row.sha256) throw new Error(`Source image changed: ${row.file}`);
  figures.set(row.file, bytes);
}

const font = "Arial";
const titleInk = "#192936";
const bodyInk = "#394A55";
const deck = Presentation.create({ slideSize: { width: 1600, height: 900 } });
const slide = deck.slides.add();
slide.background.fill = "#FFFFFF";
function label(value, left, top, width, height, fontSize, bold = false) {
  const text = slide.shapes.add({ geometry: "textbox", position: { left, top, width, height },
    fill: "none", line: { fill: "none", width: 0 } });
  text.text = value;
  text.text.style = { typeface: font, fontSize, bold, color: bold ? titleInk : bodyInk, autoFit: "none" };
}
async function crop(file, bounds, left, top, width, height, alt) {
  const bytes = await sharp(figures.get(file)).extract(bounds).png().toBuffer();
  slide.images.add({ blob: bytes, contentType: "image/png", alt, fit: "contain",
    position: { left, top, width, height } });
}

const names = ["Oracle", "Bronze", "Seal", "Tall bronze"];
const keys = ["historical_32/oracle", "historical_32/bronze", "historical_32/seal", "bronze_tall_128/bronze"];
const data = keys.map(key => evidence.fixed_codec.summary[key]);
const categories = names.map((name, i) => `${name} (${data[i].cases})`);
const exact = data.map(row => Number((row.exact / row.cases).toFixed(6)));
const f1 = data.map(row => Number(row.mean_foreground_f1.toFixed(6)));
label("Historical forms through the glyph codec", 56, 32, 1488, 66, 48, true);
label("a  Reconstruction fidelity", 56, 125, 765, 47, 32, true);
const chart = slide.charts.add("bar", {
  position: { left: 54, top: 188, width: 790, height: 516 }, categories,
  series: [{ name: "Exact images", values: exact, fill: "#496E88" },
    { name: "Mean foreground F1", values: f1, fill: "#C98755" }],
  barOptions: { direction: "column", grouping: "clustered", gapWidth: 72 }, hasLegend: true,
  legend: { position: "bottom", overlay: false, textStyle: { fontSize: 24, fill: bodyInk } },
  xAxis: { textStyle: { fontSize: 22, fill: bodyInk } },
  yAxis: { min: 0, max: 1, majorUnit: 0.25, numberFormatCode: "0%",
    textStyle: { fontSize: 21, fill: bodyInk },
    majorGridlines: { fill: "#DDE4E8", width: 1, style: "solid" } },
});
applyPresentationChartFont(chart, { fontFamily: font });
label("Numbers in parentheses give the number of forms.", 65, 720, 752, 45, 23);

label("b  Source rasterization", 872, 125, 670, 47, 32, true);
const resolution = "historical_visual_sparse_source_resolution_20261111.png";
for (let i = 0; i < 3; i++) {
  const size = [32, 64, 128][i];
  const x = 902 + i * 205;
  label(`${size} × ${size}`, x + 20, 182, 145, 32, 23);
  await crop(resolution, { left: 7 + i * 146, top: 36, width: 128, height: 128 },
    x, 218, 160, 160, `B12643 at ${size} pixels square`);
}
const measured = evidence.sparse_source_resolution.raster_measurements;
label(`B12643 retains ${measured["32"].ink_pixels} ink pixels at 32 × 32 and ${measured["128"].ink_pixels} at 128 × 128.`,
  875, 384, 670, 55, 23);

label("c  Four-tile decoding", 872, 465, 670, 47, 32, true);
const tall = "historical_visual_bronze_tall_128_20261111.png";
for (let i = 0; i < 2; i++) {
  const x = 944 + 183 * i;
  label(i ? "Decoded" : "Source", x, 523, 145, 32, 23);
  await crop(tall, { left: 448 + i * 70, top: 27, width: 64, height: 256 },
    x + 16, 563, 58, 232, `B12642 ${i ? "decoded from four tiles" : "source crop"}`);
}
const tallWorst = evidence.fixed_codec.records.find(row => row.label === "B12642");
label(`B12642\nF1 ${tallWorst.foreground_f1.toFixed(3)}`, 1313, 620, 220, 106, 25);
label("Supplied-image reconstruction through the fixed codec. Historical answer generation remains unmeasured.",
  57, 837, 1483, 43, 23);
slide.speakerNotes.textFrame.setText(
  "Source: references/evidence/historical_visual_evaluation_20261111.json. " +
  "The panels show observed source and decoded pixels. B12643 and B12642 illustrate the sparsest input and lowest tall-crop F1.",
);

const candidate = path.join(stage, "candidate.pptx");
await (await PresentationFile.exportPptx(deck)).save(candidate);
const preview = await slide.export({ format: "png", scale: 1 });
await fs.writeFile(path.join(stage, "artifact-preview.png"), new Uint8Array(await preview.arrayBuffer()));
await fs.mkdir(path.join(stage, "output"), { recursive: true });
await finalizePresentation({
  workspaceDir: stage, candidatePath: candidate, finalPath: path.join(stage, "output/Figure_11_historical_forms.pptx"),
  explicitTotalSlideCount: 1, requiredNativeChartOwnerSlides: [1],
  materializeLiteralChartWorkbooks: true, fontPolicy: { basis: "design", families: [font] },
  pythonExecutable: python,
  integrityValidatorPath: path.join(skill, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(skill, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: ["--expected-slide-size-emu", "15240000,8572500", "--validate-heading-fit"],
  verifyArtifactToolImport: true, receiptPath: path.join(stage, "presentation_validation.json"),
});
