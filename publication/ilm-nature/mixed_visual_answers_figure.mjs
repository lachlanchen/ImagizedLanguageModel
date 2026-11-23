// Use the same physical dimensions, axes and glyph scale for both comparisons.
import path from 'node:path';
import fs from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
const [root,stage]=process.argv.slice(2);
const modulePath=path.join(stage,'paragraph_evidence_layout.mjs');
await fs.copyFile(path.join(root,'publication/ilm-nature/paragraph_evidence_figure.mjs'),modulePath);
await import(pathToFileURL(modulePath).href);
