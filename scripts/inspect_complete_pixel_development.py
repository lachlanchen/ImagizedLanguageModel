#!/usr/bin/env python3
"""Build unscaled, all-case review plates after the independent full audit.

Only labels use a font. Prompt, answer and reference pixels are copied from
their original PNGs. These plates do not assign semantic correctness.
"""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect(run, output):
    audit = load(run / 'audit.json')
    producer = run / 'execution/report.json'
    status = load(run / 'execution/status.json')
    if status['phase'] != 'completed' or status['pid'] is not None:
        raise ValueError('finish and independently audit the model process first')
    if audit['report_sha256'] != sha(producer):
        raise ValueError('the independent audit belongs to a different report')
    if not audit['all_output_PNGs_match_emitted_tensors']:
        raise ValueError('the independent physical-output audit must pass')
    scores = {(r['index'], r['route']): r for r in audit['complete_cases']
              if r['stage'] == 'development'}
    expected = {(i, route) for i in range(32, 56) for route in ('normal', 'blank')}
    if set(scores) != expected:
        raise ValueError('all 48 registered development streams are required')
    cases = load(run / 'cases.json')
    if output.exists():
        raise FileExistsError('preserve the original review and use a fresh path')
    output.mkdir(parents=True)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
    plates = []
    for index in range(32, 56):
        case = cases[index]
        sections = []
        hashes = {}

        def original(path, digest):
            if sha(path) != digest:
                raise ValueError('an audited source image changed: ' + str(path))
            hashes[str(path.relative_to(run))] = digest
            with Image.open(path) as image:
                return image.convert('RGB')

        for route in ('normal', 'blank'):
            image = original(run / case[route], case[route + '_sha256'])
            sections.append((f'{route}: complete input PNG ({image.width} x {image.height})', [image]))
            folder = run / 'execution/development' / f'{index:02}_{route}'
            receipt = load(folder / 'receipt.json')
            identity = receipt['identity']
            if identity['index'] != index or identity['route'] != route:
                raise ValueError('the output source identity differs')
            path = folder / 'answer.png'
            image = original(path, receipt['hashes']['answer.png'])
            row = scores[index, route]
            label = (f'{route}: emitted {row["blocks"]} blocks; exact {row["exact"]}; '
                     f'F1 {row["foreground_f1"]:.6f}; stop {row["stopped"]}; cap {row["capped"]}')
            # Crops partition every column exactly; no resampling or ink repair.
            parts = [image.crop((x, 0, min(x + 1024, image.width), image.height))
                     for x in range(0, image.width, 1024)]
            sections.append((label, parts))
        for j, reference in enumerate(case['references']):
            image = original(run / reference['path'], reference['sha256'])
            sections.append((f'Complete owning reference {j + 1} ({image.width} x {image.height})',
                             [image.crop((x, 0, min(x + 1024, image.width), image.height))
                              for x in range(0, image.width, 1024)]))
        width = max(1100, max(im.width for _, parts in sections for im in parts) + 24)
        height = 50 + sum(28 + sum(im.height + 4 for im in parts) + 12 for _, parts in sections)
        canvas = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 8), f'Case {index:02} | {case["source"]} | {case["id"]} | original pixels',
                  font=font, fill='black')
        y = 42
        for label, parts in sections:
            draw.text((12, y), label, font=font, fill='black')
            y += 28
            for part in parts:
                canvas.paste(part, (12, y))
                y += part.height + 4
            y += 12
        path = output / f'case_{index:02}.png'
        canvas.save(path)
        plates.append(dict(index=index, source=case['source'], pair=case['pair'],
                           path=path.name, sha256=sha(path), original_images_sha256=hashes,
                           width=width, height=height))
    result = dict(audit_sha256=sha(run / 'audit.json'), cases_sha256=sha(run / 'cases.json'),
                  code_sha256=sha(Path(__file__)), cases=24, output_streams=48, plates=plates,
                  generated_pixels_resized=False, generated_pixels_repaired=False,
                  semantic_accuracy_assigned=False, actual_human_review_recorded=False)
    (output / 'manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'plates'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inspect(args.run.resolve(), args.output.resolve())
