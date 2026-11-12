"""CPU-only identifiability analysis on authentic, complete visual targets.

No neural checkpoint, teacher IDs, OCR, or generated answer is used. This is a
representation experiment and cannot establish language learning.
"""
import argparse
from bisect import bisect_right
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ilm.visual_lm.glyph_volume import pack_glyph_volume, unpack_glyph_volume

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def board(path, example):
    x = example[:8]
    recovered = unpack_glyph_volume(pack_glyph_volume(x))
    low = unpack_glyph_volume(pack_glyph_volume(x), retain_details=False)
    image = Image.new('RGB', (1050, 550), 'white')
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('DejaVuSans.ttf', 17)
    draw.text((20, 12), 'Ordered glyph-volume recovery: real corpus pixels; no neural generation', fill='black', font=font)
    for row, (label, cells) in enumerate([
        ('Original stream', x), ('Reversed stream', x[::-1]),
        ('Recovered with details', recovered), ('Mean only: details removed', low),
    ]):
        y = 55 + row * 92
        draw.text((20, y + 24), label, fill='black', font=font)
        for i, cell in enumerate(cells):
            tile = Image.fromarray(np.rint(255 * (1 - cell[0])).astype(np.uint8)).resize((80, 80), Image.Resampling.NEAREST)
            image.paste(tile, (300 + 89 * i, y))
    overlay = x.max(0)[0]
    tile = Image.fromarray((255 * (1 - overlay)).astype(np.uint8)).resize((92, 92), Image.Resampling.NEAREST)
    image.paste(tile, (300, 430)); image.paste(tile, (480, 430))
    draw.text((20, 456), 'Crushed ink: both orders', fill='black', font=font)
    draw.text((420, 460), '=', fill='black', font=font)
    draw.text((600, 448), 'Identical overlay; different stream order.', fill='black', font=font)
    draw.text((600, 480), 'Exact recovery retains all detail channels.', fill='black', font=font)
    image.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'artifacts/visual_continuation_full_logger_recovery_v2')
    parser.add_argument('--corpus', type=Path, default=ROOT / 'data/visual_sequence_corpus_v1')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=256, help='0 means all eligible complete targets')
    parser.add_argument('--max-seconds', type=float, default=120)
    args = parser.parse_args()
    if args.limit < 0 or args.max_seconds <= 0:
        parser.error('Use a nonnegative limit and positive time bound')
    if args.out.exists():
        parser.error('Output directory already exists; preserve earlier evidence')
    began = time.perf_counter()
    prepared = json.loads((args.run / 'preparation.json').read_text())
    manifest_file = args.run / 'training_manifest.json'
    if sha(manifest_file) != prepared['training_manifest_sha256']:
        raise ValueError('Registered manifest changed')
    if sha(args.corpus / 'report.json') != prepared['corpus_report_sha256']:
        raise ValueError('Corpus report changed')
    rows = json.loads(manifest_file.read_text())['rows']
    if args.limit:
        rows = rows[:args.limit]
    shards = [s for s in json.loads((args.corpus / 'report.json').read_text())['shards'] if s['split'] == 'train']
    offsets = [0]
    for s in shards:
        offsets.append(offsets[-1] + s['records'])
    cached_shard = None; pixels = None; glyph_offsets = None
    counts = dict(records=0, glyphs=0, exact_roundtrip_records=0, nonpalindromic_records=0,
                  sum_overlay_collisions=0, max_overlay_collisions=0, ordered_volume_distinguishes=0,
                  low_only_changed_records=0, source_scalars=0, coefficient_scalars=0,
                  shorter_volume_positions=0)
    kinds = {}; checked_shards = {}; ledger = hashlib.sha256(); error_sum = 0.; example = None
    maximum_error = 0.
    for row in rows:
        if time.perf_counter() - began > args.max_seconds:
            raise TimeoutError('CPU analysis time bound exceeded; no complete result emitted')
        index = row['corpus_index']; shard = bisect_right(offsets, index) - 1
        if not 0 <= index < offsets[-1]:
            raise ValueError('Corpus index outside authenticated shards')
        if shard != cached_shard:
            name, = [n for n in shards[shard]['files'] if n.endswith('_pixels.npz')]
            if Path(name).name != name:
                raise ValueError('Unsafe shard filename')
            digest = sha(args.corpus / name)
            if digest != shards[shard]['files'][name]:
                raise ValueError('Corpus pixels changed')
            checked_shards[name] = digest
            with np.load(args.corpus / name, allow_pickle=False) as arrays:
                pixels = arrays['ink']; glyph_offsets = arrays['record_glyph_offsets']
            cached_shard = shard
        first, last = glyph_offsets[index - offsets[shard]:index - offsets[shard] + 2]
        ink = np.unpackbits(pixels[first:last], bitorder='big').reshape(-1, 1, 32, 32)[row['cut']:]
        if len(ink) != row['target_glyphs']:
            raise ValueError('Incomplete target')
        volume = pack_glyph_volume(ink)
        recovered = unpack_glyph_volume(volume)
        maximum_error = max(maximum_error, float(np.max(np.abs(recovered - ink))))
        exact = np.array_equal(recovered, ink)
        if not exact:
            raise ValueError('Ordered reconstruction failed')
        counts['records'] += 1; counts['glyphs'] += len(ink); counts['exact_roundtrip_records'] += int(exact)
        counts['source_scalars'] += ink.size; counts['coefficient_scalars'] += volume.coefficients.size
        counts['shorter_volume_positions'] += len(volume.coefficients)
        kinds[row['kind']] = kinds.get(row['kind'], 0) + 1
        reverse = ink[::-1]
        if not np.array_equal(ink, reverse):
            counts['nonpalindromic_records'] += 1
            counts['sum_overlay_collisions'] += int(np.array_equal(ink.sum(0, dtype=np.uint32), reverse.sum(0, dtype=np.uint32)))
            counts['max_overlay_collisions'] += int(np.array_equal(ink.max(0), reverse.max(0)))
            counts['ordered_volume_distinguishes'] += int(not np.array_equal(volume.coefficients, pack_glyph_volume(reverse).coefficients))
        low = unpack_glyph_volume(volume, retain_details=False)
        counts['low_only_changed_records'] += int(not np.array_equal(low, ink))
        error_sum += float(np.square(low - ink).sum(dtype=np.float64))
        ledger.update((row['id'] + ':' + hashlib.sha256(ink.tobytes()).hexdigest() + '\n').encode())
        if example is None and len(ink) >= 8 and not np.array_equal(ink[:8], ink[:8][::-1]):
            example = ink[:8].copy()
    if not args.limit and (counts['records'] != prepared['training_cases'] or counts['glyphs'] != prepared['target_positions_per_epoch']):
        raise ValueError('Full-corpus coverage differs from registration')
    args.out.mkdir(parents=True)
    if example is not None:
        board(args.out / 'stream_recovery.png', example)
    report = dict(schema='ordered-glyph-volume-identifiability-v1', neural_training=False,
        neural_inference=False, teacher_ids_loaded=False, language_quality_measured=False,
        full_eligible_corpus=not bool(args.limit), group=4, counts=counts, kinds=kinds,
        max_absolute_recovery_error=maximum_error, low_only_pixel_mse=error_sum / counts['source_scalars'],
        source_bitpacked_bytes=(counts['source_scalars'] + 7) // 8,
        volume_float32_bytes=4 * counts['coefficient_scalars'],
        temporal_positions_ratio=counts['glyphs'] / counts['shorter_volume_positions'],
        scalar_count_ratio=counts['coefficient_scalars'] / counts['source_scalars'],
        finite_rate_warning='Shorter time axis with extra channels is not fewer bits or measured model speedup.',
        registration_manifest_sha256=prepared['training_manifest_sha256'], corpus_report_sha256=prepared['corpus_report_sha256'],
        checked_pixel_shards=checked_shards, ordered_case_pixel_ledger_sha256=ledger.hexdigest(),
        source_sha256={str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__).resolve(), ROOT / 'ilm/visual_lm/glyph_volume.py']},
        elapsed_seconds=time.perf_counter() - began)
    (args.out / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({k: report[k] for k in ['counts','kinds','max_absolute_recovery_error','low_only_pixel_mse','elapsed_seconds']}))


if __name__ == '__main__':
    main()
