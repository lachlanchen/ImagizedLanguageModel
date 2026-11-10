"""Collect completed copy, free-answer and feedback controls; no neural calls."""
import hashlib
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
EVIDENCE = ROOT / 'references/evidence'
DOCUMENTS = (0, 5, 10, 15)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text())


def collect():
    bound = {}

    def read(name):
        path = EVIDENCE / name
        bound[str(path.relative_to(ROOT))] = sha(path)
        return load(path)

    copy = [read(name + '/qualification.json') for name in
            ['genuine_glyph_readout_v1', 'direct_pixel_readout_g_only_v1']]
    c3 = read('c3_glyph_language_assessment_v1/scores.json')
    archive = read('c3_glyph_language_assessment_v1/archive.json')
    review = read('c3_glyph_language_assessment_v1/review_manual_review.json')
    feedback = read('glyph_feedback_ceiling_v1/scores.json')
    helper = read('glyph_feedback_ceiling_v1/report.json')
    preflight = read('glyph_tail_preflight_v2/preflight_fit.json')
    require(not c3['runtime_answers_replaced'] and not c3['approximate_nearest_glyph_decoding'],
            'repaired outputs are not evidence')
    require(len(c3['records']) == 128 and archive['every_output_PNG_preserved'], 'retain the whole C3 cohort')
    require(feedback['helper_only_symbolic_decoding'] and not feedback['learned_ILM_accuracy'],
            'source policy is a helper, not an ILM')
    cells = []
    for document in DOCUMENTS:
        for order in ['original', 'rotated']:
            for style in ['letter', 'phrase']:
                for version in ['baseline', 'candidate']:
                    # The fixed filename is authenticated against both the
                    # complete archive and the scored actual output record.
                    name = f'{version}_doc{document:02}_{order}_{style}.png'
                    rows = [r for r in c3['records'] if r['png'] == name]
                    require(len(rows) == 1, 'fixed C3 example missing')
                    row = rows[0]
                    path = EVIDENCE / 'c3_glyph_language_assessment_v1/raw_outputs' / name
                    digest = sha(path)
                    require(digest == row['png_sha256'] == archive['files_sha256']['raw_outputs/' + name],
                            'actual C3 output pixels changed')
                    size = Image.open(path).size
                    require(size == (max(1, row['glyphs']) * 32, 32), 'complete glyph stream geometry differs')
                    bound[str(path.relative_to(ROOT))] = digest
                    cells.append(dict(document=document, order=order, style=style, version=version,
                        file=str(path.relative_to(ROOT)), sha256=digest, size=list(size), glyphs=row['glyphs'],
                        stopped=row['stopped'], capped=row['capped'], displayed_glyphs=min(10, row['glyphs']),
                        all_pixels_embedded=True, display_is_prefix_excerpt=row['glyphs'] > 10))
    return dict(copy_qualification=copy, c3_summary=c3['summary'], feedback=feedback['groups'],
        helper_report=helper, preflight_fit=preflight, fixed_documents=list(DOCUMENTS), cells=cells,
        source_files_sha256=bound, full_C3_streams=128, neural_forward_calls=0,
        runtime_outputs_repaired=False, new_tail_language_results_included=False,
        source_collector_sha256=sha(__file__), visual_review_receipt_sha256=sha(
            EVIDENCE/'c3_glyph_language_assessment_v1/review_manual_review.json'))


if __name__ == '__main__':
    print(json.dumps(collect(), indent=2))
