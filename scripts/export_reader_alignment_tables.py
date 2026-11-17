#!/usr/bin/env python3
"""Source-bound figure tables for coordinate fitting, not language scores."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'references/evidence/reader_alignment_20261118'


def main():
    sources=[ROOT/'references/evidence/nonlinear_feedback_alignment_result_20261118.json',
             ROOT/'references/evidence/feedback_ridge_refit_result_20261118.json']
    fit,ridge=[json.loads(p.read_text()) for p in sources]
    assert not fit['language_measured'] and not ridge['language_measured']
    assert fit['source_update_rows']==ridge['complete_rows']==33941
    OUT.mkdir(exist_ok=False)
    with (OUT/'coordinate_fit.csv').open('x',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['arm','stage','epoch','native_relative_error','fitting_rows','language_measured'])
        for arm in fit['arms']:
            writer.writerow([arm['arm'],'initial',0,fit['initial']['weighted_error'],33941,False])
            for row in arm['history']:
                writer.writerow([arm['arm'],'adamw',row['epoch'],row['weighted_error'],33941,False])
        for arm in ridge['arms']:
            writer.writerow([arm['arm'],'fixed_feature_ridge','',arm['actual_weighted_error'],33941,False])
    with (OUT/'ridge_precision.csv').open('x',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['arm','actual_bf16_error','ideal_matrix_error','book_error','qa_error','long_qa_error','language_measured'])
        for arm in ridge['arms']:
            writer.writerow([arm['arm'],arm['actual_weighted_error'],arm['solve']['ideal_weighted_native_error'],
                *[v['relative_error'] for v in arm['complete_training_cases']],False])
    bound=[*sources,Path(__file__),OUT/'coordinate_fit.csv',OUT/'ridge_precision.csv']
    receipt=dict(measurement='Weighted native-coordinate fitting error on training support; not glyph F1 or language accuracy.',
        rows=33941,language_measured=False,goal_achieved=False,
        bindings={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in bound})
    with (OUT/'provenance.json').open('x') as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(output=str(OUT.relative_to(ROOT)),coordinate_curve_rows=44,ridge_precision_rows=2,language_measured=False)))


if __name__=='__main__':main()
