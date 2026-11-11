"""Protocol aggregate checks, never a model or language-quality demonstration."""
import copy
import json
from pathlib import Path
import pytest

from scripts.audit_visual_continuation_protocol import pilot_checks


def fixture():
    r=json.loads((Path(__file__).resolve().parents[1]/'references/evidence/visual_continuation_full_logger_recovery_registration_20261112.json').read_text())
    final=r['total_steps'];normal=dict(step=final,kind='qa',condition='normal',cases=32,
        complete_exact=12,paired_both_correct=4,readable=24)
    erased=[dict(normal,condition=c,complete_exact=0,paired_both_correct=0) for c in ['question_blank','passage_blank']]
    summary=dict(pilot_passed=True,core_results={layout:[copy.deepcopy(normal),*copy.deepcopy(erased)] for layout in ['fixed_cells','proportional']},
        long_book_results=[dict(arm='candidate_long640',step=final,condition='normal',cases=12,coherent_complete=4)],
        retention={k:dict(required=v,retained=v.copy()) for k,v in r['exact_retention'].items()})
    return r,summary


def test_complete_literal_registered_gates_pass():
    r,s=fixture();a=pilot_checks(r,s)
    assert a['registered_protocol_pilot_passed'] and len(a['all_four_QA_erasure_arms'])==4
    assert a['goal_achieved'] is False and a['untouched_confirmation_required']


def test_fixed_erasure_is_not_silently_omitted():
    r,s=fixture();s['core_results']['fixed_cells'][1]['complete_exact']=3
    a=pilot_checks(r,s)
    assert s['pilot_passed'] is True  # Legacy expression only sees proportional erasures.
    assert not a['checks']['each_QA_erasure_arm'] and not a['registered_protocol_pilot_passed']


def test_missing_erasure_and_relabelled_kind_fail_closed():
    r,s=fixture();s['core_results']['fixed_cells'].pop()
    with pytest.raises(ValueError):pilot_checks(r,s)
    r,s=fixture();s['core_results']['proportional'][0]['kind']='book'
    with pytest.raises(ValueError):pilot_checks(r,s)


def test_no_rescue_or_threshold_relaxation():
    r,s=fixture();s['pilot_passed']=False
    with pytest.raises(ValueError):pilot_checks(r,s)
    r,s=fixture();s['core_results']['proportional'][0]['complete_exact']=11
    assert not pilot_checks(r,s)['registered_protocol_pilot_passed']
