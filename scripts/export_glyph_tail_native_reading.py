#!/usr/bin/env python3
"""Export complete, authenticated helper-only reading evidence without private runtime inventories."""
from pathlib import Path
import json

from ilm.image_edit.baseline import ROOT,save_json,sha


def main():
    source=ROOT/'artifacts/glyph_tail_native_reading_learning_v1/summary.json'
    value=json.loads(source.read_text())
    if (not value['all448_cases_authenticated'] or value['actual_adapted_image_prefills_bit_exact']!=192
            or not value['raw_IDs_text_states_stops_verified'] or len(value['records'])!=448
            or value['new_optimizer_updates'] or value['ILM_image_answers_generated']):
        raise ValueError('Complete helper-only evidence required')
    public=ROOT/'references/evidence'
    registration=public/'glyph_tail_native_reader_registration_20261111.json'
    if sha(registration)!=value['registration_sha256']:
        raise ValueError('Published registration differs')
    value.update(original_analysis_sha256=sha(source),export_source_sha256=sha(Path(__file__)),
        public_registration_sha256=sha(registration),
        CPU_analysis_receipt_schema_recovery='The first CPU analysis stopped before saving scores: the earlier fixed-cell receipt binds review/scores.json inside bindings, whereas the proportional receipt uses scores_sha256. The retry authenticates those same original score hashes. No generation, canonical reference, denominator or scoring rule changed.',
        all448_raw_source_IDs_logits_states_retained_locally=True,
        public_records_contain_unchanged_decoded_text=True,
        inherited_lexical_decoder_is_helper_only=True,
        one_shared_body_used_serially=True,matched_quality_efficiency_claim=False)
    encoded=json.dumps(value,ensure_ascii=False)
    if any(x in encoded for x in ['/home/','boot_id','"pid"','project_processes']):
        raise ValueError('Private runtime metadata in public evidence')
    path=public/'glyph_tail_native_reader_learning_20261111.json';save_json(path,value)
    print(json.dumps(dict(path=str(path.relative_to(ROOT)),cases=448,
        actual_prefills_verified=192,new_ILM_images=0,goal_achieved=False)))


if __name__=='__main__':main()
