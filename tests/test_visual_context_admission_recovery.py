import json
from pathlib import Path
import tempfile
import unittest
from scripts.resume_visual_context_comparison import FAILURE, admission_ready, failed_admission_evidence, stages


class AdmissionRecoveryTests(unittest.TestCase):
    def test_complete_schedule_skips_only_completed_treatment_work(self):
        reg=dict(run='artifacts/comparison',evaluation='artifacts/evaluation',snapshot_updates=[0,1956,4049,6152,8172])
        plan=stages(reg,5.25);names=[s['name'] for s in plan]
        self.assertEqual(len(plan),17)
        self.assertNotIn('trainable_fit',names);self.assertNotIn('trainable_audit',names)
        self.assertEqual(names.count('frozen_fit'),1);self.assertEqual(names.count('frozen_audit'),1)
        self.assertEqual(names[-1],'prepare_review')
        for arm in ('trainable','frozen'):
            generates=[s for s in plan if s['name'].startswith(arm+'_generate_')]
            self.assertEqual(len(generates),5)
            self.assertIn(arm+'_long_books',names);self.assertIn(arm+'_cold_replay',names)
            profiles=[s['arguments'][s['arguments'].index('--profile')+1] for s in generates]
            self.assertEqual(profiles,['all','normal','normal','normal','all'])
            self.assertEqual(sum(120 if p=='all' else 44 for p in profiles)+24,396)
        self.assertTrue(plan[0]['output'].endswith('_admission_recovery1'))
        self.assertEqual(plan[0]['output'].split('/')[-2],'trainable')
        self.assertEqual(float(plan[0]['arguments'][-1])+5.25,7200.)
        outputs=[s['output'] for s in plan if s['output']]
        self.assertEqual(len(outputs),len(set(outputs)))

    def test_unchanged_pressure_and_resource_boundaries(self):
        settings=dict(host_available_gib=48.,gpu0_free_mib=20480,disk_free_gib=48.,max_some_pressure_avg10=.5,max_full_pressure_avg10=.1)
        good=dict(available_gib=48.,gpu0_free_mib=20480,disk_free_gib=48.,some_pressure_avg10=.5,full_pressure_avg10=.1,mutex_available=True)
        self.assertTrue(admission_ready(good,settings))
        for key,bad in [('available_gib',47.999),('gpu0_free_mib',20479),('disk_free_gib',47.999),
                        ('some_pressure_avg10',.501),('full_pressure_avg10',.101),('mutex_available',False)]:
            self.assertFalse(admission_ready(dict(good,**{key:bad}),settings),key)

    def test_outputs_or_different_failure_forbid_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);value=dict(phase='failed_visual_context_inference',error=FAILURE,position={},automatic_retry=False,elapsed_seconds=5.25)
            (folder/'status.json').write_text(json.dumps(value));(folder/'request.json').write_text('{}')
            self.assertEqual(failed_admission_evidence(folder),value)
            (folder/'partial_answer.png').write_bytes(b'preserve')
            with self.assertRaises(ValueError):failed_admission_evidence(folder)
            (folder/'partial_answer.png').unlink();value['error']='a model error';(folder/'status.json').write_text(json.dumps(value))
            with self.assertRaises(ValueError):failed_admission_evidence(folder)


if __name__=='__main__':unittest.main()
