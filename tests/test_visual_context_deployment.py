from contextlib import ExitStack
import ast
import fcntl
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.verify_visual_context_deployment_sources import expected_sources,verify_sources,verify_text


class DeploymentPlanTests(unittest.TestCase):
    def test_only_fresh_cold_and_original_remaining_control_work(self):
        from scripts.continue_visual_context_deployment import stages,treatment_folders
        reg=dict(run='artifacts/comparison',evaluation='artifacts/evaluation',snapshot_updates=[0,1956,4049,6152,8172])
        plan=stages(reg);names=[v['name'] for v in plan]
        self.assertEqual(len(plan),11);self.assertEqual(names[0],'trainable_cold');self.assertEqual(names[1],'frozen_fit')
        self.assertEqual(names.count('frozen_fit'),1);self.assertEqual(names[-1],'prepare_review')
        self.assertFalse(any('trainable_generate' in n or n=='trainable_fit' or n=='trainable_audit' for n in names))
        self.assertEqual(len(treatment_folders(reg)),6)
        outputs=[s['output'] for s in plan if s['output']]
        self.assertEqual(len(outputs),len(set(outputs)))
        self.assertEqual(sum(s['name'].endswith('_cold') for s in plan),2)
        self.assertTrue(all(s['output'].endswith('/cold_deployment_v1') for s in plan if s['name'].endswith('_cold')))
        generated=[s for s in plan if s['name'].startswith('frozen_generate_')]
        profiles=[s['arguments'][s['arguments'].index('--profile')+1] for s in generated]
        self.assertEqual(profiles,['all','normal','normal','normal','all'])
        self.assertTrue(all(s['arguments'][0]=='scripts/infer_visual_context_deployment.py' for s in generated))
        self.assertEqual(sum(120 if p=='all' else 44 for p in profiles)+24+396,792)
        long,=[s for s in plan if s['name']=='frozen_long_books'];self.assertEqual(long['arguments'][-1],'640')

    def test_cold_verifier_uses_the_original_strict_audit(self):
        from scripts.verify_visual_context_deployment_sources import ROOT
        source=(ROOT/'scripts/verify_visual_context_deployment_cold.py').read_text();tree=ast.parse(source)
        self.assertTrue(any(isinstance(n,ast.ImportFrom) and n.module=='scripts.verify_full_rank_visual_span_cold'
            and {a.name for a in n.names}=={'audit_access','compare_outputs'} for n in ast.walk(tree)))
        self.assertIn('access=audit_access(output,working,prompt,checkpoint)',source)
        self.assertNotIn('pixel_language_diversity_training_v2',source)

    def test_original_strict_audit_still_rejects_old_runtime_metadata(self):
        from scripts.verify_full_rank_visual_span_cold import audit_access,CODEC,ROOT
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory);working=out/'empty';working.mkdir();prompt=out/'prompt.png';checkpoint=out/'model.pt'
            paths=[prompt,checkpoint,CODEC.resolve()]
            def trace(extra=()):
                (out/'access.fixture').write_text(''.join(f'openat(AT_FDCWD<{working}>, "{p}", O_RDONLY) = 3<{p}>\n' for p in [*paths,*extra]))
            trace();self.assertFalse(audit_access(out,working,prompt,checkpoint)['other_project_artifacts_read'])
            for name in ['status.json','events.jsonl']:
                trace([ROOT/'artifacts/pixel_language_diversity_training_v2/execution'/name])
                with self.assertRaises(ValueError):audit_access(out,working,prompt,checkpoint)


class DeploymentSourceTests(unittest.TestCase):
    def test_actual_sources_have_only_registered_admission_delta(self):
        self.assertTrue(verify_sources()['numerical_inference_source_identical_except_admission_import'])

    def test_generation_and_guard_changes_are_rejected(self):
        inference,guard=expected_sources()
        with self.assertRaises(ValueError):verify_text(inference.replace('default=8)','default=9)'),guard)
        with self.assertRaises(ValueError):verify_text(inference,guard.replace('minimum_entry_ram_gib=48.','minimum_entry_ram_gib=32.'))
        with self.assertRaises(ValueError):verify_text(inference,guard+'\nopen("old_status.json")\n')


class DeploymentGuardTests(unittest.TestCase):
    def invoke(self,changes=None,free=20480,disk=48.,inventory=None,locked=False):
        from ilm.visual_lm import visual_context_deployment_resources as guard
        calls=[]
        fake=SimpleNamespace(set_num_threads=lambda n:calls.append(('threads',n)),
            cuda=SimpleNamespace(set_device=lambda n:calls.append(('device',n)),
                                 reset_peak_memory_stats=lambda:calls.append(('reset_peak',))),
            device=lambda name:name)
        entry=dict(available_gib=48.,some_pressure_avg10=.5,full_pressure_avg10=.1,swap_fraction=.8)
        entry.update(changes or {})
        with tempfile.TemporaryDirectory() as directory,ExitStack() as stack:
            root=Path(directory);(root/'artifacts').mkdir()
            if locked:
                lock=stack.enter_context((root/'artifacts/ilm_gpu0.lock').open('a'))
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            stack.enter_context(patch.object(guard,'ROOT',root))
            stack.enter_context(patch.object(guard,'host_resources',return_value=entry))
            stack.enter_context(patch.object(guard,'project_processes',return_value=inventory or []))
            stack.enter_context(patch.object(guard.subprocess,'check_output',return_value=str(free)))
            stack.enter_context(patch.object(guard.shutil,'disk_usage',return_value=SimpleNamespace(free=disk*2**30)))
            stack.enter_context(patch.dict(os.environ,CUDA_VISIBLE_DEVICES='0'))
            stack.enter_context(patch.dict(sys.modules,torch=fake))
            stack.enter_context(patch('scripts.train_semantic_glyph_bridge.owner_gone',side_effect=AssertionError('Obsolete owner metadata must not be consulted')))
            with guard.full_reader_gpu() as (device,receipt):
                self.assertEqual(device,'cuda:0')
                self.assertEqual(receipt['minimum_entry_available_gib'],48.)
                self.assertEqual(receipt['minimum_runtime_available_gib'],24.)
                self.assertEqual(receipt['maximum_allocated_vram_gib'],22.)
                self.assertTrue(receipt['high_global_swap_occupancy_recorded'])
            self.assertEqual(calls,[('threads',4),('device',0),('reset_peak',)])

    def test_exact_resource_boundaries_and_no_old_metadata_dependency(self):
        self.invoke()

    def test_ram_and_pressure_rejections(self):
        for change in [dict(available_gib=47.999),dict(some_pressure_avg10=.501),dict(full_pressure_avg10=.101)]:
            with self.subTest(change=change),self.assertRaises(RuntimeError):self.invoke(change)

    def test_gpu_disk_and_obsolete_process_rejections(self):
        with self.assertRaises(RuntimeError):self.invoke(free=20479)
        with self.assertRaises(RuntimeError):self.invoke(disk=47.999)
        for command in ['Xvfb :99','x11vnc -display :99','websockify 6080','train_pixel_language_diversity.py --stage run']:
            with self.subTest(command=command),self.assertRaises(RuntimeError):self.invoke(inventory=[dict(command=command)])

    def test_real_mutex_contention_is_rejected(self):
        with self.assertRaises(BlockingIOError):self.invoke(locked=True)


if __name__=='__main__':unittest.main()
