"""Deployment admission without obsolete-job JSON dependencies."""
from contextlib import contextmanager
import fcntl
import os
import shutil
import subprocess

from ilm.image_edit.baseline import ROOT
from scripts.train_semantic_glyph_bridge import project_processes
from scripts.score_pixel_language_first_blocks import host_resources, require


@contextmanager
def deployment_bridge_gpu(prepared):
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='0', 'physical GPU0 only')
    with (ROOT/'artifacts/ilm_gpu0.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        entry=host_resources()
        require(entry['available_gib']>=prepared['minimum_entry_ram_gib'], 'bridge RAM entry reserve unavailable')
        require(entry['some_pressure_avg10']<=.5 and entry['full_pressure_avg10']<=.1,
                'bridge memory pressure entry limit')
        inventory=project_processes()
        obsolete=[p for p in inventory if any(s in p['command'] for s in
                  ('Xvfb ', 'x11vnc ', 'websockify ', 'train_pixel_language_diversity.py --stage run'))]
        require(not obsolete, 'inspect project runtime before bridge admission')
        free=int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free',
                                         '--format=csv,noheader,nounits'],text=True).strip())
        require(free>=20*1024,'another GPU0 owner may be active')
        require(shutil.disk_usage(ROOT).free >= 48*1024**3, 'deployment disk reserve unavailable')
        import torch
        torch.set_num_threads(4)
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats()
        yield torch.device('cuda:0'),dict(entry=entry,project_processes=inventory,
            mode='separate_small_bridge_training_RAM_and_pressure_protocol',
            original_training_admission_unchanged=True,foreign_processes_changed=False,
            high_global_swap_occupancy_recorded=entry['swap_fraction']>.75)


@contextmanager
def full_reader_gpu():
    # Keep the existing mutex, low-pressure entry check, project-process check
    # and 20-GiB free-device requirement. The measured full-reader preflight
    # ran with >=49 GiB host RAM available and <=17.724 GiB allocated GPU RAM.
    # Its independent recovery admission uses 48 GiB, rather than borrowing
    # the old read-only diagnostic's 64-GiB post-swap reserve.
    with deployment_bridge_gpu(dict(minimum_entry_ram_gib=48.)) as (device, admission):
        yield device, dict(admission, mode='glyph_full_reader_48GiB_entry_24GiB_runtime',
            minimum_entry_available_gib=48., minimum_runtime_available_gib=24.,
            maximum_allocated_vram_gib=22., recovery_resource_admission_amendment=True,
            original_parent_admission_functions_unchanged=True)
