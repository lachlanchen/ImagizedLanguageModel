"""Explicit admission for the measured single-GPU glyph reader workload."""
from contextlib import contextmanager

from scripts.train_semantic_glyph_bridge import bridge_gpu


@contextmanager
def full_reader_gpu():
    # Keep the existing mutex, low-pressure entry check, project-process check
    # and 20-GiB free-device requirement. The measured full-reader preflight
    # ran with >=49 GiB host RAM available and <=17.724 GiB allocated GPU RAM.
    # Its independent recovery admission uses 48 GiB, rather than borrowing
    # the old read-only diagnostic's 64-GiB post-swap reserve.
    with bridge_gpu(dict(minimum_entry_ram_gib=48.)) as (device, admission):
        yield device, dict(admission, mode='glyph_full_reader_48GiB_entry_24GiB_runtime',
            minimum_entry_available_gib=48., minimum_runtime_available_gib=24.,
            maximum_allocated_vram_gib=22., recovery_resource_admission_amendment=True,
            original_parent_admission_functions_unchanged=True)
