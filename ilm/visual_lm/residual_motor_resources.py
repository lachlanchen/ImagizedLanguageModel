"""Admission sized for the physical motor, preserving shared runtime reserves."""
from contextlib import contextmanager

from .visual_context_deployment_resources import deployment_bridge_gpu


@contextmanager
def physical_motor_gpu():
    # Entry 40 = retained runtime reserve 24 + maximum process RSS 8 + margin 8.
    # The worker enforces the RSS/GPU limits and the original monitor retains
    # the 24-GiB live host reserve. No large contextual foundation is loaded.
    with deployment_bridge_gpu(dict(minimum_entry_ram_gib=40.)) as (device, admission):
        yield device, dict(admission, mode='physical_motor_only_40GiB_entry_24GiB_runtime',
                           minimum_entry_available_gib=40., minimum_runtime_available_gib=24.,
                           maximum_process_rss_gib=8., maximum_allocated_vram_gib=8.,
                           additional_entry_margin_gib=8.,
                           original_full_context_admission_unchanged=True)
