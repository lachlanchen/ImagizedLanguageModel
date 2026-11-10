"""Bounded full-reader execution with unchanged physical admission thresholds."""
import time

from scripts.train_semantic_glyph_language import Monitor
from scripts.score_pixel_language_first_blocks import host_resources, require


class GlyphReaderMonitor(Monitor):
    pressure_wait_allowance_seconds = 1800.

    def guard(self):
        import torch
        require(torch.cuda.max_memory_allocated()/1024**3 <= 22., 'glyph reader allocation cap reached')
        while True:
            require(time.perf_counter()-self.began < self.seconds, 'glyph reader stage time cap reached')
            value = host_resources(); self.samples.append(value)
            require(value['available_gib'] >= 24., 'glyph reader RAM reserve reached')
            if value['some_pressure_avg10'] <= .5 and value['full_pressure_avg10'] <= .1:
                return
            remaining = min(self.pressure_wait_allowance_seconds-self.waited,
                            self.seconds-(time.perf_counter()-self.began))
            require(remaining > 0., 'glyph reader pressure wait exhausted')
            self.status('waiting_for_memory_pressure', resources=value)
            start = time.perf_counter(); time.sleep(min(10., remaining)); self.waited += time.perf_counter()-start

    def receipt(self):
        return dict(super().receipt(), pressure_wait_allowance_seconds=self.pressure_wait_allowance_seconds,
                    stage_deadline_checked_during_pressure_wait=True)
