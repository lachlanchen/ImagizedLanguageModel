# White padding alone breaks the calibrated reader

With the same completed dense checkpoint, all copying ink and coordinates,
generic request and latent/readout seeds fixed, adding white bottom padding
changes copying from **29/32 to 7/32 exact stopped answers**. Chinese falls
from **13/16 to 0/16**; English falls from **16/16 to 7/16**. This is a
controlled geometry failure. It is not a comparison between two differently
trained models.

| Fixed checkpoint, complete request | Original copying strip | White-padded square | Tight whole-request crop |
| --- | ---: | ---: | ---: |
| Chinese copying /16 | 13 | 0 | 0 |
| English copying /16 | 16 | 7 | 0 |
| Chinese QA /32 | 0 | 0 | 0 |
| English QA /32 | 0 | 0 | 0 |
| Copying responses reaching EOS /32 | 32 | 22 | 32 |

The original copying canvas is 1024 by 256 with 1,024 visual positions;
white padding makes it 1024 square with 4,096. Restoring the removed exterior
to every tight crop also reconstructs its complete original square exactly.
Tight copying inputs contain 161--304 visual positions, but give no correct
copies. Preserving all visible marks does not preserve this model's latent
features, interpolated positions or reading behavior. The experiment changes
these jointly and does not isolate a unique internal cause.

The first Chinese reference is `树田红星风`. The original strip gives that
exact answer; its padded version gives `林石三石林`, and its tight version
gives `1990`. The first English reference is `garden silver open blue`;
responses are respectively exact, `green silver silver` and `1990年`.
Other padded copies reach the cap through repeated characters. These are
substantive errors, not only formatting failures. No output is repaired.

All **480** primary calls are complete. Before changed layouts are evaluated,
the original-layout arm replays all **160** previously retained decision
sequences exactly, including the full Chinese and English erasure controls.
The [prospective protocol](native_canvas_control_protocol.md) fixes all arms,
whole-request cropping, seeds, unchanged 96-decision/EOS scoring and the
absence of optimization. No OCR, transcript, target or semantic role selects
a crop; every non-white input pixel and the surrounding 16-pixel margin is
preserved. QA pages remain identical between the legacy and square arms.

The [complete export](evidence/native_canvas_control_result/result.json)
binds 487 files, including all 480 input PNGs, all raw responses and exact
reconstruction checks. The three 160-call arms take 86.856, 120.337 and
64.060 seconds excluding loading, with allocated peaks of 11.161, 11.161
and 10.506 GiB on one RTX 4090 D. Faster failed reading is not an efficiency
advantage at useful language quality.

The complete learned source has 2,839,267,040 parameters, including its
8,489,984 visual-adapter parameters, plus a 126,892,531-parameter codec and
external pretraining. This diagnostic uses its temporary Unicode head and
produces no continuous image answers.

## Consequence for the next learning intervention

The earlier 13/16 Chinese-copy result cannot be compared with the later
square/coarse learner as if both used identical geometry. A calibrated
visual frontend already supports partial Chinese identity in its original
layout, while QA remains zero. The next
[attention-learning protocol](native_reader_attention_protocol.md) preserves
that exact source and geometry, freezes the visual features and trains only
new contextual-attention residuals. That choice precedes these canvas outcomes.
The tight crop is rejected as a demonstrated inference repair. No new
learning result, image-answer qualification or full ILM completion follows.

Reproduction uses fresh output paths:

```bash
env CUDA_VISIBLE_DEVICES= PYTHONPATH=. python scripts/probe_native_canvas.py --stage prepare --output artifacts/native_canvas_control/prepared_v1
env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. artifacts/external/native_unified_source/runtime/bin/python scripts/probe_native_canvas.py --stage evaluate --output artifacts/native_canvas_control/assessment_v1
env CUDA_VISIBLE_DEVICES= PYTHONPATH=. python scripts/probe_native_canvas.py --stage score --output references/evidence/native_canvas_control_result
```

CMRC 2018/SQuAD 1.1 retain CC-BY-SA-4.0 attribution to Cui et al. (2019) and
Rajpurkar et al. (2016); Show-o2 source/model are NUS Show Lab, Apache-2.0.
