# Learn a fine-image correction to the native reader

This is a complete runnable **reader-learning experiment**, with a fixed
protocol and autonomous assessment. It is one prerequisite of the independent
image-input/image-output ILM. Its temporary native Unicode head does not
provide image answers. The [protocol](../references/native_state_alignment_protocol.md)
defines the model, data, objective, optimizer and six quality gates.

The frozen low-resolution source retains its measured initial behavior.
A zero-initialized full-rank linear bridge adds a correction derived from the
original high-resolution page. Only that bridge and fine-view visual/fusion
adapters learn: 10,849,280 parameters. The complete adapted model has
2,841,626,336 parameters plus the 126,892,531-parameter Wan codec and inherited
external pretraining. This is transfer learning, not foundation training from
scratch on one GPU.

The main losses are answer cross entropy and mean squared normalized
pre-answer state error at seven frozen language-body layers. Original-text
states are prepared offline from the same source body, without an answer
history. They are absent during autonomous assessment. Preserving the original
low-view features at zero correction provides a concrete initial invariant;
lower state loss is not proof of language understanding.

## Prepare and execute

Reuse the existing [source setup](native-visual-access.md), CMRC preparation
and pinned SQuAD source. No extra model weights or services are required.
Preparation uses all 384 original training IDs and 256 English questions
from 128 distinct paragraphs in 32 other articles. All sixteen English
assessment articles are excluded. Full source/reference overflow and missing
font coverage are logged; teacher outputs do not select training data.

```bash
export ILM_NATIVE_PYTHON="$PWD/artifacts/external/native_unified_source/runtime/bin/python"
env CUDA_VISIBLE_DEVICES= PYTHONPATH=. python scripts/prepare_native_state_alignment.py --output artifacts/native_state_alignment/prepared_v1
env CUDA_VISIBLE_DEVICES= PYTHONPATH=. python scripts/prepare_native_state_assessment.py --output artifacts/native_state_alignment/assessment_v1
env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. "$ILM_NATIVE_PYTHON" scripts/train_native_state_alignment.py --stage cache --output artifacts/native_state_alignment/cache_v1
env CUDA_VISIBLE_DEVICES= PYTHONPATH=. "$ILM_NATIVE_PYTHON" scripts/run_native_state_alignment.py
```

The controller reuses only completed stages with matching code bindings. It
records actual child PIDs and waits for terminal exit before starting a
successor. It runs three preflight updates, the complete 32-copy source
baseline, 2,560 fresh main-run updates, all 160 autonomous after-readouts,
an additional fresh-source zero-correction replay and CPU scoring. An
incomplete/live stage is preserved and stops a new launch;
attach to that process rather than launching another controller. Every
four-pass training ID appears exactly four times. No validation-driven early
stopping or best-checkpoint selection occurs.

The completed cache contains 640 detached text-state targets and 1,280 native
image encodings, prepared in 238.713 seconds. The 160-condition assessment
retains full source pixels; all 128 Chinese/English native low-view hashes
match their existing source baselines exactly. Cache preparation and image
view equality are execution/data evidence, not a language result.

The [completed execution export](../references/evidence/native_state_alignment_preflight/result.json)
retains the real three-update preflight, all 317 finite/nonzero gradients on
the third update and the exact source copying baseline. The FP32 normalized
history-control error is 7.6294e-6, below its declared 2e-5 tolerance. All 159
initially zero fine-up/bridge tensors change. The source path remains exact
when the fine adapters/correction are disabled. The reported 6.388 GiB peak
covers the three optimizer updates after the memory counter reset; it does
not measure the earlier temporary FP32 control's peak.

The [protocol deviation](../references/native_state_alignment_protocol.md#execution-deviation-and-additional-source-check)
records that the initial worker replayed four native readouts but compared
the corrected prefix/mask only on its training probe. The additional complete
four-case corrected-path check is required before final scoring. The current
controller was launched before that extra stage was added: after it is
terminal, rerun the updated controller to reuse completed stages and execute
the missing source check and final scorer. Never launch a successor while
the existing training or assessment child is live.

Inspect `artifacts/native_state_alignment/controller_status.json` and the
stage logs. Numeric weights, optimizer/cache files and private runtime handoff
remain outside Git. Stage output paths must be fresh.

## Interrupted training

Use the wrapper below rather than the learner's low-level `--resume` switch.
The wrapper copies the authenticated journal through the checkpoint's saved
update, preserves any uncheckpointed tail in the original directory, and
continues the same numeric optimizer/RNG state into a fresh path:

```bash
env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. "$ILM_NATIVE_PYTHON" scripts/resume_native_state_alignment.py --resume artifacts/native_state_alignment/learning_v1/checkpoint.pt --output artifacts/native_state_alignment/learning_resumed_v1
```

Never restart a live job because a tool observation timed out. The controller
also accepts explicit `--resume`, `--training-output`, `--after-output` and
`--score-output` paths after the previous process is verified terminal. Keep
the implementation, data and schedule unchanged. The wrapper's interrupted
main-run path has not yet been exercised by an actual training interruption.

## Assess the actual answers

The complete assessment is 32 held-out copies, 48 Chinese paired conditions
and 80 English normal/erasure conditions. It uses fresh high/low images and
numeric model state, without original transcripts, teacher states or correct
answer histories. The unchanged paired scorers retain all alternatives,
punctuation/case rules and EOS requirements.

The final scorer verifies all 2,560 optimizer records, four passes per ID,
every parameter's recorded finite gradients, the preflight causal/source
controls and every before/after input/seed binding. The public export includes
a compact per-update journal; its complete per-parameter raw gradient journal
is retained locally with SHA256. No failed response is repaired or replaced.

Passing all six reader gates would admit a separate continuous image-writing
experiment. It cannot complete the ILM goal, establish historical-form
capability or demonstrate efficiency relative to a conventional LLM. There is
no completed learning-quality result from this experiment yet.
