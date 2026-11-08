# Complete native visual flow assessment

The training protocol already fixes final update 1536, four assessment cohorts,
matched source/final input pixels and seeds, and six developmental requirements.
This coordinator executes that complete assessment after the actual training
owner exits successfully. It changes no training objective, model source,
sampler setting, checkpoint-selection rule or evaluation example.

Prepare the assessment while training runs, with CUDA hidden. Bind the model
implementation, independent assessor/coordinator, existing offline OCR
implementation, all eight input/reference files and four completed reference
calibrations. Freeze those sources before generating assessment outputs.
The source QA file supplies accepted alternatives to the offline assessor only.
Original article-disjoint partitions and reused-development limitations remain.

The coordinator waits for a specific training PID, start time and command.
Once that owner exits, require the completed 1536-update report and checkpoint.
Export final adapters on CPU. The generator then accepts only the separate
four-field input index: `id`, `input`, `input_sha256`, `seed`. No reference
manifest, correct answer or training cache enters generation.

Run the **final** model first, then the **initial source** control. Within
each arm, use normal, question-erased, passage-erased and copying cohorts in
that order. Cohort sizes are 64, 32, 32 and 64: 192 outputs per arm, 384 new
scheduled generations in total. Use 1024-square output, 50 native steps and
guidance 4 throughout. The source arm installs zero-output adapters of the
same graph and loads no learned adapter tensors. Native zero-adapter replay
has already passed. Preserve all original output pages and actual tensors.

Only one model child runs on GPU0 at a time through the shared project lock.
The parent is a lightweight CPU process. Score each completed cohort in a
separate CPU process, using its unchanged original reference calibration.
Retain every attempt log. Do not automatically restart, reroll noise, shrink
images, change thresholds or choose a checkpoint by quality. An explicit
unchanged coordinator resume may skip stages whose completion hashes still
match. An interrupted partial generation folder remains preserved; the
current inference CLI intentionally refuses to overwrite it. Resolve that
failure explicitly before another same-seed attempt, and report any repeated
calls and cost rather than concealing them in successful-stage totals.

The final offline assessor checks the complete optimizer/schedule state and
all 24 fixed training-preview draws, then authenticates each actual decoder
tensor and PNG using the source conversion `clamp(127.5*x+128,0,255).uint8`.
It reconstructs canonical/source-accepted OCR scores, both-correct passage
pairs and all six already registered developmental conditions. Identical
question-erased inputs and seeds must have bit-identical decoded pixels.
Inference source, deployment, input, seed, calibration and observation hashes
must agree. No manual transcription or output repair changes those scores.

Complete raw-image review remains a separate required step. OCR calibration
is imperfect, and a passing developmental rule alone would not establish
reliable writing, broad language competence, historical-script semantics or
efficiency at matched quality. Retained external pretraining and the complete
runtime parameter count remain part of the system's provenance. The overall
independent ILM goal is not qualified by this coordinator.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/run_native_visual_flow_assessment.py \
  --stage prepare --training-run artifacts/native_visual_flow_learning_v1 \
  --output artifacts/native_visual_flow_assessment_v1
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/run_native_visual_flow_assessment.py \
  --stage run --training-run artifacts/native_visual_flow_learning_v1 \
  --output artifacts/native_visual_flow_assessment_v1 --wait-for-training-pid <actual-pid>
```

The coordinator uses the verified existing model Python and NVIDIA wrapper;
it neither installs packages nor duplicates source weights. The standalone
assessor is `scripts/assess_native_visual_flow.py --training-run <run> --output
<assessment>`, with CUDA hidden. Registering this protocol supplies no new
autonomous-answer result.
