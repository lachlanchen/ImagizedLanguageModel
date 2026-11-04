# Training the connection between questions and writing

V63 trains the conditional likelihood of complete answer images. Two questions
about the same paragraph supply two accepted answer images. Both the normal
and swapped combinations are evaluated. The assignment term rewards assigning
each image to its own question while a positive-likelihood term preserves
the writing distribution.

For mean answer-pixel energies `eij = NLL(answer j | prompt i)`,

```text
margin = e01 + e10 - e00 - e11
loss = (e00 + e11)/2 + weight * 0.01*softplus(-margin/0.01)
```

The aligned arm uses weight one and the control zero. Both have identical
weights at initialization, parameter count, data order, optimizer schedule and
four likelihood graphs per pair. Question-only and answer-only offsets cancel
from the margin. That property does not guarantee semantic learning, so actual
free image generation and question controls determine the result.

The [registered protocol](../references/question_assignment_v63_protocol.md)
acknowledges earlier failed assignment-critic and pooled semantic-distillation
experiments. This test applies the loss directly to an existing full image
generator; it adds no candidate bank, Unicode IDs, teacher service or OCR to
inference. External MIXAR foundation pretraining is still attributed separately.

The [real preflight](../references/question_assignment_v63_preflight_result.md)
passes physical data assembly, assignment gradients, full-model memory,
exact optimizer/RNG resume and future-pixel causality. Its longest training
pair fits in 9.20 GiB. Production quality is not established by that result.

Run from the repository root using the existing Python 3.12 runtime:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/prepare_question_assignment_v63.py
CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=. \
  artifacts/external/mixar_env_py312/bin/python scripts/check_question_assignment_v63.py
CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONPATH=. \
  artifacts/external/mixar_env_py312/bin/python scripts/run_question_assignment_v63.py
```

Preparation and completed preflights are immutable: reuse existing receipts
instead of rerunning those commands into the same directory. The coordinator
trains control then aligned for 2,000 updates each, exports and replays both
standalone actors, and runs the locked audit only after both finish. It waits
when GPU 0 is occupied or host resources exceed the shared-workstation limits.
Its lock prevents duplicate coordinators. Interrupted training resumes exact
optimizer/RNG state; journal rows after the durable checkpoint are retained in
an explicit uncommitted-tail archive before replaying those updates.

Artifacts live under `artifacts/question_assignment_v63_20261105/`.
`pipeline_status.json` records the actual coordinator/child PIDs and stage;
each arm has a console, metrics, pair order and resumable checkpoint. Verify
those PIDs or the owned tmux session before treating a job as live. A zero
process exit without the final report is an interruption.

After production completes, image inference uses the unchanged independent
actor interface:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
  artifacts/external/mixar_env_py312/bin/python scripts/infer_correlated_native_language.py \
  --checkpoint artifacts/question_assignment_v63_20261105/aligned/model_final.pt \
  --prompt-image prompt.png --out answer.png
```

Typed prompts are rendered outside the actor. The deployed model receives
physical image tensors and emits new image tensors. The training assignment
helper is absent from deployment. Ranking retained references remains a
separate diagnostic; it cannot substitute for correct autonomous writing.
The sealed final partitions stay unopened and general language remains the
full objective.
