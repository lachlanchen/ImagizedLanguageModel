# Autonomous evaluation of the conditional joint writer

The full objective remains useful independent image-input/image-output language,
with whole-model resource accounting and later extension to historical writing.
A physical form gate cannot establish it. This execution path implements the
708 new candidate streams already fixed by the
[joint-writer protocol](continuation_joint_writer_protocol.md); it changes no
training source, model, corpus, threshold or prescribed comparison.

`scripts/evaluate_joint_writer_language.py` prepares the complete case plan on
CPU, binds the actual fitted snapshots after physical qualification, and runs
the existing visual-continuation generator with `JointGlyphBridge`. Every
snapshot uses the **same final 41,298-update frozen reader**, continuation
dynamics, query, feedback encoder, affine statistics and stop head. The five
steps here measure new joint-head updates, rather than the earlier reader's
learning trajectory. Head step zero includes training-only normalization but
no head optimizer updates. No trained or language-quality result is implied by
this implementation.

## Fixed coverage and generation

| Head updates | Core streams, cap 128 | Book streams, cap 640 |
| --- | ---: | ---: |
| 0 | 216 | 24 |
| 10,324 | 76 | 0 |
| 20,649 | 76 | 0 |
| 30,973 | 76 | 0 |
| 41,298 | 216 | 24 |

The 660 core and 48 long streams preserve all 32 QA cases in each layout,
sixteen paired passage groups, question/passage erasures in both layouts and
all twelve normal/erased fixed-layout book prefixes. Interior snapshots retain
all normal cases. No intermediate checkpoint replaces the prescribed final
one. Reuse the complete original point-writer control and frozen scores; do
not spend another GPU run regenerating that control.

The generator predicts every physical mask greedily, rereads the actual emitted
pixels and uses the unchanged learned stop decision at probability 0.5. It
receives neither reference answers nor answer lengths. There is no deployed
tokenizer, lexical output head, font lookup, OCR, retrieval or live teacher.
Existing raw-stream, feedback, display-raster and stop/cap records remain
unaltered. Each case is saved atomically and authenticated before reuse after
an explicit manual resume. All outputs require subsequent complete opaque
review; generation alone assigns no language score.

## Order of execution

The `plan` action is read-only and needs neither Torch nor model weights:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/evaluate_joint_writer_language.py plan
```

Do not run the next actions while the original pipeline is live or before its
physical review passes. The
[freeze-before-unmask workflow](joint_writer_physical_review_workflow.md) must
finish first. Registration authenticates its sealed readings, independently
recounts all physical groups and rejects a missing or failed gate before any
neural import or evaluation-directory creation:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python scripts/evaluate_joint_writer_language.py register
```

Commit and push the resulting
`references/evidence/continuation_joint_writer_language_registration_v1.json`
before generation. It binds the original joint registration, all fixed head
weights, actual final reader, source code, prompts, physical verdict and reused
control. The private copy and outputs reside in
`artifacts/continuation_joint_writer_v1/language/`.

Then launch one durable evaluator, after checking existing ownership/resources:

```bash
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=. \
  python -u scripts/evaluate_joint_writer_language.py generate
```

The evaluator uses a separate owner lock and the existing exclusive GPU0 mutex.
Prelaunch checks retain a 48-GiB host entry threshold plus 2 GiB headroom and
20 GiB free VRAM. Runtime retains 24 GiB host memory and at most 22 GiB allocated
VRAM. The registered eight-hour cumulative ceiling includes neural imports,
admission waiting and generation. At most 1,800 seconds of admission/runtime
pressure waiting are allowed within it. Other projects and GPU1 remain intact.
An error stops execution. Only explicit `--resume` can reuse authenticated
finished cases, with every prior attempt's time and pressure allowance deducted.
An unterminated attempt must be preserved and reconciled before recovery;
there is no automatic retry or reset. Parameter/buffer version checks detect
mutation during inference. Report actual complete deployed parameter elements,
head size, VRAM, wall time and inherited preparation separately from quality.

## Acceptance and remaining validation

Final proportional QA still requires at least 12/32 exact, 4/16 both-correct
pairs and 24/32 readable complete answers, plus at least 4/12 coherent complete
long-book continuations. Retain all eight required proportional and both fixed
exact cases. Each of the four QA-erasure arms must have at most two exact
answers, with a proportional normal-minus-better-erased gap of at least eight.
Freeze all complete-output readings before new ownership/reference scoring;
retain every stop, cap and uncertain reading. Untouched source-group
confirmation and standalone execution/resource evidence remain necessary after
any development pass. Neither clean conditional forms nor a completed generator
establishes usable language, lightweight foundation training or LLM parity.

The five tests in `tests/test_joint_writer_language_evaluation.py` inspect the
real registered cohort's full coverage and reject missing controls, premature
qualification, wrong head/reader provenance and reset or unknown attempt costs.
They validate planning and execution invariants only. Actual GPU integration,
generated streams, complete output review and language scores remain unmeasured
until the corresponding stages execute. Do not replace those missing results
with zero scores or test-fixture outcomes.
