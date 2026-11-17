# Training the visual context

The [complete comparison](../references/trainable_visual_context_result_20261117.md)
has finished and rejects the tested recipe. Learned and frozen context each
produce 0/32 strict complete answers, 0/16 both-correct pairs, 0/32 readable
complete answers and 0/12 credited coherent complete long-book continuations.
All 792 streams are reviewed and readings were frozen before reference scoring.
Both full fit audits and both separate strict cold replays pass; these execution
checks do not establish useful language. Treatment improves natural long-book
stopping to 12/12, while writing remains unreadable.

The [failure decision](../references/trainable_visual_context_failure_decision_20261117.md)
selects a separately registered training-only accounting of conditional and
auxiliary gradients before choosing any further fit. No completed stage is to
be restarted. The [public export](../references/evidence/trainable_visual_context_result_20261117.json)
retains all snapshot summaries, resource scopes and evidence hashes. Earlier
failed admission and cold attempts remain preserved. This implementation has
not demonstrated a working ILM.

## Learning path

A source is an ordered stream of visible 32×32 ink tiles. A frozen codec maps
each tile to a continuous 768-dimensional vector. The learned visual encoder
groups consecutive tiles and produces continuous contextual updates. These
updates enter a causal transformer with **no lexical input embedding table**.
Its state predicts the next visible span's length and a conditional flow over
that span's continuous glyph vectors. Length zero means stop.

For visible span X(t), write z(t)=C(X(t)), u(t)=E(z(t)) and
h(t)=F(u(1),…,u(t)). The writer predicts X(t+1) from h(t), without receiving
X(t+1) as that state's context. Its image-flow objective is

\[
 \mathbb E_{\epsilon,\tau}
 \left\|v_\psi((1-\tau)\epsilon+\tau z_{t+1},\tau,h_t)
             -(z_{t+1}-\epsilon)\right\|_2^2,
 \qquad \epsilon\sim\mathcal N(0,I),\quad \tau\sim U(0,1).
\]

Training also supervises visual grouping, the next span's length, source-feedback
coordinates and saved teacher states. These six original objectives retain equal
weights. Native source vectors and teacher states are offline supervision; the
deployed generator does not consult a vocabulary, tokenizer, teacher, font or
answer catalogue. Inference decodes each predicted span to actual binary pixels
and rereads those pixels before continuing. Continuous latent feedback alone
does not substitute for this emitted-image history.

The experiment changes whether all 1,409,410,048 inherited contextual parameters
learn. Both arms retain the same image reader, full-rank writer, initialization,
complete data, FP32 weights, BF16 forward arithmetic and ordinary AdamW. Moment
buffers reside in host RAM and bounded chunks update weights on the GPU. This
changes memory placement, not the optimizer's mathematics. It also requires
substantial host memory; GPU memory alone is not the complete resource account.

## What is established

The [six real complete-record trials](../references/trainable_visual_context_preflight_result_20261116.md)
pass: every treatment context tensor receives finite nonzero gradients and
changes, every sampled update agrees exactly with PyTorch AdamW, and every full
state resets exactly. Peak allocated VRAM is 11.252 GiB. The deployed model has
1,465,381,142 parameters; its training-only teacher projection adds 8,388,608.
These are execution measurements. Training loss, copying and readable individual
glyphs do not demonstrate correct complete answers.

The [algorithm assessment](../references/visual_language_algorithm_assessment_20261117.md)
connects this comparison to the measured first-span failures and primary
pixel-language research. It specifies what joint visual/contextual learning
could establish and how to interpret either outcome. It does not change the
registered sources, schedule or acceptance rules.

Each full arm uses all 13,785 sequences, 2,576,239 target glyph positions,
2,380,620 saved teacher states and 8,172 updates. The fixed snapshots are updates
0, 1,956, 4,049, 6,152 and 8,172. Evaluation retains 396 streams per arm, including
paired questions, erased controls, complete books and natural stopping. Review
all actual images and freeze readings before new reference scoring. Two separate
traced final-model replays check execution independence.

The pretrained context already inherits text-language training. A successful
comparison would establish visual adaptation on these tasks, not from-scratch
language learning, broad historical-script reading or a compute advantage over
a matched-quality local LLM. The current interface requires ordered tile grids;
arbitrary page layout and proportional writing remain separate extensions.

## Code and saved results

| Component | Source or local output |
|---|---|
| Model construction and optimizer groups | [trainable_visual_context.py](../ilm/visual_lm/trainable_visual_context.py) |
| Host-moment AdamW | [offloaded_adamw.py](../ilm/visual_lm/offloaded_adamw.py) |
| Full learned-context checkpoint | [trainable_context_checkpoint.py](../ilm/visual_lm/trainable_context_checkpoint.py) |
| Complete trainer | [train_visual_context_comparison.py](../scripts/train_visual_context_comparison.py) |
| Standalone image generation | [infer_visual_context_deployment.py](../scripts/infer_visual_context_deployment.py), equivalent numerical source with self-contained admission |
| Serial comparison and review | [run_visual_context_comparison.py](../scripts/run_visual_context_comparison.py), [review_visual_context_comparison.py](../scripts/review_visual_context_comparison.py) |
| Original training corpus | `data/visual_sequence_corpus_v1/` |
| Existing offline teacher cache | `artifacts/visual_sequence_teacher_v1/cache/` |
| Current comparison | `artifacts/trainable_visual_context_comparison_v1/` |
| Treatment fit and checkpoints | `artifacts/trainable_visual_context_comparison_v1/trainable/fit/` |
| Treatment update ledger | `artifacts/trainable_visual_context_comparison_v1/trainable/fit/updates.jsonl` |
| Generated raw tiles and PNGs | Each arm's `generated_step_*/<case>/` and `generated_final_long_books/<case>/` |

Large weights, corpora, raw runtime identity and generated outputs remain local.
Public registrations and result summaries bind them by hash. A completed failed
run is evidence, not permission to restart it. All comparison workers have exited. Any successor heavy diagnostic needs its
own committed execution binding and verified ownership.

Standalone generation from a trained deployment snapshot uses:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=. OPENBLAS_NUM_THREADS=2 \
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python scripts/infer_visual_context_deployment.py \
    --image prompt_tiles.png --glyph-count 64 \
    --checkpoint artifacts/trainable_visual_context_comparison_v1/trainable/fit/model_step_0008172.pt \
    --output artifacts/visual_context_answer
```

Use a binary PNG composed of gapless 32×32 tiles in reading order, with the
actual tile count. The final snapshot is retained from the completed failed language comparison.
Standalone execution acquires the same GPU mutex and refuses a competing job.
This command is a usage example, not a request to repeat the completed evaluation.
Typed text may be
rendered at the input boundary; generated answer pixels are never font-rendered.
