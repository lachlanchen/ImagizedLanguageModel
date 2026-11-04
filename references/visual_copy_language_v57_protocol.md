# V57: grounded visual question answering with a copy distribution

Frozen before model implementation or QA training, 2026-11-05. This is a
bounded Chinese reading-comprehension experiment, not a general ILM claim.
All training/inference must use physical GPU 0 alone. Preserve V55/V56 weights,
protocols, results and the sealed book partition.

## Probability model and runtime boundary

Initialize the complete V55 joint EMA
`ae4598c609648ba92a4eaa8680fa0842966a1f11b0757ed4a14b83038c048394`.
Reuse the 384-wide, eight-layer causal image reader and exact raster density.
Extend only its allowed context length from 64 to 512; RoPE is parameter-free,
and this change is not an assertion of trained long-context competence.
Add learned 384-to-384 query/key projections, one copy-gate linear head and
one stop-hazard linear head. Source keys are contextual states at actual
observed prompt positions. Queries are causal states after the prompt or
generated answer prefix. No future source position is available to a query.

For continued output, with observed source images X_i,

    p(Y|H,continue) = sigmoid(g(H)) sum_i a_i(H) 1[Y=X_i]
                     + sigmoid(-g(H)) p_raster(Y|H).

Attention a is a softmax over valid visible positions, not a persistent
character vocabulary. Duplicate identical source rasters contribute summed
mass. Exact raster equality is computed from pixels. A Bernoulli stop hazard
s(H) yields p(stop|H)=sigmoid(s), and p(Y,continue|H)=sigmoid(-s)*p(Y|H,continue).
Stop is transport framing, not a glyph or Unicode output symbol. Train the
normalized joint law, with no symbolic character identity supervision.

The matched no-copy arm has the same initialized reader/density/stop head,
data, optimizer schedule and update count; its copy weight is fixed to zero.
No external model, OCR, tokenizer, global glyph inventory or reference answer
is available at deployed inference. Host rasterization and lengths used for
packing/training masks stay outside the student. No gold answer length is
given to inference. Native aligned image input remains supported.

## Data and optimization

Use the prepared full-paragraph CMRC subset, unchanged:

- train: 3,211 records, SHA-256 `742a473db53d1a55e319f0b52befc8da5fa3d884df395b37e66581e882d1496c`;
- development: 1,321, `8bef093075ac0715d5878a1abe61aa38092893ade8753b935cdf657551ebc6a7`;
- 393 paired development questions: `3bcb2e0534290ed9cf3dd1a4e378bbaa5a7eb4903dce0b4b0befb8f3ce58029d`.

Do not crop around an answer. Prompt <=480 cells, retained answer <=16. Use
the first retained reference for teacher forcing and all retained references
for answer evaluation. Canonical 32x32 Noto Sans CJK rendering, fixed 26-pixel
font, threshold 0.5, no augmentation, as in V55. Data attribution remains
CC-BY-SA-4.0, Cui et al., EMNLP-IJCNLP 2019.

Each arm makes 4,000 updates with batch four. Three of every four updates use
uniform training questions with replacement. The fourth retains the original
public Chinese stream objective: 16 causal positions per stream, targets only
at the next position, attention to source positions no later than the query.
A quarter of those natural targets receive supplied-image motor conditions;
their reader states are zeroed, exact observed fields replace predicted fields,
copy routing is disabled and stop loss is omitted. Other natural targets are
labelled continue. QA includes its learned terminal stop event.

AdamW betas (0.9,0.95), weight decay .01, gradient norm clip one; rates 3e-5
reader, 1e-4 density, 3e-4 new heads; 200-update warmup and cosine to .1 of peak.
Loss is mean exact event NLL over valid continued images and stop events,
without a 1/1024 image-only rescaling. BF16 training, high matmul precision,
reader dropout disabled, EMA .999. Seed 20265710; data seed 20265711. Both
arms complete before opening development performance. No best-checkpoint
selection: use final EMA. Checkpoint optimizer, RNGs, hashes and data index.
Interrupted/continuous smoke equality, future-causality, duplicate-mass and
normalized-law tests must pass before production. Smoke outputs are excluded
from scientific results. Run arms sequentially; abort safely if allocated
VRAM reaches 16 GiB or a run exceeds four hours.

## Decoding and evaluation

Use highest FP32 matmul precision at evaluation. Primary decoding is explicitly
hierarchical branch-greedy: stop at hazard >=.5; otherwise choose the copy
component when its gate >=.5, marginalize attention over identical source
images and emit the largest-mass image (first position breaks ties). Otherwise
use raw-pixel beam eight. This does not claim global mixture-mode optimality.
Do not add output penalties, OCR correction, answer-aware filters or span rules.
Allow at most 17 emitted cells, recording any capped/unstopped output as such.

Primary autonomous audit: 256 development questions sampled without replacement
using NumPy default_rng(20265720), from records sorted by id. Both trained
arms generate raw image answers. The copy arm also sees a paired control
where only the question's pixels are blanked; context and framing are retained.
Do not provide answer location/length or a candidate answer list. Report exact
sequence match, maximum multiset raster F1 over references, stopping, length,
copy/free routing, raw first audit examples and resource use. Exact equality
and F1 are computed on rasters, not nearest glyph identities or OCR.

Also compute teacher-forced QA event NLL on all 1,321 development questions.
For all 393 same-context/disjoint-answer question pairs, generate the copy
arm's answers to both questions and report both-correct pair fraction. This
secondary task checks correct answer changes, not merely changed output.
Report paired source-article bootstrap intervals with 2,000 draws, seed
20265819. Development has been used for data filtering, so this is not a
sealed final test or independent training-seed replication.

Bounded success requires all: image-only boundary; clean split and finite
resource checks; exact supplied motor copying on 256 V55 audit images; primary
exact answer match >=5%; primary raster F1 >=25%; ordered-minus-blank-question
F1 lower 95% interval >0; and both-correct paired questions >=2%. Report the
matched no-copy comparison regardless of outcome. Even all gates passing
would establish only this grounded reading capability. General language,
free-form instructions, historical scripts, page images and LLM-parity remain
open requirements of the active goal.
