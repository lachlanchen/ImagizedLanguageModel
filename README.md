[English](README.md) · [العربية](i18n/README.ar.md) · [Español](i18n/README.es.md) · [Français](i18n/README.fr.md) · [日本語](i18n/README.ja.md) · [한국어](i18n/README.ko.md) · [Tiếng Việt](i18n/README.vi.md) · [中文 (简体)](i18n/README.zh-Hans.md) · [中文（繁體）](i18n/README.zh-Hant.md) · [Deutsch](i18n/README.de.md) · [Русский](i18n/README.ru.md)


[![LazyingArt banner](https://github.com/lachlanchen/lachlanchen/raw/main/figs/banner.png)](https://github.com/lachlanchen/lachlanchen/blob/main/figs/banner.png)

# Imagized Language Model (ILM)

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/status-research-orange)
![Focus](https://img.shields.io/badge/focus-text--as--image-0A7EA4)
![Paradigm](https://img.shields.io/badge/paradigm-predictive%20visual%20field-16835B)
![License](https://img.shields.io/badge/license-unspecified-lightgrey)
![Domain](https://img.shields.io/badge/domain-historic%20etymology%20%7C%20glyph%20models-2F80ED?logo=github)

![ILM-V image-native language model concept: image input to image output with 言 glyph evolution](publication/ilm-image-native/figures/ilm_v_yan_readme_hero.png)

*Image-native language modeling concept: a writing image enters ILM-V, the model reasons in visual latent space, and the answer is rendered as an image. The glyph panels use local hanziyuan-derived ziyuan data for the evolution of `言` (YAN, U+8A00).*

## Research North Star: A Visual Word-Origin Book

The concrete product target is an independent image-native model that accepts a
rendered English or Chinese question, or a photographed page, and emits a
readable answer **as a page image**. A word-origin answer should combine modern
English/Chinese explanation with real provenance-linked oracle, bronze, seal,
clerical, traditional, simplified, manuscript, or unencoded forms. OCR may add
a searchable sidecar after inference; the UI may display that text beside the
native answer image, but it is not the model's language channel.

The interface still behaves like a normal prompt box. Typed text is rendered
into a clean prompt band; an optional book page, inscription, or glyph image is
placed beside or below it on the same visual canvas. The model can therefore
answer ordinary typed questions or questions grounded in an attached page
without receiving hidden text metadata.

The canonical interface is a **Visual Language Stream** with sequence/time,
optional geometric depth, height, width, and sensory channels. A page is the
`T=1,D=1` case; a book is an ordered stream of fields; a 3D Chinese or English
character string uses depth; and a character movie also uses time. These are
one continuous input/output contract, not separate token vocabularies. V24
completes the first bounded 2D Chinese grammar proof: it reads a variable raster
packet stream, writes one image, rereads those generated pixels, and writes a
second image on one RTX 4090. V25 then replaces that designed packet algebra
with ordinary rendered Chinese. It finds a weak ordered-history signal but is
rejected as a language model. V26 separates last-glyph appearance from earlier
history and predicts continuous visual particles, but its pixel-identical
suffix intervention shows that changed history states still produce chance
target preferences. V27 jointly trains a causal visual context and arbitrary
candidate-image compatibility. The raw retina still distinguishes cross-font
forms at `99.95%`, but full-context pair assignment is `50.71%` and only
`0.15` percentage point above shuffled context; natural top-1 remains below
unigram and bigram. This rejects the global query/key compatibility route. The
next proof must preserve raw visual geometry and make ordered future-field
prediction causally necessary at 64 cells before adding page scale, 3D
geometry, or motion.

Concretely, the intended model maps prompt frames
`X_prompt[Tp,D,H,W,C]` to generated answer frames
`Y_answer[Ta,D,H,W,C]`. Typed questions are rendered into `X_prompt`; scanned
pages or handwriting enter directly. `Ta=1` is an answer page and `Ta>1` is a
text-image stream or movie. A valid understanding result must change the
generated answer appropriately under held-out prompt changes; reconstruction,
OCR, glyph classification, and attractive writing alone do not satisfy it.

The deployed student must not call Qwen, an OCR engine, a tokenizer, a Unicode
lookup, or a glyph database to decide its answer. External models and extracted
text may help build and audit an offline curriculum, but every student batch and
checkpoint must pass a boundary receipt showing that its learned path contains
only writing pixels and continuous visual states. The measurable roadmap and
source-book policy are in
[`docs/first-imagized-language-model-goal.md`](docs/first-imagized-language-model-goal.md)
and
[`references/word_origin_ilm_dataset_plan.md`](references/word_origin_ilm_dataset_plan.md).

## Latest Natural-Language Test: V27 Joint Compatibility Rejected

![Measured V27 joint visual-compatibility result: candidate images remain visible, but full context does not beat shuffled context or frequency baselines](publication/ilm-image-native/figures/joint_visual_compatibility_v27_result.png)

V27 tests whether a compact model can learn language by scoring an arbitrary
next-glyph image directly from 64 preceding glyph images. Its
`18,599,553`-parameter image-only student initializes an online retina from
V16, builds a context query with eight causal rotary blocks, and scores an EMA
candidate-image key. It contains no strings, token or Unicode IDs, OCR,
vocabulary matrix, codebook, candidate bank, glyph lookup, or external model.
Candidate order is randomized independently, and the evaluator removes that
permutation before scoring.

The single preregistered run performs 8,000 BF16 updates on one RTX 4090.
Training plus audit takes `39.12` minutes with `2.268 GiB` peak allocated
CUDA memory. The fixed 2,048-window natural audit reaches `1.6113%` top-1,
below the image unigram (`2.0508%`) and symbolic bigram (`12.5977%`).
Full context improves target log probability over suffix-4 by `0.05925` nat,
but only `0.00273` nat over a suffix-preserving prefix shuffle.

The decisive 512-pair audit keeps the final four glyph images bitwise equal
while changing earlier history and the target. The unchanged V16 retina
identifies cross-font candidate forms at `99.9512%`, and candidate
permutation error is exactly zero, so candidate visibility and row-position
shortcuts are controlled. Full-context assignment is nevertheless
`50.7080%`, compared with `50.5615%` after shuffling the prefix. Separately,
learned cross-font identity over the 1,024-image bank is `94.8730%` and misses
its `99%` gate; that 1,024-way metric is not directly comparable to the
two-candidate raw control. V27 passes only
`7/13` mechanism gates and `1/5` language gates. It is rejected, the frozen
partition remains sealed, and no writer is trained.

The next experiment should keep the full `N x 1 x 32 x 32` glyph-image
stream authoritative, freeze or exactly preserve the strong raw retinal
geometry, and optimize dense per-position predictions with explicit
suffix-preserving order interventions. A reversible 2D lattice can accelerate
that stream later; depth and motion remain observable extensions rather than
identity encodings. See the
[complete V27 receipt](docs/joint-visual-compatibility-v27-result.md),
[preregistered protocol](references/joint_visual_compatibility_v27_protocol.md),
and [research decision](references/deterministic_visual_compatibility_v27_research.md).

## Prior Natural-Language Test: V26 Factorized Context Rejected

![Measured V26 factorized visual-context result: earlier history changes the residual state, but matched next-glyph preference remains at chance](publication/ilm-image-native/figures/factorized_visual_context_v26_result.png)

V26 tests the repair proposed after V25 without changing the ordinary-Chinese
or image-only boundary. Its `19,142,721`-parameter model gives the last visible
glyph and the preceding 63 glyph images separate routes, fuses their continuous
states, and predicts eight 192-dimensional visual particles for each of future
horizons 1, 2, 4, and 8. The deployed student receives no strings, token or
Unicode IDs, OCR, labels, glyph table, candidate bank, or external model state.

The fixed run completes 8,000 BF16 updates in `31.05` minutes on one RTX 4090,
using `0.888 GiB` peak allocated CUDA memory. On 2,048 development windows,
full-history top-1 is `0.0488%`, versus `1.4160%` for the image unigram and
`13.5254%` for the symbolic bigram. Full history improves target log
probability over last-only by `0.17045` nat, but only by `0.01973` over the
same four-cell suffix and `0.00369` over a prefix shuffle.

The decisive audit uses 512 cross-record context pairs with pixel-identical
four-glyph suffixes and different targets. Their appearance-state difference
is exactly zero and their mean history-residual difference is `4.62826`, so the
history branch is active. Correct pair ranking and swapped-residual target
accuracy are nevertheless both exactly `50%`, with mean score margin
`0.0000677`. A perfect retina-bank oracle rules out a blind evaluator. V26
therefore fails its mechanism and language gates; no frozen evaluation or
writer is authorized.

This localizes the failure to conditional binding, not visual detection or the
mere existence of a history state. Low memory and materially different hidden
states are not useful language prediction. See the
[complete V26 receipt](docs/factorized-visual-context-v26-result.md),
[preregistered protocol](references/factorized_visual_context_v26_protocol.md),
and [research decision](references/factorized_visual_context_v26_research.md).

### Pre-V27 Localization: Frozen Compatibility Probe

![Frozen V26 visual compatibility diagnostic: retina identity remains nearly perfect while history and fused-state next-glyph assignment stay at chance](publication/ilm-image-native/figures/v26_frozen_visual_compatibility_probe.png)

A post-hoc diagnostic freezes all `19.14M` V26 parameters and trains three
small candidate-conditioned image scorers for one pass over the existing
`16,384` train suffix pairs. On `512` disjoint-record development pairs and
`2,048` cross-font decisions, appearance-only accuracy is exactly `50.000%`,
history-residual accuracy is `50.684%`, and fused-state accuracy is `50.342%`.
The same frozen retina identifies the paired target image across fonts at
`99.951%`, with a `0.74724` mean cosine margin. Candidate ambiguity therefore
does not explain the chance language result.

This diagnostic is not preregistered evidence and opens no frozen data. It
motivated V27's joint causal-context and deterministic image-candidate test.
The preregistered result above shows that this relation did not pass, so no
stochastic writer was authorized. See the
[diagnostic receipt](docs/v26-frozen-visual-compatibility-probe.md) and
[V27 research decision](references/deterministic_visual_compatibility_v27_research.md).

## Prior Natural-Language Test: V25 Visual Cell Stream Rejected

![Measured V25 visual-cell result: full image history carries a weak ordered signal, but the language model and exploratory writer fail their fixed gates](publication/ilm-image-native/figures/visual_cell_stream_v25_result.png)

V25 is the first experiment here to train directly on ordinary Chinese book
language as an ordered `64 x 1 x 32 x 32` visual-time stream. Its
`25,549,714`-parameter model uses a frozen image retina, an eight-layer causal
continuous field, a vocabulary-free next-state proposal, and a flow writer that
can append and reread actual generated pixels. The student receives no strings,
token or Unicode IDs, OCR transcript, character labels, glyph lookup, discrete
codebook, or external model call.

The fixed 2,400-update language run completes on one RTX 4090. On 2,048
development windows, full 64-cell history reaches `1.123%` next-cell top-1,
versus `0.146%` last-only and `0.342%` with prior history shuffled. This is a
real ordered-history effect, but it is too small: the image unigram reaches
`1.611%` and the symbolic bigram `12.158%`. Counterfactual switch accuracy is
`12.891%`, target cosine is `0.2751`, and six fixed semantic/causal gates fail.
Peak allocated CUDA memory is `0.598 GiB`; low memory is not evidence of
language efficiency when predictive quality remains below a bigram. The frozen
partition stays sealed.

An explicitly labeled exploratory writer run after rejection preserves
position-16 ink density (`0.977x`) and avoids blanks, but reaches `0%` generated
identity top-1, `0.0802` reread cosine, and `0.3221` pixel F1. Its output is
glyph-like texture, not readable continuation. This diagnostic does not alter
the fixed evidence verdict.

The result points to the next controlled problem: separate exact visible
appearance from a context-predictive residual and prove that both states are
causally needed before scaling context. The implemented reversible
[`serpentine visual lattice`](references/serpentine_visual_lattice.md) can later
fold up to 65,536 clean cells into a long-context retinal field without losing
the authoritative `32x32` glyph stream, but it was not used in V25 and is not a
claimed fix. See the [complete V25 receipt](docs/visual-cell-stream-v25-result.md)
and [unchanged frozen protocol](references/visual_cell_stream_v25_protocol.md).

## Latest Stream Test: V24 Visual Packet Rereading Accepted

![Measured V24 visual packet stream: variable raster packets are localized from visible headers, routed through a visual relation, emitted as a glyph image, reread from generated pixels, and followed by a generated label image; paired controls and the single frozen evaluation pass](publication/ilm-image-native/figures/visual_packet_stream_v24_result.png)

V24 is the first accepted **variable-input, multi-frame-output** visual stream
in this repository. The student receives `15`, `18`, `21`, or `24` grayscale
`32x32` frames grouped into visibly headed packets. It locates two bindings, an
operation, and a query from header images; emits the selected unseen Chinese
glyph as frame 1; rereads the actual generated pixels through the frozen visual
retina; and emits the glyph's visibly bound label as frame 2. Its deployed path
receives no strings, token or Unicode IDs, OCR, role or operation labels,
packet indices, active length, padding mask, glyph lookup, discrete codebook, or
external language model.

Only `1,347` parameters are trained. On a fresh `1,024`-episode paired audit,
query, operation, and generated-history switch accuracy are `0.99219`,
`0.99316`, and `0.99609`. The corresponding query-blind, operation-blind, and
history-blind controls each have exactly `0.0` switch accuracy and `0.0`
output-pixel change for the factor they cannot observe. A header-blind control
falls to `0.08203` minimum role localization, versus `1.0` for the candidate.
Every arm has identical parameter names, shapes, and count.

An opaque agent visual audit, performed before opening its sealed answer key,
scores `47/48` for frame 1 and `48/48` for frame 2, including `12/12` for both
frames at the held-out `T=24` length. This is an agent visual audit, not a human
study.

The single authorized frozen run covers `107` unseen identities and `1,024`
episodes. It performs no model selection, changes no threshold, and is not
repeated.

| V24 frozen gate | Measured | Required | Result |
|---|---:|---:|:---:|
| Frame-1 binary choice | `0.99805` | `>0.95` | pass |
| Query switch | `0.97656` | `>0.90` | pass |
| Operation switch | `0.97266` | `>0.90` | pass |
| Generated-history switch | `0.99609` | `>0.90` | pass |
| Held-out minimum switch | `0.96353` | `>0.85` | pass |
| Frame-1 identity top-1 | `0.98672` | `>0.75` | pass |
| Frame-2 label top-1 | `0.99727` | `>0.95` | pass |
| Frame-1 / frame-2 pixel F1 | `0.83714 / 0.72860` | `>0.68 / >0.58` | pass |
| Held-out `T=24`, frame 1 / frame 2 | `0.98473 / 1.00000` | each `>0.90` | pass |
| Packet-permutation consistency | `1.00000 / 1.00000` | each `>0.99` | pass |

V24 proves a fixed packet grammar and causal two-frame image answer, not
arbitrary sentence understanding or a finished language model. Packet arity,
header semantics, same/other algebra, and output length remain designed into
the task. It does not yet answer etymology questions, continue pages, write
unrestricted text, or emit a movie. The next milestone must learn from rendered
Chinese prompts and passages, generate a learned-length image-line stream, and
pass semantic counterfactuals and blind-history controls. See the
[complete V24 receipt](docs/visual-packet-reread-stream-v24-result.md).

## Prior Prompt Test: V23 Visual Relation Circuit Accepted

![Measured V23 visual relation circuit: six raster prompt frames pass through a frozen retina, learned visual comparison and operation gate, routed source pixels, and a frozen canonicalizer; paired controls and the single frozen evaluation pass](publication/ilm-image-native/figures/visual_relation_circuit_v23_result.png)

V23 is the first complete positive image-prompt-to-image-answer result in this
repository. Six `32x32` writing images enter the student and one `32x32` answer
image comes out. The prompt visibly binds two previously unseen Chinese glyphs
to two labels, supplies `同` or `异`, and ends with a visual query label. The
student compares images, reads the operation from its image, routes one visible
source glyph, and renders a canonical answer. Its learned path receives no
strings, token or Unicode IDs, OCR, character labels, answer indices, codebook,
glyph lookup, or external language model.

The relation-aware candidate selected under the fixed development protocol.
On a fresh 1,024-episode paired audit it reaches `0.99805` query and operation
switch accuracy and `0.99512` identity top-1. Query-blind and operation-blind
controls have exactly `0.0` switch accuracy and `0.0` output-pixel change for
the factor each cannot see. An opaque agent visual review then scores `48/48`
overall and `12/12` on held-out compositions before the sealed key is opened.

The single authorized frozen run covers 98 unseen identities, 1,024 episodes,
and 4,096 prompt variants. It performs no model selection and changes no
threshold.

| V23 frozen gate | Measured | Required | Result |
|---|---:|---:|:---:|
| Binary choice | `0.99829` | `>0.95` | pass |
| Query switch | `0.99609` | `>0.90` | pass |
| Operation switch | `0.99707` | `>0.90` | pass |
| Held-out minimum switch | `0.99606` | `>0.85` | pass |
| Unseen-identity top-1 | `0.99463` | `>0.75` | pass |
| Pixel F1 | `0.78478` | `>0.68` | pass |
| Target cosine | `0.93994` | `>0.82` | pass |
| Query-label visual match | `0.99951` | `>0.98` | pass |
| Operation-gate accuracy | `1.00000` | `>0.98` | pass |
| Pair-swap consistency | `1.00000` | `>0.99` | pass |

V23 proves bounded visual relation following, not open-ended language. Frame
roles and the two-pair same/other algebra remain fixed; the output is a
canonicalized form of one visible source glyph. It does not yet parse arbitrary
sentences, answer etymology questions, continue pages, or emit an image stream
or movie. V24 takes the next bounded step by removing absolute frame roles,
reading a variable-length packet stream, and generating two answer frames while
rereading the first. See the
[complete V23 receipt](docs/visual-relation-circuit-v23-result.md).

## Prior Prompt Test: V22 Binding Mechanism Rejected

![Measured V22 visual binding stream: the query-aware selector collapses onto the operation frame, candidate and query-blind outputs remain nearly identical, and the preregistered prompt-binding gates reject the model](publication/ilm-image-native/figures/visual_binding_stream_v22_result.png)

V22 is the first bounded implementation of the requested visual prompt stream:
six `32x32` writing images enter the student and one answer image comes out.
Each prompt visibly binds two previously unseen Chinese glyph images to labels,
shows `同` or `异`, and ends with a visual query label. A paired
counterfactual changes only that final image, so a model that understands the
prompt must switch its generated answer. The query-aware candidate and
query-blind control each have exactly `3,410,128` trainable parameters and use
no strings, token/Unicode IDs, OCR, glyph lookup, answer codebook, or external
language model.

The model does **not** pass. At step `1,600`, candidate switch accuracy is only
`0.0078`, versus `0.0` for the query-blind control. Identity top-1 is `0.1592`
versus `0.1533`, and pixel F1 is `0.5107` versus `0.5125`. Changing the visible
query changes candidate pixels by only `0.0089` mean L1, far below the fixed
`0.08` requirement.

| V22 candidate development gate | Measured | Required | Result |
|---|---:|---:|:---:|
| Binary choice | `0.4824` | `>0.85` | fail |
| Counterfactual switch | `0.0078` | `>0.80` | fail |
| Held-out-combination switch | `0.0113` | `>0.75` | fail |
| Unseen-identity top-1 | `0.1592` | `>0.45` | fail |
| Identity gain over query shuffle | `+0.0225` | `>0.20` | fail |
| Pixel F1 | `0.5107` | `>0.58` | fail |
| Oracle-writer F1 | `0.6020` | `>0.64` | fail |
| Paired-output L1 | `0.0089` | `>0.08` | fail |
| Frozen images instantiated | `0` | `0` | pass |

The endpoint audit explains why: the candidate gives the operation frame the
maximum selector weight in all **`1,024/1,024`** original and counterfactual
prompts, with mean operation attention `1.0` and mean query attention
`1.37e-13`. A relational answer needs the operation, query-to-label match, and
label-to-glyph binding jointly; collapsing six frames into one selected frame
cannot perform that composition. V23 therefore replaces single-frame selection
with an explicit, differentiable multi-frame visual relation circuit. No V22
candidate selected, so paired, human, and frozen evaluation remain forbidden.
See the [complete V22 receipt](docs/visual-binding-stream-v22-result.md).

## Prior Causal Test: V21 Field-Complete Route Works, Writer Rejected

![Measured V21 field-complete writer: the local continuous field carries the complete spatial plan, but simple, medium, and overall quality gates reject the writer](publication/ilm-image-native/figures/field_complete_writer_v21_result.png)

V21 tests whether a continuous local field can carry the **complete** visual
plan. Its candidate and tiled-global control each have exactly `582,336`
trainable parameters. Every local cell emits coarse occupancy plus 63
Walsh--Hadamard zero-DC coefficients for its corresponding `8x8` patch. Global
state and style provide only spatially uniform modulation; there are no
coordinates, position parameters, cell mixing, or global spatial projection.

At the best diagnostic step `1,400`, correct-field dense F1 is `0.7053`, versus
`0.5314` after shuffling the field and `0.3588` after zeroing it. The fixed
gains `+0.1739` and `+0.3465` pass. Identity top-1 is `79.10%`, target cosine
is `0.8331`, all exact-basis invariants pass, and quadrant locality is exactly
`1.0`. The equal-parameter control collapses to repeated textures, reaching
only `0.1473` overall and `0.3074` dense F1 at its selected structural step.

| V21 candidate development gate | Measured | Required | Result |
|---|---:|---:|:---:|
| Overall pixel F1 | `0.6038` | `>0.66` | fail |
| Simple pixel F1 | `0.5648` | `>0.58` | fail |
| Medium pixel F1 | `0.5945` | `>0.60` | fail |
| Dense pixel F1 | `0.7053` | `>0.70` | pass |
| Dense gain over shuffled field | **`+0.1739`** | `>0.15` | pass |
| Dense gain over zero field | **`+0.3465`** | `>0.20` | pass |
| Identity top-1 | `79.10%` | `>74%` | pass |
| Target cosine | `0.8331` | `>0.82` | pass |
| Occlusion locality | **`1.0000`** | `>0.95` | pass |
| Detail block-mean magnitude | `3.87e-7` | `<5e-6` | pass |

The writer is still rejected because no candidate checkpoint passes all quality
gates. The comparison to the control is descriptive, not a formal paired audit;
the paired evaluator must refuse an unselected candidate. Human review and
frozen evaluation were not authorized, and frozen images remained
uninstantiated. V21 proves a field-complete causal route, not prompt
understanding or autonomous language generation. See the
[complete V21 receipt](docs/field-complete-writer-v21-result.md).

## Prior Causal Test: V20 Routes Local Detail, Writer Rejected

![Measured V20 retinal topology router: correct local fields carry necessary detail and local occlusion stays local, but quality and paired-control gates reject the writer](publication/ilm-image-native/figures/retinal_topology_router_v20_result.png)

V20 reserved within-block detail for a local `4x4x192` field while global state
supplied coarse occupancy. It passed the field-shuffle (`+0.1218`), zero-field
(`+0.3613`), and locality (`1.0`) gates, but failed overall F1, target cosine,
an exact-decomposition invariant, and the matched-control margin. V21 removed
that remaining global spatial route. See the
[complete V20 receipt](docs/retinal-topology-router-v20-result.md).

## Prior Routing Test: V19 Rejected

![Measured V19 spatial retinal residual: correct, shuffled, and zero spatial fields produce nearly identical writing, so the preregistered causal topology gate fails](publication/ilm-image-native/figures/spatial_motor_plan_v19_result.png)

V19 tested the next proposed correction instead of assuming it worked. A clean
`2,358,977`-parameter global planner was first trained from scratch on a new
salted split. Its weights and the V16 retina were then frozen while a
`764,545`-parameter adapter learned from the retina's continuous `4x4x192`
spatial field. The target image supplied loss only. The student still received
no token IDs, Unicode IDs, OCR, strings, character labels, lookup, codebook, or
external language model.

On a fresh 512-candidate development audit, dense pixel F1 is `0.7278`, but the
correct field beats a shuffled field by only **`0.0088`** and a zero field by
only **`0.0054`**. The prospectively fixed margins were `>0.12` and `>0.03`.
Overall F1 (`0.6710`) and identity top-1 (`72.66%`) also miss their fixed gates.

| V19 development gate | Measured | Required | Result |
|---|---:|---:|:---:|
| Overall pixel F1 | `0.6710` | `>0.68` | fail |
| Dense pixel F1 | `0.7278` | `>0.58` | pass |
| Dense gain over shuffled field | **`+0.0088`** | `>0.12` | fail |
| Dense gain over zero field | **`+0.0054`** | `>0.03` | fail |
| Identity top-1 | `72.66%` | `>75%` | fail |
| Target cosine | `0.8416` | `>0.84` | pass |

The automatic gate is rejected, so human review was not authorized and the
frozen split remains sealed. The result identifies a routing failure: an
optional local residual can polish a complete global writer without making
local topology necessary. V20 subsequently fixed that causal routing defect but
still failed writer selection. See the
[full V19 result](docs/spatial-retinal-motor-plan-v19-result.md) and the
[2026 continuous-sensory decision scan](references/continuous_sensory_language_scan_2026.md).

## Accepted Development Proof: Visual Motor Plan V18

![Measured V18 visual motor plan: a compact image-native decoder writes recognizable held-out Chinese forms from continuous visual intent](publication/ilm-image-native/figures/visual_motor_plan_v18_result.png)

V18 is a real `2,358,977`-parameter deterministic visual motor planner trained
for 1,600 updates on one RTX 4090. It receives a continuous `192`-dimensional
state read from a different-font image plus a separate style image and emits a
`32x32` continuous ink plan. Target pixels provide loss only. The learned path
has no token IDs, Unicode IDs, OCR, strings, character labels, output
vocabulary, glyph lookup, finite visual codebook, candidate classifier, or
external language model.

On a fresh 512-example **development-only** audit, V18 reaches **73.63%** global
visual-identity top-1 versus **0.98%** after shuffling only intended states.
Target cosine is **0.8462** versus `0.0716`; pixel F1 is **0.6577** versus
`0.3129`. Ink occupancy is identical in both branches. Most reviewed simple and
medium forms are recognizable, while dense forms can still merge strokes.

| Fresh development measurement | Correct intent | Shuffled intent | Result |
|---|---:|---:|---|
| Global visual identity top-1 | **73.633%** | 0.977% | strong causal control |
| Target cosine | **0.8462** | 0.0716 | gain `+0.7746` |
| Pixel F1 | **0.6577** | 0.3129 | automatic topology gate passes |
| Human review | simple/medium readable | unrelated forms | dense forms still fail |
| Peak allocated CUDA memory | **0.778 GiB** | - | far below 4090 capacity |

This breaks a categorical claim: structured writing can be learned and emitted
as continuous image topology by a small consumer-GPU model without a token
output table. It does not prove autonomous language generation. V18 is supplied
the intended state, the human gate lacked a prespecified numeric rubric, and
the new frozen bank remains untouched. The full protocol, limitations, and
reproduction receipt is in
[`docs/visual-motor-plan-v18-result.md`](docs/visual-motor-plan-v18-result.md).

## Prior Causal Actuator: V17

![Measured Visual State Actuator V17: an image-derived continuous state causally controls generated pixels on a frozen split, but exact stroke topology and human readability fail](publication/ilm-image-native/figures/visual_state_actuator_v17_result.png)

V17 is a real `5,729,921`-parameter visual actuator trained for 1,600 updates on
one RTX 4090. It receives a `192`-dimensional state read from a different-font
image of the intended form plus a continuous style image, then generates
`32x32` ink pixels. The target image supplies loss only; its spatial pixels do
not enter the condition. The learned path has no token IDs, Unicode IDs, OCR,
character labels, output vocabulary, visual codebook, candidate classifier,
glyph lookup, or external language model.

On one untouched frozen split of 512 generated examples, V17 obtains **58.59%**
global visual-identity top-1 versus **0.98%** when intended states are shuffled
while style and initial noise remain fixed. Target-state cosine is **0.7130**
versus `0.0861`, a gain of `+0.6269`. The intervention establishes that a small
continuous visual state causally controls generated writing pixels rather than
merely copying style.

| Frozen actuator gate | Correct state | Shuffled state | Result |
|---|---:|---:|---|
| Global visual identity top-1 | **58.594%** | 0.977% | passes causal-control gate |
| Target cosine | **0.7130** | 0.0861 | gain `+0.6269` |
| Pixel F1 | **0.4385** | 0.2800 | **fails required 0.5000** |
| Human readability | rejected | rejected | pseudo-characters remain |
| Peak allocated CUDA memory | **1.588 GiB** | - | fits far below 4090 capacity |

This breaks a categorical claim, not the whole language problem: a consumer GPU
can train a compact non-token visual state to control image generation. V17 is
still rejected as a readable actuator. Its mostly pseudo-character outputs show
that retinal identity can be optimized without preserving exact stroke
topology. It is also an isolated actuator test supplied with the intended state,
not autonomous next-language generation. V16 remains below a symbolic bigram.

V18 implements the correction: it decodes continuous intent into a directly
supervised spatial **visual motor plan** and makes readable writing emerge on
development data. V17 remains the frozen causal-control baseline. Its complete
selection, frozen receipt, limitations, and reproduction command are in
[`docs/visual-state-actuator-v17-result.md`](docs/visual-state-actuator-v17-result.md).

## Language Core: Predictive Visual Field V16

![Measured Predictive Visual Field V16: writing images enter a frozen retina, recurrent base, and residual multiscale causal visual memory; frozen evaluation shows continuous predictions use full history while remaining below a symbolic bigram](publication/ilm-image-native/figures/predictive_visual_field_v16_result.png)

V16 is a real `16,471,809`-parameter image-only causal state model trained on
one RTX 4090. It adds a `6,001,536`-parameter residual multiscale memory, with
dilated local visual fields and global causal attention, to the proven V15
recurrent base. Its learned path receives sequences of `32x32` writing images
and produces continuous next-image states. It has no token IDs, Unicode IDs,
OCR, character labels, output vocabulary, visual codebook, candidate
classifier, or external language model.

On the unchanged frozen 512-form, four-view Chinese benchmark, the selected
continuous proposal obtains **6.264%** top-1 (`112/1,788`), versus **3.971%**
with only the last image, **1.734%** unigram, **0.224%** random dynamics,
**0.195%** chance, and **13.143%** symbolic bigram. Full image history adds
`+0.0773` normalized target log-probability. The parallel stochastic field
obtains **3.691%**, versus **3.244%** last-only, with sampled-state context
cosine gain `+0.0795`.

| Frozen gate | V15 | V16 | Result |
|---|---:|---:|---|
| Proposal full-context top-1 | `5.872%` | **`6.264%`** | +7 correct contexts; directional only |
| Proposal last-image top-1 | `4.418%` | **`3.971%`** | V16 context separation is larger |
| Proposal context log-probability gain | `+0.0707` | **`+0.0773`** | full history helps |
| State-flow full-context top-1 | `3.412%` | **`3.691%`** | beats last-only and unigram |
| Symbolic bigram | `13.143%` | `13.143%` | **not beaten** |
| Peak allocated CUDA memory | `1.181 GiB` | `1.479 GiB` | fits far below 4090 capacity |
| Coupled pixel actuator | absent | absent | isolated V18 planner is readable on development, not yet coupled |

This breaks a narrow but important claim: a small model can learn causal
language signal directly from rendered writing images on a consumer GPU. It
does **not** yet establish general language understanding, readable image
generation, historical question answering, or parity with an LLM. Seven extra
correct frozen contexts are not a statistically established architecture win.
The V16 selection, compute, gate, and limitation receipt is in
[`docs/predictive-visual-field-v16-memory-result.md`](docs/predictive-visual-field-v16-memory-result.md);
the V8-V15 history remains in
[`docs/predictive-visual-field-v15-result.md`](docs/predictive-visual-field-v15-result.md).

## Paradigm: Separate Language From Drawing

![Predictive Visual Field: writing images become continuous retinal states, a causal field predicts the next visual state, a separate visual actuator writes it, and the generated pixels are reread](publication/ilm-image-native/figures/predictive_visual_field_paradigm.png)

RFLM V7 exposed a structural error: one conditional pixel flow was being asked
to discover the next linguistic identity and render its strokes in the same
operation. V14 through V16 now implement the first half of a factorized solution
without relaxing the image-only boundary:

1. A retina learns a continuous manifold directly from writing images.
2. A causal field predicts a low-variance **continuous visual proposal**.
3. A hyperspherical flow models a distribution over alternative next states.
4. A deterministic visual motor planner renders intended topology; V18 makes
   simple and medium held-out forms readable on development data.
5. Optional stochastic flow can refine style only after topology is stable.
6. The retina will reread the rendered pixels and feed them back into the field.

There is no nearest-character lookup or output vocabulary. The continuous state
proof now passes random, last-only, unigram, context-use, and target-signal
gates. It still fails the bigram language gate. V18 passes automatic development
topology gates but is not frozen-promoted because its human rubric was
underspecified and dense forms still fail. V19 then rejects an additive spatial
residual as the repair: correct, shuffled, and zero spatial fields produce
nearly the same output. V20 makes local detail structurally necessary and
topographic, but still misses writer quality and matched-control gates. V21
makes both occupancy and detail field-causal and passes every structural
invariant, but its disjoint patches miss simple, medium, and overall quality.
The autonomous prompt-to-answer write-reread loop remains withheld until a
continuity-preserving local writer and the language core pass independently.

The strict student boundary remains:

```text
writing pixels -> continuous visual dynamics -> continuous ink pixels
```

The student receives no strings, token IDs, Unicode IDs, OCR transcript,
character labels, external language model, or discrete visual codebook. Typed
input is supported only by deterministic rasterization before this boundary.
An uploaded page can enter directly as pixels.

### Earlier precursor: Retinal Flow V7

![Retinal Flow Language Model: ordered image fixations become a recurrent visual field, a rectified-flow writer generates candidate ink, and the candidates are reread and fed back](publication/ilm-image-native/figures/retinal_flow_paradigm.png)

The earlier runnable model is an 11.69M-parameter **Retinal Flow Language
Model**, a concrete
read-predict-write-reread loop:

1. A small convolutional retina reads ordered `32x32` grayscale fixations.
2. A three-layer recurrent visual field integrates the fixation history.
3. A continuous energy function scores arbitrary candidate images; it has no
   character output table.
4. A conditional rectified flow writes the next fixation directly in pixel
   space.
5. The model rereads its generated ink, selects a candidate by visual energy,
   and feeds those pixels back into the recurrent state.

### Measured status, not a capability claim

V7 kept the model at `11,690,244` parameters, added 800 updates on one RTX
4090, and generated `25.3` visual cells per second in its matched run. It added
normalized context advantage against independent image anchors and
backpropagation through sampled flow endpoints. V6 and V7 were tested on the
same 512 common Han characters, four font views, 2,423 eligible held-out
contexts, and frozen bank SHA-256.

| Gate | V6 closed loop | V7 selected step 5,800 | Interpretation |
|---|---:|---:|---|
| Retina oracle top-1 | `98.18%` | `98.27%` | Basic cross-font perception is not the main bottleneck. |
| Full-context top-1 | `1.20%` | **`2.31%`** | V7 beats last-only (`2.02%`) and unigram (`1.86%`), but not bigram (`13.58%`). |
| Normalized context log-probability gain | `-0.9066` | **`-0.2155`** | The calibrated deficit shrank by 76%, but full history still lowers mean target probability. |
| Generated context cosine gain | `+0.0077` | **`+0.0303`** | V7 passes the held-out generated-signal gate. |
| Late/early autonomous ink | `1.168` | `1.050` | Both loops keep nontrivial ink without late occupancy drift. |
| Sparse autonomous cells | `18.75%` | `15.63%` | V7 is denser, but its continuation is still unreadable. |

![Matched V6 and V7 autonomous comparison](publication/ilm-image-native/figures/anchor_identity_v7_result.png)

**Verdict: V7 is rejected as a language model.** It establishes a useful
training correction, not a complete language system. Raw target energy was
positive while normalized target probability was negative, proving that raw
score margins were an invalid acceptance measure. V7 does not prove readable
continuation, historical question answering, efficiency over a text LLM, or
Qwen-8B parity. The result motivates the Predictive Visual Field separation
shown above.

## Reproduce The V21 Field-Complete Test

The two fixed arms must be trained separately. They use the same frozen V16
retina and exactly matched parameter counts:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/train_field_complete_writer.py \
  --pvf-checkpoint artifacts/predictive_visual_field_v16_memory_pilot/checkpoint_step_0002200.pt \
  --route-mode field_complete \
  --out artifacts/field_complete_writer_v21_field_evidence_20260813

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/train_field_complete_writer.py \
  --pvf-checkpoint artifacts/predictive_visual_field_v16_memory_pilot/checkpoint_step_0002200.pt \
  --route-mode tiled_global_control \
  --out artifacts/field_complete_writer_v21_control_evidence_20260813
```

The paired evaluator requires two selected checkpoints. It deliberately rejects
the measured candidate because no candidate checkpoint passed selection; it
cannot access the frozen partition. Full hashes, metrics, and the expected
rejection command are in the
[V21 result receipt](docs/field-complete-writer-v21-result.md).

## Reproduce The V20 Topology Test

The prior V20 commands, exact hashes, and expected paired-evaluator rejection
remain in the
[`V20 result receipt`](docs/retinal-topology-router-v20-result.md).

## Audit The V19 Spatial Test

Reproduce the fresh V19 development audit. The evaluator verifies the clean
global-baseline hash and cannot access the sealed frozen split:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/eval_spatial_motor_plan_development.py \
  --checkpoint artifacts/spatial_motor_plan_v19_pilot/checkpoint_latest.pt \
  --out artifacts/spatial_motor_plan_v19_step1600_development_audit \
  --samples 128 --batch-size 32 --num-workers 8 \
  --sample-count 32 --sample-columns 8 --device cuda --precision bf16
```

The fixed protocol, full training commands, hashes, and failed gate are in
[`docs/spatial-retinal-motor-plan-v19-result.md`](docs/spatial-retinal-motor-plan-v19-result.md).

## Run The Visual Motor Plan

Audit the selected V18 checkpoint on fresh development renderings. This command
cannot access the sealed frozen split:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/eval_visual_motor_plan_development.py \
  --checkpoint artifacts/visual_motor_plan_v18_pilot/checkpoint_selected_development.pt \
  --out artifacts/visual_motor_plan_v18_step1400_development_audit_v2 \
  --samples 128 --batch-size 32 --num-workers 8 \
  --sample-count 32 --sample-columns 8 --device cuda --precision bf16
```

The evaluator refuses to overwrite an existing receipt. Full training settings
and the sealed-development decision are in
[`docs/visual-motor-plan-v18-result.md`](docs/visual-motor-plan-v18-result.md).

## Run The V17 Baseline

Evaluate the selected V17 checkpoint once on its frozen record split:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/eval_visual_state_actuator.py \
  --checkpoint artifacts/visual_state_actuator_v17_pilot/checkpoint_step_0001600.pt \
  --out artifacts/visual_state_actuator_v17_frozen_eval \
  --samples 128 --batch-size 32 --num-workers 8 \
  --sample-count 12 --device cuda --precision bf16
```

The evaluator refuses to overwrite an existing frozen receipt. Full training
settings and the rejected readability audit are in
[`docs/visual-state-actuator-v17-result.md`](docs/visual-state-actuator-v17-result.md).

## Run The Predictive Visual Field

Evaluate a trained PVF checkpoint on the fixed image bank:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/eval_predictive_visual_field.py \
  --checkpoint artifacts/predictive_visual_field_v16_memory_pilot/checkpoint_step_0002200.pt \
  --out artifacts/predictive_visual_field_v16_step2200_eval \
  --device cuda \
  --precision bf16
```

The implementation, exact V16 continuation settings, checkpoint-selection rule,
and metric definitions are recorded in
[`docs/predictive-visual-field-v16-memory-result.md`](docs/predictive-visual-field-v16-memory-result.md).
Training and evaluation artifacts remain git-ignored.

## Run The Retinal Precursor

Build the provenance-bearing public-domain Chinese manifest:

```bash
PYTHONPATH=. python scripts/build_visual_grammar_manifest.py \
  --wikisource-root ../Books/resources/curated-books/chinese-classics/public-domain-canon \
  --out data/visual_grammar/chinese_wikisource_public_domain.jsonl
```

Train the current combined RFLM objective from scratch on one 24 GiB GPU:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/train_retinal_flow_lm.py \
  --manifest data/visual_grammar/chinese_wikisource_public_domain.jsonl \
  --out artifacts/retinal_flow_chinese_anchor_identity \
  --sequence-length 48 \
  --energy-positions-per-sequence 8 \
  --batch-size 32 \
  --maximum-steps 6000 \
  --context-anchor-bank-size 512 \
  --context-anchor-views 4 \
  --context-advantage-weight 0.5 \
  --context-advantage-margin 0.5 \
  --sampled-identity-weight 0.2 \
  --sampled-identity-steps 2 \
  --rollout-start-step 800 \
  --rollout-ramp-steps 400 \
  --rollout-batch-size 8 \
  --rollout-steps 2 \
  --rollout-candidates 2 \
  --rollout-sample-steps 2 \
  --precision bf16
```

The exact measured V7 continuation command, frozen-bank receipt, and autonomous
comparison are recorded in
[`docs/retinal-flow-v7-anchor-identity-result.md`](docs/retinal-flow-v7-anchor-identity-result.md).

Run the strict fixed-bank evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/eval_retinal_flow_lm.py \
  --checkpoint artifacts/retinal_flow_chinese_anchor_identity/checkpoint_latest.pt \
  --bank-size 512 \
  --prototype-views 4 \
  --evaluation-samples 3000 \
  --generation-contexts 192 \
  --out artifacts/retinal_flow_chinese_anchor_identity/fixed_glyph_bank
```

Generate an autonomous image continuation from typed or image input:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/infer_retinal_flow_lm.py \
  --checkpoint artifacts/retinal_flow_chinese_anchor_identity/checkpoint_latest.pt \
  --text '天地玄黃，宇宙洪荒。日月盈昃，辰宿列張。' \
  --new-cells 32 \
  --candidate-samples 8 \
  --out artifacts/retinal_flow_chinese_anchor_identity/autonomous_demo
```

The primary inference artifact is `complete_page.png`; `receipt.json` records
the model boundary, parameter count, throughput, VRAM, font hashes, every
candidate-selection step, and early/late autonomous trajectory summaries.
Generated checkpoints and data remain git-ignored.

The earlier whole-page U-Net, latent diffusion, associative-memory, and causal
InkStream implementations remain as baselines. They are not the current model.

ILM is a research codebase for **language learned and generated as visible
writing**. Its current experiment predicts continuous retinal states with a
causal proposal and hyperspherical flow, then tests visual actuation separately.
V18 writes recognizable development forms through a deterministic spatial motor
plan. V19 shows that simply adding local retinal features as a residual does not
make those features causally responsible for topology. V20 forces fine topology
through the local field and verifies local causality. V21 then forces both
coarse occupancy and detail through that field and passes every causal and
algebraic invariant, but still rejects the writer on simple, medium, and overall
fidelity. The next bounded experiments must improve local raster continuity
without reopening a global drawing path and must separately learn prompt-image
to answer-image state transitions. The writer remains separate from the
still-sub-bigram language core. Older structured
embeddings, codebooks, and page diffusion
experiments remain available as falsified or comparative baselines; they do not
define the current model boundary.

> The repository intentionally keeps a practical etymology pipeline and long-horizon ILM experimentation side-by-side.

## 📌 Overview

This repository has three connected tracks:

1. Retinal-flow image-native language modeling and strict held-out evaluation.
2. Historic Chinese glyph etymology ingestion and provenance-preserving assets.
3. Earlier glyph, codebook, diffusion, folio, and InkStream baselines retained
   for reproducibility.

This README documents all three tracks and keeps the etymology workflow as a first-class, reproducible path.

## 🔗 Key Links

| Area | Path |
|---|---|
| Conceptual write-up | `docs/imagized-language-model.md` |
| Current engineering goal | `docs/first-imagized-language-model-goal.md` |
| V21 field-complete writer result | `docs/field-complete-writer-v21-result.md` |
| V20 topology-router result | `docs/retinal-topology-router-v20-result.md` |
| V19 spatial causal-test result | `docs/spatial-retinal-motor-plan-v19-result.md` |
| 2026 continuous-sensory research scan | `references/continuous_sensory_language_scan_2026.md` |
| V18 visual motor-plan result | `docs/visual-motor-plan-v18-result.md` |
| V17 causal actuator result | `docs/visual-state-actuator-v17-result.md` |
| V16 predictive visual-field result | `docs/predictive-visual-field-v16-memory-result.md` |
| V7 anchor-identity experiment | `docs/retinal-flow-v7-anchor-identity-result.md` |
| Closed-loop V6 experiment | `docs/retinal-flow-v6-closed-loop-result.md` |
| Research dossier and evidence | `references/image-native-language-model-research.md` |
| Archived diffusion plan | `docs/ilm-visual-diffusion-code-plan.md` |
| Archived embedding "color" plan | `docs/embedding-color-plan.md` |
| Historical development plan | `docs/development-plan.md` |
| Etymology module readme | `ilm/etymology/README.md` |

## ✨ Features

- 🏺 Etymology ingestion from `hanziyuan` and `chineseetymology`-style sources.
- 👁️ Continuous foveal retina with recurrent visual context and cross-font invariance.
- ✒️ Deterministic continuous visual motor plan for directly supervised stroke topology.
- 🖋️ Conditional pixel-space rectified-flow writer with a differentiable write-read cycle.
- 🔁 Autonomous image-only inference with candidate rereading, energy reranking, and pixel feedback.
- 🧭 Training on exact model-induced visual rollouts with state alignment, next-image energy, and recovery flow.
- 🧪 Fixed 512-character visual-bank evaluation against random, unigram, and bigram baselines.
- 🌐 Robust AJAX + HTML ingestion path with retries, throttling, and cache.
- 🧩 Stage-labeled glyph extraction including `<img>` and CSS `background-image` data URIs.
- 🗃️ SQLite-backed storage for chars/glyph metadata plus filesystem asset layout.
- 🖥️ Tornado web UI for ad-hoc ingest + gallery preview.
- 🔤 Glyph rendering utilities for multilingual token images.
- 🧠 Product-code style embedding/codebook modules.
- 🧱 Sentence frame packing and diffusion/inpainting training/evaluation scripts.
- 📊 Reporting and visualization scripts for embedding and pipeline inspection.
- 📄 Publication artifacts in LaTeX/PDF under `publication/`.

## 🧱 Project Structure

```text
.
├── README.md
├── AGENTS.md
├── configs/
│   ├── color.yaml
│   └── diffusion.yaml
├── docs/
├── i18n/
├── ilm/
│   ├── code/
│   ├── data/
│   ├── datasets/
│   ├── db/
│   ├── diffusion/
│   ├── encoders/
│   ├── english_tiles/
│   ├── etymology/
│   ├── frames/
│   ├── models/
│   ├── visual_lm/
│   └── utils/
├── scripts/
├── publication/
├── assets/
├── logs/
└── *.ipynb
```

## 🧰 Prerequisites

| Requirement | Notes |
|---|---|
| Python `3.10+` | Core runtime |
| `pip` | Package installation |
| Optional GPU | Helpful for PyTorch CUDA training scripts |
| Optional LaTeX toolchain | Needed for publication builds |

Assumption note: there is currently no single root dependency lock/spec file (`pyproject.toml`, `requirements.txt`, etc.), so dependencies are inferred from imports and script usage.

## ⚙️ Installation

### Minimal (etymology toolkit)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install requests beautifulsoup4 tornado
```

### Extended (modeling/training workflows)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install requests beautifulsoup4 tornado pyyaml numpy pillow matplotlib torch fonttools
```

If a specific script needs additional packages, install them from the import error shown by that script.

## 🚀 Usage

### Quick Start: Historic Glyph Ingestion (CLI)

1. Hanziyuan (recommended): char-only AJAX flow

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --site hanziyuan --char 中
```

2. ChineseEtymology (direct URL)

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --site chineseetymology --url "https://www.chineseetymology.org/CharacterEtymology.aspx?characterInput=%E4%B8%AD"
```

3. Batch file ingestion (lines can be `char\turl`, `url`, or `char url`)

```bash
PYTHONPATH=. python scripts/ingest_etymology.py --from-file urls.txt
```

### Outputs

| Output Type | Location |
|---|---|
| Files | `data/historic/glyphs/<char>/<stage>/<label>.<ext>` |
| Cache | `data/historic/cache/*.html` |
| DB | `data/historic/etymology.sqlite3` |

### Web Demo (optional)

```bash
PYTHONPATH=. python scripts/serve_etymology.py
```

Open `http://127.0.0.1:8888`, choose site, enter a character (for example `中`).

### Polite Crawling and Site Respect

- The fetcher uses per-host throttling, retries with backoff, and caching.
- Keep delays `>= 0.5s`, avoid bursts, and honor site terms/robots/licensing.
- Do not bypass paywalls or interactive protections.
- If you see `403`/`429`, slow down and retry later.

### Additional ILM Workflows

These scripts exist and are actively part of the repo surface, but they are research workflows and may require prepared local datasets/checkpoints.

1. Data download/prep

```bash
python scripts/download_alpaca.py --outdir data/raw
python scripts/download_corpora.py --out data/raw
python scripts/sample_paragraphs.py --out data/processed/test_100.jsonl
python scripts/build_images_common_freq.py --out data/processed/images_common_freq --size 128 --en 5000 --zh 5000
```

2. Glyph DB lifecycle

```bash
python scripts/glyphdb_init.py --db data/glyphdb/glyphs.sqlite3
python scripts/glyphdb_ingest_index.py --db data/glyphdb/glyphs.sqlite3 --index data/processed/images_common_freq/index.tsv
```

3. Code/color model training

```bash
python scripts/train_color_codes.py --config configs/color.yaml
python scripts/train_codes_from_qa.py --en-json data/raw/alpaca_en.json --zh-json data/raw/alpaca_zh.json --epochs 1
python scripts/train_ilmglyph_codes.py --en data/raw/alpaca_en.json --zh data/raw/alpaca_zh.json --out artifacts/ilm_glyph_train
```

4. Diffusion/inpainting

```bash
python scripts/train_diffusion.py --config configs/diffusion.yaml
python scripts/train_inpaint_frames.py --ckpt-code artifacts/ilm_glyph_train/ckpt_epoch1.pt --out artifacts/inpaint
```

5. Evaluation/reporting

```bash
python scripts/eval_color_codes.py --checkpoint artifacts/color_codes_e1.pt
python scripts/eval_diffusion.py --checkpoint artifacts/diffusion_unet.pt
python scripts/eval_qa_retrieval.py --checkpoint artifacts/color_codes_qa.pt
python scripts/report_ilmglyph_pipeline.py --ckpt artifacts/ilm_glyph_train/ckpt_epoch1.pt --lang en --text "hello world"
```

## 🧩 Configuration

Primary YAML configs:

- `configs/color.yaml`
  - data path: `data/processed/images_common_freq/index.tsv`
  - model/code params: `d_glyph`, `d_code`, `K`, `C`, temperature/anneal
  - optimizer/log settings

- `configs/diffusion.yaml`
  - input JSONL: `data/processed/test_100.jsonl`
  - frame/grid + model size settings
  - train mask ratio range and checkpoint settings

Override settings via CLI flags where supported (`--epochs`, `--batch-size`, `--lr`, etc.).

## 🧪 Examples

- Build a single English tile glyph:

```bash
python scripts/build_english_tile_glyph.py "language" artifacts/language_tile --save-tensor
```

- Run inpainting demo with trained checkpoints:

```bash
python scripts/inpaint_demo.py \
  --ckpt-code artifacts/ilm_glyph_train/ckpt_epoch1.pt \
  --ckpt-inpaint artifacts/inpaint/ckpt_epoch1.pt \
  --lang en \
  --text "the quick brown fox jumps" \
  --mode infill \
  --out artifacts/inpaint_demo
```

- Bulk ingest common characters from Hanziyuan:

```bash
PYTHONPATH=. python scripts/bulk_ingest_hanziyuan.py --limit 200 --resume
```

## 📝 Development Notes

- This is a research repository with both robust CLIs and exploratory artifacts (including notebooks and prototype scripts).
- Generated large files are intended for `data/` and `artifacts/` (both ignored in `.gitignore`).
- Publication source and PDFs are under `publication/`; helper build script: `scripts/latex_build.sh`.
- Collaboration/process conventions are documented in `AGENTS.md`.

## 🛠️ Troubleshooting

- `ModuleNotFoundError: ilm...`
  - Run scripts from repo root.
  - Use `PYTHONPATH=.` for scripts that expect local package resolution.

- `FileNotFoundError` for data/index/checkpoints
  - Run prerequisite data/build scripts first.
  - Confirm defaults such as `data/processed/images_common_freq/index.tsv` and `data/processed/test_100.jsonl` exist.

- CUDA/device issues
  - Switch to CPU with script flags/config (`device: cpu` or `--device cpu`).

- Missing package errors
  - Install required dependency from the specific script import path (`torch`, `pyyaml`, `Pillow`, etc.).

- HTTP `403` / `429` while scraping
  - Increase `--delay`, retry later, and keep requests polite.

## 🗺️ Roadmap

- Implement continuous next-retina flow as the primary language distribution; keep pixel flow as a conditioned actuator.
- Add fast, line, and page visual states only through measured ablations, starting with the smallest causal state flow.
- Require full visual context to beat last-fixation, unigram, and bigram baselines.
- Require stable, readable 32-cell autonomous continuations before scaling width or corpus size.
- Add multiscale page memory and provenance-gated historical glyph composition only after the causal gate passes.
- Improve environment reproducibility with one authoritative dependency specification and focused tests.

For deeper conceptual and staged planning details, see:

- `docs/imagized-language-model.md`
- `docs/ilm-visual-diffusion-code-plan.md`
- `docs/development-plan.md`

## 🤝 Contributing

- Follow `AGENTS.md` for conventions (atomic commits, push after change, no credentials in code).
- Group related edits in focused commits with conventional messages.
- Prefer reproducible script invocations with explicit flags and input paths.
- For scraping-related changes, preserve throttling/cache behavior and site-respect constraints.

## ❤️ Support

| Donate | PayPal | Stripe |
| --- | --- | --- |
| [![Donate](https://camo.githubusercontent.com/24a4914f0b42c6f435f9e101621f1e52535b02c225764b2f6cc99416926004b7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f446f6e6174652d4c617a79696e674172742d3045413545393f7374796c653d666f722d7468652d6261646765266c6f676f3d6b6f2d6669266c6f676f436f6c6f723d7768697465)](https://chat.lazying.art/donate) | [![PayPal](https://camo.githubusercontent.com/d0f57e8b016517a4b06961b24d0ca87d62fdba16e18bbdb6aba28e978dc0ea21/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f50617950616c2d526f6e677a686f754368656e2d3030343537433f7374796c653d666f722d7468652d6261646765266c6f676f3d70617970616c266c6f676f436f6c6f723d7768697465)](https://paypal.me/RongzhouChen) | [![Stripe](https://camo.githubusercontent.com/1152dfe04b6943afe3a8d2953676749603fb9f95e24088c92c97a01a897b4942/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f5374726970652d446f6e6174652d3633354246463f7374796c653d666f722d7468652d6261646765266c6f676f3d737472697065266c6f676f436f6c6f723d7768697465)](https://buy.stripe.com/aFadR8gIaflgfQV6T4fw400) |

## 📄 License

No top-level license file is currently present in this repository.

Assumption note: treat the project as research code with unspecified licensing until a `LICENSE` file is added by maintainers.
