# Visual Future Block Language V48: Measured Result

Date: 2026-08-15

Status: production run and strict amended development audit complete; V48 does
not qualify

## Answer

V48 validates part of the proposed path but rejects its central mechanism as
a complete solution.

The 16.28M-parameter model trains efficiently on one RTX 4090, learns ordered
Chinese raster continuation, predicts a diverse next-image distribution,
removes V47's terminal-position collapse, emits finite nonblank rasters, and
passes 11 of 16 frozen gates. It does **not** beat V42's immediate language
state, does not make all later horizons outperform their host-only controls,
and does not preserve enough identity through deterministic inverse-DCT
rasterization.

The likely next path is therefore not more future heads or a larger causal
reader. It is an image-native **distributional writer and discriminative
visual metric**: retain the successful ordered visual reader, stop treating a
multimodal next-glyph distribution as one mean DCT field, and stop using a
background-dominated raster cosine as the sole predictive geometry.

## Frozen Evidence

| Evidence | SHA-256 |
| --- | --- |
| production checkpoint | `d281f8c8403d07b2662bc6d091145287f218727b3c8df2f2ea87da04c70165f3` |
| training summary | `fe658885bed9ebc10cb8ff4cf61a487c2b0334b6feabf3e2108059f101ed0c7b` |
| development report | `2d3e5857ce7c50c8b3b85f5056f7c9f26b3ed3b6eb162c68c3ad7ca5a94742fa` |
| development summary | `a1371bd7be804100e351e83e04b1df6d56dcd03fdb13cc89cdfc64d461f59d9d` |
| raster evidence sheet | `bce170e2e5342c0d7d61735e98bd14f8668c00583ac63f3c218bd3a8326aeeea` |
| corpus manifest | `76048753b52735d14c98f0d9e4eb8d751401fe8810e82eaaaebff517f6866c03` |
| evaluator bank | `a88509b3c1d3093e63dd1ceb77dcd86c7ef282c80927284b93c4ef09cd9456ad` |
| matched next-cell windows | `4665be2d5cf1714a21d6523ca697749456036d7719302171762bc30291f451f3` |
| four-future windows | `9ba79f8f7e196192cd07daa7438daa8a60665b78650dc1b3f4c3d441db561d03` |
| counterfactual pairs | `b1d62615aebeda8b48bc9ee526ad5362320bc95dd0a7be612a789a37b6e16e1a` |
| terminal windows | `e68db6c27b75a4d502a0dac7fe08487f1c5e3850d46ea1e561c7f18a5757e630` |

The strict report passes all 23 integrity checks. It loads only the final
update-10,000 checkpoint, reproduces V42's registered `0.19970703125` top-1 in
the same process, recovers every frozen corpus/window/pair/control digest,
keeps evaluator labels outside student calls, produces visible rasters before
candidate-bank scoring, feeds back only emitted pixels, reports finite
metrics, and leaves the frozen partition closed.

The first strict audit exposed a polarity bug in the integrity aggregate: it
stored the required fact `frozen_partition_opened: false` and then applied
`all(...)` to the dictionary. The committed
[audit erratum](visual_future_block_language_v48_audit_erratum.md) permits
only the equivalent positive invariant
`frozen_partition_remained_closed: true`. The amended evaluator verifies that
all five other checkpoint-registered sources are byte-identical and records
both evaluator hashes. No model, data, metric, threshold, or gate changed.

## Compute Result

| Measurement | Result | Frozen requirement |
| --- | ---: | ---: |
| trainable parameters | 16,278,401 | < 17,000,000 |
| training segments | 160,000 | fixed |
| production updates | 10,000 | fixed |
| training elapsed time | 1,369.04 s | < 7,200 s |
| peak training allocation | 0.4921 GiB | < 18 GiB |
| peak training-or-audit allocation | 2.1490 GiB | < 18 GiB |
| strict audit elapsed time | 89.69 s | reported |

The full production run took 22.8 minutes on CUDA device 0. The result rules
out GPU memory or runtime as the bottleneck for the next compact experiment.

## Immediate Visual-Language Result

| Development measurement | V42 | V48 | V48 minus control |
| --- | ---: | ---: | ---: |
| full-history top-1 | 0.19971 | 0.18066 | -0.01904 vs V42 |
| symbolic bigram top-1 | — | 0.12256 | +0.05811 |
| shuffled-history top-1 | — | 0.16162 | +0.01904 |
| ordered target log probability | — | -5.34618 | +0.16133 vs shuffled |

V48 passes the bigram, ordered-top-1, and ordered-log-probability gates. Its
full state uses order and long history. It nevertheless loses 1.90 percentage
points to V42, so dense future prediction did not improve the best immediate
visual language state.

The symmetric values are informative: V48 gains `+0.01904` over shuffled
history while losing `-0.01904` to V42. The model learned real order, but the
new objective traded away some of V42's stronger next-cell coordinate.

## Four-Future Result

| Horizon | V48 top-1 | Offset control | Gain | Distinct top-1 | Largest mode |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.15527 | 0.10742 | +0.04785 | 525 | 0.02979 |
| 2 | 0.04053 | 0.03613 | +0.00439 | 407 | 0.04053 |
| 3 | 0.01758 | 0.02686 | -0.00928 | 388 | 0.03906 |
| 4 | 0.03027 | 0.03174 | -0.00146 | 350 | 0.05615 |

Every horizon is diverse and no horizon collapses to one common output. Only
horizon 1 clears the required `+0.01` control margin. Horizon 2 is weakly above
its control, while horizons 3 and 4 are below theirs.

The strongest representation warning is the inverse relation between cosine
and identity:

| Horizon | Target-field cosine | Top-1 |
| ---: | ---: | ---: |
| 1 | 0.59368 | 0.15527 |
| 2 | 0.63102 | 0.04053 |
| 3 | 0.64450 | 0.01758 |
| 4 | 0.65811 | 0.03027 |

The later fields become *closer* under the fixed signed-raster cosine while
becoming less correct as glyph identities. Shared background and coarse ink
geometry dominate this metric. Multi-horizon angular and pixel losses can
therefore improve without learning the requested future language form.

## Order And Terminal Stability

The same-suffix counterfactual arm accuracy is `0.53223`, below the frozen
`>0.55` gate but above its shuffled result `0.52051`. Full context contributes
some useful information, not enough to qualify.

V48 completely removes V47's terminal-position defect:

| Visible length | Top-1 | Distinct predictions | Largest mode |
| ---: | ---: | ---: | ---: |
| 63 | 0.17383 | 589 | 0.02539 |
| 64 | 0.17432 | 590 | 0.02441 |

Top-1 changes by `+0.00049` rather than collapsing. Dense supervision at every
position solved the localized failure mechanism, even though it did not
improve the overall state.

## Visible Raster And Closed Loop

| Direct-raster measurement | Result | Gate |
| --- | ---: | ---: |
| proposal identity top-1 | 0.18066 | reported |
| visible-reread identity top-1 | 0.14893 | > 0.15, fail |
| identity retention | 0.82431 | >= 0.85, fail |
| proposal-to-visible cosine | 0.84631 | > 0.82, pass component |
| visible pixel F1 | 0.46614 | > 0.46, pass |
| blank rate | 0.00000 | < 0.02, pass |
| ink-density ratio | 1.60827 | reported |

Visible identity misses its threshold by only `0.00107`, but the result is not
rounded into a pass. More importantly, the evidence sheet shows that the
thresholded fields are dense speckled conditional averages rather than crisp
glyph samples. Pixel F1 is not sufficient evidence of readable language.

Four-step autonomous generation passes its frozen aggregate gate: mean
identity is `0.04395`, above image-unigram top-1 by `0.02979`, with no blank or
non-finite output. The per-step curve exposes severe error propagation:

| Generated step | Identity top-1 | Pixel F1 |
| ---: | ---: | ---: |
| 1 | 0.14453 | 0.46389 |
| 2 | 0.01563 | 0.42799 |
| 3 | 0.00391 | 0.41341 |
| 4 | 0.01172 | 0.41129 |

The first step dominates the aggregate pass. Once a deterministic average
raster is reread, its off-manifold noise rapidly destroys language identity.

## Frozen Gate Outcome

V48 passes 11 of 16 gates.

Passed:

1. image-only student/deployed boundary;
2. parameter budget;
3. VRAM budget;
4. runtime budget;
5. gain over symbolic bigram;
6. ordered top-1 gain over shuffled history;
7. ordered log-probability gain over shuffled history;
8. horizon diversity;
9. terminal stability;
10. direct-raster F1 and nonblank output; and
11. finite/nonblank closed-loop aggregate gain.

Failed:

1. V42 improvement (`0.18066` versus `0.19971`);
2. all-horizon control advantage;
3. counterfactual arm accuracy (`0.53223` versus `>0.55`);
4. direct visible identity (`0.14893` versus `>0.15`); and
5. proposal identity retention (`0.82431` versus `>=0.85`).

## Mechanism Diagnosis

Three observations now constrain the next model.

### 1. Point prediction is the wrong writer for a multimodal next image

Many glyphs are plausible after the same context. A deterministic unit-field
head minimizes its losses by moving toward a conditional average. The exact
inverse-DCT sign boundary turns small uncertain coefficients into widespread
pixel decisions. The resulting raster has surface overlap but is not a clean
sample from the future-glyph distribution.

### 2. Raster agreement and discriminative language geometry must be separated

The fixed DCT remains valuable because it is exact, cheap, invertible, and
works as a sensory/write carrier. Its raw signed-pixel cosine is not a good
standalone language metric: common white background and coarse ink structure
can increase cosine while identity worsens. The next model needs a continuous,
image-derived, foreground-balanced predictive chart while retaining a full
invertible raster chart for output. Neither chart requires tokens, Unicode,
OCR, a glyph inventory, or a codebook.

### 3. Autonomous training must include its own visible errors

Teacher-forced horizon offsets do not teach recovery from an emitted noisy
raster. Later open-loop heads and true closed-loop recurrence are different
problems. The next writer must train on corrupted/emitted image states and
learn to return to the manifold of readable glyph images.

## V49 Decision Boundary

Do not scale parameters, add fonts, add etymology supervision, or add an LLM
teacher yet. First preregister a smaller mechanism study with three parts:

1. derive an ink-balanced continuous metric from rasters alone and verify on a
   training-partition diagnostic that target identity no longer moves opposite
   to similarity;
2. keep the strongest single-next-image causal reader instead of forcing one
   shared point head to explain four increasingly uncertain futures;
3. train a compact conditional raster distribution—such as a binary
   denoising/flow writer—with explicit sampling and visible-reread recovery,
   no glyph candidate bank in training or deployment, and no symbolic target.

Only if that mechanism produces crisp image samples and preserves next-image
identity should later-horizon, page, instruction, historical-form, and
etymology data be introduced.

## Claim Boundary

V48 establishes that a compact image-only causal model can learn ordered
Chinese continuation, predict diverse continuous future-image fields, train
in under 23 minutes on one RTX 4090, avoid an absolute-position collapse, and
run a finite visible-raster loop without tokens or a deployed inventory. It
does not establish readable autonomous writing, a superior language model,
general semantics, instruction following, etymology answering, arbitrary
historical-form generation, human-like reading, or parity with Qwen.

