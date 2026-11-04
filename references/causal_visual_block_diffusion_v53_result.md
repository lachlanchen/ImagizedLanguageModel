# Direct conditional visual block diffusion: measured result

Date: 2026-11-04. Decision: reject as a complete language generation mechanism.

## Measured outcome

The 22,177,219-parameter image-input/image-output model completed 8,000
updates on GPU 0 in 361.592 seconds, with 0.543 GiB peak allocated training
memory. The trained conditional denoiser has 5,899,970 parameters. The reader
remained frozen. The new trainer passed a byte-identical continuous versus
interrupted/resumed smoke comparison. The field-loss gradient correction and
synchronous-noise amendment were frozen before optimization.

The audit used 512 development windows for denoising, 256 for one primary
four-cell generation and 64 for two-block visible feedback. It took 21.205
seconds and peaked at 0.193 GiB allocated memory. The evaluator bank was
instantiated only after generation. Sealed records were not opened.

| Measurement | Ordered | Shuffled | Zero condition |
| --- | ---: | ---: | ---: |
| first-position identity | 1.171875% | 1.171875% | 0% |
| block-mean identity | 0.585938% | 0.488281% | 0.195313% |
| block pixel F1 | 0.319814 | 0.317569 | 0.301784 |
| blank rate | 0.195313% | 0.097656% | 0.195313% |
| isolated-ink fraction | 0.005016 | 0.005280 | 0.005338 |
| generated/target ink ratio | 1.195869 | 1.196849 | 1.163355 |
| distinct nearest evaluator identities | 476 | 479 | 466 |

The block ordered-minus-shuffled difference is 0.097656 percentage point.
Its 2,000-resample source-record bootstrap interval is [0, 0.307377]
percentage point. The first-position paired difference is exactly zero.
High-noise denoising cosine is 0.853008 ordered and 0.852636 shuffled, a gain
of only 0.000373. The model passes seven of twelve registered gates, including
surface quality and a positive feedback difference, but fails the five gates
for high-noise context advantage, first-position identity/advantage, block
identity and pixel F1.

The raw sheet contains coherent written forms and pseudo-glyphs. It is a
substantial surface change from the speckled reader plan but the rows are
largely unrelated to their targets. It does not establish useful language.

## Measurement limits

The block contains punctuation and less frequent forms as well as Han
characters. Only 60.94%, 58.98%, 64.45% and 68.36% of the four target positions
are represented exactly in the 1,024-Han evaluator bank. Out-of-bank targets
receive no credit. This coverage defect limits the identity measurement, but
cannot explain the identical first-position ordered/shuffled result or the
weak high-noise condition gain. Pixel F1 and raw images cover every target.
Changing the bank after seeing results would require a separate diagnostic
and cannot revise this run's frozen decision.

Stochastic target matching is not equivalent to argmax ranking. Even a correct
conditional sampler has expected agreement sum_x p(x|H)^2, whereas a mode
selector has accuracy max_x p(x|H). The earlier reader's 19.97% top-1 is not a
valid expected generation threshold. This distinction must accompany claims
about the reader/writer gap.

One training seed and a repeatedly used development partition do not support
broad statistical or generalization claims. Bootstrap intervals quantify
source-record variability conditional on this checkpoint, not training
uncertainty.

## Next solution test

The measured advance is clean visual synthesis with weak context use. A
further whole-model redesign is not yet justified. The next diagnostic should
ask whether the trained writer can reproduce an intended image at all when
an exact visual field and plan replace the uncertain first-cell proposal.
This is a target-supplied motor ceiling, not autonomous language. Compare it
with the exact inverse transform and the ordinary predicted condition using
paired noise. If the writer fails even that supplied-image task, train and
qualify image-conditioned copying before spending more language-training
compute. If it passes, the next learning target is a conditional distribution
over full image representations, with direct access to the visible history.

The model needs to learn a correspondence from source pixels to generated
pixels before it can be credited with independently selecting meaning. The
compact manuscript and next protocol must keep those two tasks measurable.

## Evidence

- Training: `artifacts/causal_visual_block_diffusion_v53_20261104/training_summary.json`
- Locked report: `artifacts/causal_visual_block_diffusion_v53_20261104/development/development_report.json`
- Raw pixels: `artifacts/causal_visual_block_diffusion_v53_20261104/development/generated_image_evidence.pt`
- Gallery: `artifacts/causal_visual_block_diffusion_v53_20261104/development/target_ordered_shuffled_zero_plan.png`
- Checkpoint SHA-256: `bd8dd3e58483d4ef527ae9a0fc4db1c0df545d18ec0fa0067ff594256d7cc71b`
