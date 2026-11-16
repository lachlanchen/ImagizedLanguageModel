# Complete-image preferences without a scalar reward model

The two completed reward calibrations reject the proposed scalar judges. An
offline preference objective is another way to supervise complete answers, but
it still needs reliable labels. No preference fit is admitted, and the original
white-target fit, sources, schedule and acceptance rules remain unchanged.

Direct Preference Optimization (DPO) fits preferences through policy likelihood
ratios under a preference model, without first fitting a separate reward network.
Diffusion-DPO applies this approach to image generation with a diffusion-specific
likelihood bound. Our physical autoregressive writer already exposes a normalized
conditional law, so its candidate objective would use actual image-trajectory
likelihoods rather than a diffusion surrogate. This is an application of existing
methods, not a new optimization theorem or a demonstrated language improvement.
[DPO, section 4](https://arxiv.org/html/2305.18290v2),
[Diffusion-DPO](https://arxiv.org/abs/2311.12908).

## Exact trajectory contract

For a prompt image X, fixed preferred and rejected image streams Y+ and Y-, and
an immutable reference policy, define

\[
 \ell_\theta(Y,z\mid X)=
 \sum_{t<T}\log(1-s_\theta(h_t))+
 \sum_{t<T}\sum_{j=1}^{128}\log q_\theta(b_{tj}\mid h_t,b_{t,<j})
 +z\log s_\theta(h_T),\qquad h_t=R_\theta(X,Y_{<t}),
\]

\[
 \Delta_\theta=
 [\ell_\theta(Y^+)-\ell_\theta(Y^-)]-
 [\ell_{\rm ref}(Y^+)-\ell_{\rm ref}(Y^-)],\qquad
 L=\operatorname{softplus}(-\beta\Delta_\theta)
   +\lambda L_{\rm supervised},\quad \beta>0,\ \lambda\geq0.
\]

The physical patterns b are lossless eight-pixel pieces, not character labels.
Each branch must reread that branch's actual preceding pixels. Source-history
states cannot substitute for rejected-history states. Sum every physical action
and actual stop/continue log probability; per-glyph or per-length averages define
a different objective. A cap is censoring, not an invented terminal action. A
finite-cap law includes censored outcomes, and the cap must leave room for each
admitted complete positive and its actual stop. No almost-sure stopping theorem
for the uncapped actor is asserted.

Reference-policy scores can in principle be computed serially and cached with
the exact policy, prompt, output, stop and image-geometry hashes. This could avoid
two resident bodies. Full-body execution, gradient equivalence, support and
single-4090 cost have not been tested. Increasing a preference margin also need
not increase the preferred answer's absolute likelihood or greedy quality. The
supervised anchor and unchanged complete-language evaluation remain necessary;
their inclusion does not guarantee success.

## A separate source is now prepared

The [pinned Chinese-dpo-pairs card](https://huggingface.co/datasets/wenbopan/Chinese-dpo-pairs/blob/800b8cc6c8a1f7fa17ba1ab5690e50b663d832ca/README.md)
describes GPT-3.5 translations of several upstream preference datasets and declares
MIT licensing. These are machine-derived preferences, not independently verified
human judgments. Preserve upstream attribution and applicable source terms.

The executed [preparation report](evidence/chinese_image_preference_preparation_20261116.json)
authenticates the 17,430,997-byte Parquet source and preserves all 10,735 rows,
including 10,535 null source IDs. Content hashes provide stable row identities.
An initial strict-string check failed on those null IDs; its original script,
empty partial export and failure record remain preserved. Only the nullable-ID
contract changed for the successful preparation.

Eighteen rows have identical normalized answers and one has an empty rejected
answer. The remaining 10,716 pairs are structurally eligible, with semantic
labels still unqualified. A deterministic split of exact normalized prompt
groups reserves 1,095 rows for prospective calibration and 9,640 for prospective
training; those totals include structurally ineligible rows. All raw strings
remain unchanged. An independent full-export check matches every raw record
against Parquet, verifies every row hash and confirms that exact prompt groups
do not cross the proposed partitions.

No exact normalized substring match to the 32 existing evaluation questions
appears in system plus prompt. This is a narrow quarantine check, not semantic,
passage or benchmark decontamination. Existing evaluated outputs and frozen
readings remain outside this proposed training source.

Only 1,787 pairs have both answers at most 128 Unicode code points; 6,085 fit
513 and 7,157 fit 640. The longest chosen answer has 5,775 code points. These
counts are not glyph counts, font-coverage results or language tokens. No answer
is truncated. A short-cap fit cannot claim to use this complete collection.

The successful preparation takes 11.082 seconds at 0.609 GiB peak process RSS;
the separate full-export verification takes 1.123 seconds at 0.112 GiB. The
earlier failed preparation's download time and peak RSS were not captured and
are excluded from those timings. There are no rendered images, model calls or
optimizer updates. The limits are 90 seconds, 1 GiB peak RSS and 24 GiB available
host RAM. This is source preparation, not a language experiment.

## Decision before neural work

Finish the unchanged white-target comparison first. Its physical and autonomous
language gates still decide that actor's result. Neither outcome automatically
launches preference training.

Before registering a paired-preference fit, qualify labels on training-only
calibration cases; independently check actual raster coverage and legibility;
preserve uncertain judgments; and establish complete-trajectory likelihood and
gradient replay on the actual body under the existing resource admission rules.
Clean alternative-answer pairs do not themselves cover malformed self-generated
writing. A calibrated actual-output control is still required to establish that
an optimization gain corresponds to better emitted answers. No old reward,
reading, uncertainty flag or evaluation threshold is repaired by these data.

If label or image qualification fails, reject this source for that fit. If it
passes, bind sources, complete-pair exposure, reference scores, coefficients,
caps, resource ceilings and the unchanged language evaluation before execution.
Report readable complete answers, paired-question correctness, stopping,
erasure dependence and book continuation separately from preference loss.

Preparation command:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python scripts/prepare_chinese_image_preferences.py --fetch \
  --output artifacts/chinese_image_preference_sources_new
```

The source cache is `data/external/chinese_dpo_pairs_v1/`. Each output directory
contains `pairs.jsonl` and `report.json`. Existing output directories are refused;
the pinned cache is reused without another download. These private data files
are ignored by Git. The preparation does not provide an inference answer lookup,
a Unicode output head, a qualified reward or an admitted training run.
