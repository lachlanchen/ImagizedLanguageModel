# Joint image likelihood improves forms but misses the physical prerequisite

The complete joint-writer comparison is finished. Its original frozen review
credits 970/1,040 selected candidate forms (93.27%), compared with 599/1,040
(57.60%) for the unchanged point-writer control. A later pixel/blank-flag audit
finds seventeen review errors, including four empty images that were credited;
these original counts are not corrected physical accuracy. See the
[blank-flag audit](joint_writer_blank_flag_audit.md). Its registered rule requires
at least 90% in **every** candidate group. Development books and training
fixed-layout QA fail, so this actor stops before autonomous evaluation. Its
complete-language quality is **unmeasured**, not zero.

| Supplied-history group | Joint writer | Point-writer control |
|---|---:|---:|
| Development books, fixed | 91/108 (84.26%) | 13/108 (12.04%) |
| Development QA, fixed | 248/257 (96.50%) | 145/257 (56.42%) |
| Development QA, proportional | 245/257 (95.33%) | 238/257 (92.61%) |
| Training books, fixed | 100/108 (92.59%) | 22/108 (20.37%) |
| Training books, proportional | 100/108 (92.59%) | 21/108 (19.44%) |
| Training QA, fixed | 90/101 (89.11%) | 72/101 (71.29%) |
| Training QA, proportional | 96/101 (95.05%) | 88/101 (87.13%) |

All 33 opaque boards and 2,080 selected candidate/control images were actually
inspected. Readings, blank flags and uncertainty were frozen and separately
sealed before owner-map access. The independent scorer re-enumerated every
registered position and checked the raw arrays, individual pages and board
pixels. Review used one model reviewer with prior cohort familiarity and no
independent human adjudication. Readable alternatives to source glyphs count
as physical successes; the table does not measure canonical accuracy or meaning.

The new 2,642,368-parameter head completes three full epochs: 41,298 retained
updates over 13,766 sequences and 6,108,741 scheduled target positions. All five
prescribed heads are retained. The independent CPU audit verifies all source
exposures, inverse-packed target pixels, training-only normalization and actual
Adam counters. Exact step-three restore/replay passes; two additional replay
updates do not add retained exposure. The reader, contextual dynamics, image
feedback and stop predictor remain frozen.

Cumulative coordinator-measured child execution is 40,002.756 seconds for cache
preparation, including the original failed attempt and CPU recovery verification;
1,982.697 seconds for fitting; 184.130 seconds for the CPU audit; and 133.027
seconds for qualification. The fit worker reports 1,973.557 seconds including
170.164 seconds of pressure waiting, at 0.644 GiB peak allocated VRAM. The latter
is a **head-only** allocation, not a complete deployed-model footprint. The
preserved reader has over 7.522 billion inherited visual/contextual parameters;
the small new head does not establish a compact foundation or matched-quality
efficiency. Earlier reader training and inherited pretraining remain separate.

After sealing the readings, an array inventory identifies the changed failure:
fifteen of seventeen uncredited selected development-book outputs are truly
empty; two nonempty outputs were marked blank, and two other empty book outputs
were credited. The earlier point writer predominantly produced dense forms.
Four of the 108 selected source images are themselves empty, three overlapping
blank outputs. Neither that fact nor a
recount excluding whitespace changes the registered failed gate. The candidate
has no output above 500 ink pixels across all 12,597 supplied-history positions;
many nonempty images are still incorrect. For example, only 236/1,161 complete
development-book positions match their source pixels exactly. Legible writing
and correct conditional language remain distinct.

The [aggregate export](evidence/joint_writer_physical_result_20261113.json)
binds the fit, audit, qualification, frozen review, original verdict and costs.
Reproduce the export without a model call:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  PYTHONPATH=. python scripts/analyze_joint_writer_blank_density.py verify-original
```

No unchanged fit, intermediate-head selection or threshold relaxation is
authorized. The next [bounded probability diagnosis](joint_writer_blank_density_protocol.md)
separates an empty-image density preference from a greedy decoding failure
before choosing a new intervention. Its outcome is not implied by this result.
