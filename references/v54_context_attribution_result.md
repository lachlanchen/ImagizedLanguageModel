# Exact-raster context attribution: post-result evidence

Date: 2026-11-05. Read-only exploratory diagnostic, not a new acceptance test.
The original V54 decision stays **reject**. No new model was trained.

The 512 primary development windows cover 169 source records. Per-row NLL
decomposition reproduces saved ordered/shuffled per-example NLL within
0.00000763 nat; zero-condition reconstruction differs by at most 0.00001526.
The production checkpoint, trained sources and corpus pass their hashes.

| Condition | Teacher-forced NLL (nats/image) |
| --- | ---: |
| Ordered natural history | 15.21873 |
| Earlier shuffled, last four cells retained | 15.25424 |
| Complete condition from another source | 20.62988 |
| Zero condition | 47.93050 |
| True supplied image | 0.18983 |
| Supplied image from another source | 1954.87195 |

The model uses its visual condition: paired ordered advantage over a complete
condition swap is 5.41115 nats, source-record bootstrap 95% CI
[4.51361, 6.41640], 2,000 draws. The earlier-ordering advantage remains
0.03551 nat, CI [-0.10142, 0.16932]. This supports context sensitivity while
remaining compatible with a short-suffix or order-insensitive mechanism.
It does not prove exclusive dependence on the last four cells.

Rows 1–8 contain 8.64381 of 15.21873 ordered nats. Whole-condition swapping
adds 2.36456, 1.52965, 1.12674 and 0.39019 nats over the four fixed eight-row
bands. Earlier-shuffle effects are +0.11443, -0.00585, -0.06657 and -0.00650.
Do not select positive bands or treat their intervals as independent
confirmatory tests. These are model-specific likelihood differences, not
mutual-information estimates.

The correct supplied field makes the whole image nearly deterministic, while
the wrong supplied field is strongly penalized. That is motor conditioning,
not language. Under natural conditioning, already-observed target strokes
explain much of the later image probability. The raw autonomous outputs still
contain incorrect forms.

The next learning intervention should preserve the exact copy path and the
demonstrated context sensitivity while improving global future-form choice.
A coarse-to-fine image probability law is one testable candidate; it must be
compared with the current raster law under the same natural-language and
generation controls. This diagnosis alone does not establish that candidate.

Artifacts: `artifacts/exact_raster_density_v54_20261104/context_attribution/`
contains the raw per-example rows, donor indices, report and figure. A portable
report and Figure S1 accompany Supplementary Note 12 in `publication/ilm-nature/`.
The diagnostic takes 20.46 seconds, uses only physical GPU 0, opens no sealed
records and leaves all weights unchanged.
