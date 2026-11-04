# Completed vision diagnostic and post-hoc transcript bounds

5 November 2026. The fixed Qwen3-VL reader recognizes 122/128 training
reference answers exactly (95.313%), every blank/noise control empty and
valid JSON on all calibration controls. It then transcribes all 1,015 unique
anonymous actor images using the unchanged prompt and settings.

Three actor responses are invalid. Two reach the 256-output-token limit;
the third has an unterminated string. The original diagnostic therefore
reports `invalid_actor_transcription`, without a complete fixed-case score.
No response is retried, repaired or substituted. The primary V62 rejection
and the original diagnostic remain immutable.

The separately documented [post-hoc sensitivity plan](v62_partial_transcript_bounds_plan.md)
was written after invalid hashes were observed, before linking them to
conditions/gold and computing aggregates. The CPU analysis rechecks every
source prediction hash, anonymous PNG byte sequence and reader request hash.
It keeps all 256 cases in every condition. Invalid responses receive possible
score range [0,1]; no case is removed or labelled automatically incorrect.

| Condition | Valid / unknown cases | Exact transcript agreement (%) | Bounded character accuracy (%) |
| --- | ---: | ---: | ---: |
| K=1 | 254 / 2 | 0.781–1.563 | 1.388–2.169 |
| K=1, blank question | 255 / 1 | 0–0.391 | 0.186–0.577 |
| K=8 | 256 / 0 | 0 | 2.806 |
| K=8, blank question | 256 / 0 | 0 | 2.945 |

Ranges are full-case identification intervals, not confidence intervals.
The K=8 normal-minus-blank character difference is -0.139 percentage points,
with article-bootstrap sensitivity envelope [-0.962, 0.638]. The K8-minus-K1
character difference has identification interval [0.637, 1.418] points and
envelope [-1.256, 2.876]. Neither envelope resolves a positive gain.
The bootstrap uses 2,000 article resamples, seed 20266379; it does not cover
systematic reader errors, training-seed variance or reused-development bias.
Transcript agreement is an auxiliary instrument, not ground truth language
understanding or a replacement primary qualification.

Actual evaluator work consists of 1,172 calls, 157 unique calibration images
plus 1,015 actor images, in 1,092.1 seconds including startup and cleanup.
Maximum GPU-0 resident memory sampled after requests is 7,278 MiB, not an
allocation peak measurement. The owned server and runner exit, port 11437
closes, and GPU 0 returns to 16 MiB resident memory. The teacher stays outside
the ILM actor and its training budget.

Receipts are under
`artifacts/correlated_native_density_v62_20261105/vision_transcript_diagnostic/`:
`report.json`, `cleanup.json`, `calibration_report.json`, raw responses,
and `partial_bounds/report.json`. The analysis makes zero model calls.
It supports the existing conclusion that the tested density improvement
does not establish useful question-dependent writing. General ILM work
remains active.
