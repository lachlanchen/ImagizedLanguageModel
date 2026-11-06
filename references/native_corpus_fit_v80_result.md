# Native writing fails even with correct answer history

The completed native-context checkpoint does not reliably predict the writing
codes on the fixed set of 32 Chinese training questions. Giving it correct
preceding answer images and the reference answer length leaves all 32 images
inexact. Supplying the correct codes to the same frozen decoder reconstructs
the writing almost perfectly. Generated-history errors therefore cannot be
the sole cause of this model's malformed answers.

| Path on the same 32 training questions | Exact images | Mean foreground F1 |
| --- | ---: | ---: |
| Autonomous prompt-to-image generation | 0/32 | 46.235% |
| Correct preceding image codes and reference length supplied | 0/32 | 50.241% |
| Correct target codes supplied to the fixed decoder | 28/32 | 99.9953% |

The autonomous first-block foreground F1 is 71.544%. Free and correct-history
paths have identical first blocks in all 32 cases, before their histories
differ. All free streams stop, and all 32 reference-history stop traces agree
with the supplied continuation/termination positions. Stopping and some
recognizable writing do not establish an accurate answer.

The cohort is exactly the sixteen same-passage question pairs used in the
earlier writer diagnosis, authenticated before the native checkpoint was
available. Every case belongs to the native model's training set. Targets enter
the controls only after each free-generation call has finished. The complete
prompt KV is retained, and each actual prefill memory and initial state match
the original prompt-only training cache. Architecture and training histories
differ from the earlier writer; this is not a single-factor causal comparison.

At pure noise, the first-block predictor's mean target-code MSE is 0.7832 with
the owning question state and 1.3819 with its same-passage partner's state.
At the clean endpoint these values are 0.03024 and 0.08427. This sensitivity
shows that the condition affects prediction, while the actual sampled images
remain inaccurate. Constructed target-assisted denoising inputs are not free
sampling trajectories, and the comparison does not uniquely identify an
architectural or optimization cause.

The controls locate a failure in conditional code prediction under the tested
representation and training procedure. The learned visual-history interface,
context readout, code distribution and optimization remain possible causes.
The evidence does not justify another long run of the unchanged architecture
or a claim that increasing the corpus alone will repair it. A subsequent
intervention must first demonstrate reliable conditional writing on a fixed
small cohort, then retain that ability on new prompts. Training-set fit alone
would still not establish language understanding or the general ILM goal.

All 32 cases are visible in the unaltered comparison pages:
[00–07](evidence/native_corpus_fit_v80_raw/all_cases_00_07.png),
[08–15](evidence/native_corpus_fit_v80_raw/all_cases_08_15.png),
[16–23](evidence/native_corpus_fit_v80_raw/all_cases_16_23.png), and
[24–31](evidence/native_corpus_fit_v80_raw/all_cases_24_31.png).
Each case shows its reference, free answer, correct-history answer and
supplied-code reconstruction. The [receipt](evidence/native_corpus_fit_v80_result.json)
retains all per-case scores, source bindings, output hashes and the 128
unchanged individual PNGs. Independent CPU checks reconstruct all prompt
layouts, rescore free/control pixels, replay all 32 supplied-code decodes and
decode all retained correct-history coordinates. They do not replay the whole
inherited neural forward pass. No semantic transcription accuracy is claimed.

The successful GPU invocation took 167.57 seconds, including 105.36 seconds
of setup, with 17.675 GiB peak allocated memory and no optimization. The first
launch omitted the deterministic CUDA configuration and failed during prefill,
before any completed answer. Its error is retained. The corrected invocation
uses unchanged committed code and byte-identical prepared inputs in a separate
output directory; no failed launch is counted as a model prediction.
