# Chinese errors persist on seen training prompts

After the complete primary experiment, all 96 declared fit-diagnostic calls
finish without another optimizer update. The learner produces **0/8 exact
Chinese copies** and **0/8 strict Chinese QA answers** on seen training
records. English copying is **8/8**; English QA is **6/8** against the
original annotation alternatives, or 5/8 against the particular CE target.
These seen-prompt results cannot establish generalization.

| Seen group, eight records each | Source image | Final learner image | Original-text source |
| --- | ---: | ---: | ---: |
| Chinese copying | 0 | 0 | 3 |
| English copying | 2 | 8 | 7 |
| Chinese QA, annotated exact stopped | 0 | 0 | 1 |
| English QA, annotated exact stopped | 3 | 6 | 7 |

All 96 outputs stop within the unchanged 96-decision cap. The
[prospective selection](native_state_fit_protocol.md) takes the first eight
records per group in the authenticated training inventory, before the main
learner's assessment. Image arms encode complete PNGs afresh, with matched
seeds. Original transcripts enter only the explicitly privileged text arm.
No reference or correct answer history enters any arm.

Strict scores require context. In the text control, four Chinese copies
contain the correct complete string inside extra quotation marks, three
are exact, and one is translated into English. Several text-control QA
answers also differ in language, surrounding prose or date format; one
gives `2000万港元` for the reference `2亿港元`. Its 1/8 strict QA score is
not a count of all semantically related answers. The criteria remain fixed;
there is no post-result semantic rescoring.

By contrast, all eight learner Chinese copying failures change the actual
characters. For example:

| Requested visible string | Unmodified learner response |
| --- | --- |
| 花多地地山 | 花多梅树山 |
| 风言外黑目家 | 风车外黑白色 |
| 北夏风左星目 | 北夏风正直目 |

Some learner QA responses contain a correct date with different surrounding
text, but others are clearly wrong: `第四部` becomes `大清国`, and `2亿港元`
becomes `1011年`. The data support a failure of reliable Chinese training
fit, without proving a unique architectural cause or total absence of
Chinese information. More held-out examples alone cannot explain away the
observed copying errors on these already-seen strings.

The [complete export](evidence/native_state_fit_result/result.json) retains
all 96 raw outputs, all 32 original visible transcripts, references, group
scores and source/checkpoint/code bindings. The main result remains
[separate](native_state_alignment_result.md). There are no new training
updates, continuous image answers or independent-ILM qualification.

```bash
env CUDA_VISIBLE_DEVICES= PYTHONPATH=. python scripts/probe_native_state_fit.py --stage prepare --output artifacts/native_state_fit/prepared_v1
env CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=. artifacts/external/native_unified_source/runtime/bin/python scripts/probe_native_state_fit.py --stage evaluate --output artifacts/native_state_fit/assessment_v1
env CUDA_VISIBLE_DEVICES= PYTHONPATH=. python scripts/probe_native_state_fit.py --stage score --output references/evidence/native_state_fit_result
```

Use fresh output paths for a reproduction, with a complete primary result
and unchanged final checkpoint. CMRC 2018/SQuAD 1.1 retain CC-BY-SA-4.0
attribution to Cui et al. (2019) and Rajpurkar et al. (2016); Show-o2
model/source are NUS Show Lab, Apache-2.0.
