# Attention adaptation preserves copying but does not repair QA

The complete 2,560-update attention learner retains **29/32 exact stopped
copies**: 13/16 Chinese and 16/16 English. Chinese and English natural QA
both remain **0/32**. Neither language has a both-correct question pair.
All six declared reader requirements fail. This is not a qualified reader
or an independent image-answer ILM.

| Fixed original-layout assessment | Dense source | Attention learner |
| --- | ---: | ---: |
| Chinese copying /16 | 13 | 13 |
| English copying /16 | 16 | 16 |
| Chinese QA /32 | 0 | 0 |
| English QA /32 | 0 | 0 |
| Both Chinese questions correct /16 | 0 | 0 |
| Both English questions correct /16 | 0 | 0 |

All 640 training IDs appear four times. The learned residuals affect only
112 q/k/v/o projections, with 4,358,144 trainable parameters. The original
dense visual frontend, including its learned visual adapters, remains
frozen; no source pixels are resized or pooled. Copying uses the original
1024-by-256 strips and QA uses the original 1024-square pages. This choice
precedes inspection of the separate canvas-control results.

The actual optimizer invocation takes **1,667.861 seconds** at **6.345 GiB**
peak allocated VRAM on one RTX 4090 D, excluding model loading and the
262.431-second prefix cache. The complete adapted model is 2,843,625,184
parameters plus a 126,892,531-parameter codec and external pretraining.
Small trainable residuals do not define the complete model size or establish
efficiency at useful language quality.

## Actual autonomous answers

The first two Chinese fish questions both receive `1981年`, against references
`印度西太平洋区` and `印度Pondicherry`. The first English title and person
questions both receive `1961`, against `An Unearthly Child` and `Susan Foreman`.
The percentage questions receive `1980s`. All normal Chinese answers stop,
but changing an unrelated year is not question answering. Eleven of sixteen
Chinese pairs and eleven of sixteen English pairs still receive identical
answers to their different questions. Erasing the question changes only
11/32 Chinese and 12/32 English normal responses. None is correct.

The real preflight passes: four disabled/zero-enabled source pairs replay
their retained decision sequences exactly; all 224 residual tensors have
finite nonzero gradients by update three; all 112 zero output matrices
change; disabled-source logits and the FP32 causal check have zero error.
Those execution checks do not override the failed language result.

The [complete export](evidence/native_reader_attention_result/result.json)
retains all 160 final responses, source calibration bindings, all optimizer
updates in compact form, cache metadata and the actual preflight. Its full
per-parameter gradient journal and numeric weights remain local with SHA256.
Final checkpoint SHA256:
`e923d7fd0a480d275884a8fa2cdcb5c4b1b82cfe23f9167d06bf7fb9e37ab9ab`.
The [prospective protocol](native_reader_attention_protocol.md) and
[execution guide](../docs/native-reader-attention.md) preserve the original
schedule, quality floors and claim boundary. No validation-driven checkpoint
selection or further optimizer follows this assessment.

## Consequence

This rejects this attention-only recipe as a demonstrated QA repair. It does
not uniquely establish that contextual attention is incapable of learning.
The [fixed-weight canvas control](native_canvas_control_result.md) shows a
remaining confound: padding alone changes both visual features and the
context the language body receives. The next diagnostic should separate
those components while preserving the known original copying path, before
another book-scale or writing run. More attention updates alone are not an
established repair. The full image-input/image-output goal remains open.

CMRC 2018/SQuAD 1.1 retain CC-BY-SA-4.0 attribution to Cui et al. (2019) and
Rajpurkar et al. (2016); Show-o2 source/model are NUS Show Lab, Apache-2.0.
