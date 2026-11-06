# Source-reader content on the same failed image prompts

The offline source decoder and independent image writer share exactly the same
44 normal-image prefill memories. The source decoder nevertheless uses the
full layerwise visual/context KV history for continuation. These results
therefore measure imported source capability and the information boundary;
they do not qualify the ILM or prove the 52 final-layer states are sufficient.

| Source-helper route | Exact canonical answers | Owning reference included | First-piece prefix match |
| --- | ---: | ---: | ---: |
| copy/normal | 8/12 | 12/12 | 8/12 |
| copy/blank | 0/12 | 0/12 | 0/12 |
| copy/text | 6/12 | 10/12 | 6/12 |
| training/normal | 19/32 | 22/32 | 22/32 |
| training/text | 24/32 | 25/32 | 26/32 |

The corresponding V75 image writer produces 0/12 exact new-copy answers and
0/32 exact answers on this fixed training cohort. Those original outputs and
scores are unchanged. Helper exactness requires learned source termination and
outer-whitespace-trimmed equality to the canonical reference; quotations,
explanations and alternative wording are not silently removed. Inclusion and
prefix agreement are separate string diagnostics, not semantic accuracy.

There are 100 fixed source-helper generations: normal-image and source-text
controls for all 44 cases, plus fully blanked prompt images for the 12 copy
requests. Every source prompt and reference re-renders to the original physical
strip exactly. On all 44 normal-image calls, the complete 52-state BF16 suffix
matches the failed ILM reader memory byte for byte. The first source projection
uses exactly its last stored BF16 vector; subsequent helper decoding has the
full contextual KV advantage. It uses the authenticated source tokenizer,
token-row lookup and temporary output projection only inside this offline
evaluator. No symbolic decoder or teacher service enters the deployed ILM.

Independent CPU verification reconstructs all 56 source/transformed image
inputs, recompiles controls, checks every retained response and recomputes
literal metrics. It replays 725 source
projection decisions from retained hidden states; exact argmax agreements are
725, with
0 near-tie disagreements.
It does not replay the full vision/context body. All raw responses remain
unchanged, including wrong, quoted and explanatory outputs. No optimization
or sealed-data access occurs.

The helper stage takes 282.212 seconds, including
134.073 seconds setup, at
19.099 GiB allocated memory on GPU 0. This cost
excludes all external pretraining and earlier ILM/codec learning. The source
text route and image-only runtime have different interfaces; this is not an
efficiency comparison with a text language model.

The [receipt with all unmodified answers](evidence/reader_content_v77_result.json)
and [registered protocol](reader_content_v77_protocol.md) define the comparison.
Full retained input PNGs, states and decoding traces are under
artifacts/reader_content_v77_20261106/. These sources do not establish a useful
independent image-output language model.
