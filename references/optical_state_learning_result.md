# A bilingual source representation on a glyph grid

A five-character image of 汤姆问道。 produced the continuous source field from which the frozen English decoder wrote “Tom asked.” The field came from a trained 3.22-million-parameter pixel reader; the decoder remained a pretrained translation model. On 96 complete translations the learned source earned 3 credits, the exchanged-image control 0, the original text source 35, the separate independent ILM 2 and quantized Qwen3-8B 76. All stopped. This was a reused four-work panel, with two-order automated source-only grading. The work-resampled 95% interval for learned versus exchanged input included zero. [Released reader and CPU example](../models/optical_state) · [Case-level measurements](evidence/optical_state_v1/evaluation.json)

## A conserved attention measure

If a query gives a phrase unnormalized attention weight w, its r glyph positions can each carry w/r. Subtract log(r) from their identical key logits. The normalized value sum then equals the original for every query, including when values have signed coordinates. Lean checks the exact finite identities at rational resolution. In the six real decoder layers on two development works, 480/480 free sequences and all 6,388 supplied-prefix top choices reproduced. One original output reached the length limit; its copies preserved that behavior. Naively repeating states without correcting their measure reproduced 146/480 complete outputs. The largest corrected reference log-probability deviation was 0.0000153. This control uses the real pretrained states and piece boundaries. [Proof](../formal/span_list/GlyphAttentionMeasure.lean) · [Protocol](glyph_attention_measure_protocol.md) · [Results](evidence/library_teacher_v1/glyph-measure.json)

## Learning states from the image

Each direction received 12 passes and 3,552 updates on one RTX 4090 D. Source images and two boundary vectors are the only sequence inputs to the four-layer visual transformer. For English output, development normalized state error fell from 0.736 to 0.387; for Chinese output it fell from 0.786 to 0.425. The two additional fits took 30.5 and 33.7 seconds. Generating the fixed supervisory fields took 8.0 and 9.1 seconds. Costs exclude training the 77.94-million-parameter teachers. Each readout decoder owns 58.77 million parameters, including its embeddings. The successful English translations were “Tom asked.” and “I think so too.” The third successful readout interpreted “What way?” in Chinese. The reader has no demonstrated general translation ability. [Training receipts](evidence/optical_state_v1/training.json) · [Free assessment](evidence/optical_state_v1/evaluation.json)

| Source intervention | English KL per piece | English unchanged outputs / 240 | Chinese KL per piece | Chinese unchanged outputs / 240 |
|---|---:|---:|---:|---:|
| Original field and mass | 0 | 240 | 0 | 240 |
| Original field, learned mass | 0.0094 | 168 | 0.0089 | 167 |
| Learned field, original mass | 0.7355 | 25 | 1.0015 | 7 |
| Learned field and mass | 0.7270 | 24 | 1.0032 | 7 |

The divergence is from the original corrected model to each counterfactual, conditioned on the real bilingual prefix. Substituting the learned field with exact weights changed the top prediction on 1,072/3,191 English and 1,418/3,197 Chinese decisions. Normalized source-state error for previously unseen pieces was 0.878 and 0.981, compared with 0.326 and 0.369 for pieces occurring at least 100 times. Coverage and the importance of individual state errors to the decoder both warrant direct tests. [Factorial intervention](evidence/optical_state_v1/factor.json)

The same two pretrained systems were tested as whole-proposition judges. Assignment likelihood cancels source-only and target-only scoring effects. The teachers recovered all 1,334 natural passage assignments in each direction, while the optical writer recovered 1,333. Both teachers recovered all 472 controlled polarity changes. The English-output teacher recovered only 320/472 subject–object exchanges, below the fixed 90% threshold. It is unsuitable as the sole training judge for participant identity. These are constrained comparisons with two proposed assignments. [Teacher protocol](bilingual_teacher_assignment_protocol.md) · [Teacher results](evidence/library_teacher_v1/assignments.json)
