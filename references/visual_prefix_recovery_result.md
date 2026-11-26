# Can a visual writer recover from its own mistakes?

A compact Chinese–English writer learned to preserve agent, patient, voice and polarity across changes of typeface. On natural passages it produced readable ink and often changed the meaning. Giving it experience of its own errors during training preserved the narrow construction. All three visual writers scored zero on 144 fresh natural translation requests.

| Strict complete meanings | Requests | Starting writer | Reference-prefix fit | Draft-prefix fit | Qwen3-8B Q4, text | Reference | Shuffled foil |
|---|---:|---:|---:|---:|---:|---:|---:|
| New literary passages | 48 | 0 | 0 | 0 | 22 | 48 | 0 |
| New TED passages | 48 | 0 | 0 | 0 | 29 | 44 | 0 |
| New FLORES articles | 48 | 0 | 0 | 0 | 45 | 48 | 0 |
| Previously fitted book passages | 48 | 0 | 0 | 0 | 28 | 47 | 0 |
| Previously fitted TED passages | 48 | 1 | 1 | 1 | 34 | 47 | 0 |

Each held pair was translated in both directions. A stopped reply needed acceptance in two masked orders from a local quantized bilingual judge. It evaluated a deterministic readback from the emitted glyph repertoire. The comparison used typed source text for Qwen. The literary test covers four works in the local ZhJpBook library (*Emma*, *Hamlet*, *Bel-Ami* and *Dead Souls*). The other two groups each cover 24 separate documents. The fitted sample matched the lengths of held book and TED passages. All three visual writers stopped on every request. No fresh reply reproduced a complete reference bitmap sequence. [Full scores, paired votes and exact output counts](evidence/prefix_recovery_v1/results.json)

## A suffix can repair only part of a draft

If a writer has emitted a fixed prefix, the smallest attainable edit distance to a target equals the smallest distance from that draft to any prefix of the target. Copying the suffix at a minimizing cut attains the bound. Every minimizing cut occurs within twice the draft length. This result identifies the training label after an insertion or omission and an unavoidable error floor for evaluation. It does not assign meaning to a glyph distance. All 231,260 training cuts matched an independent edit-distance implementation. [Machine-checked Lean proof](../formal/span_list/PrefixRecovery.lean) · [Independent audit](evidence/prefix_recovery_v1/alignment-audit.json)

One 1,688,480-parameter writer received previous glyphs emitted by its frozen starting model. Its matched control received the closest reference prefixes. On 231,337 bilingual training pairs, half of the directions were assigned a draft. The starting writer produced 1,749,012 draft glyphs and differed from its aligned references in 204,580 drafts. Each fit ran 5,574 updates across 49,519,224 supervised positions. Reference-prefix training took 110.7 seconds and draft-prefix training 104.7 seconds across the three passes on an RTX 4090 D. Peak allocated device memory was 1.58 and 1.65 GiB respectively. These costs exclude ancestor training, evaluations and corpus preparation. Development chose pass one and pass three respectively. Greedy writing uses picture input, a learned optical encoder and selection among bitmap output prototypes. The released decoder has a finite output repertoire. [Registered comparison](visual_prefix_recovery_protocol.md) · [Weights and local inference](../models/visual_prefix/README.md)

For 143 requests, the starting model emitted at least eight glyphs. All three models continued from exactly those eight. The initial ink committed at least 776 edits across the panel. Above this floor the starting writer incurred 4,252 edits, the reference-prefix model 4,015 and the draft-prefix model 3,890. The paired gain of 0.87 edits per request had a document-bootstrap 95% interval from −0.37 to 2.44. With clean aligned prefixes the two fitted models achieved three exact completions each. Because neither fit improved free semantic credit, the recovery model fails the promotion rule. [Paired edit estimate](evidence/prefix_recovery_v1/edit-effect.json)

The recovery writer gave 472/472 exact active and 465/472 exact passive answers on controlled English prompts. Completing all eight requests in a family gave 110/118 for the parent, 103/118 for the matched control and 111/118 for recovery. The same totals held with the learned reader on Noto Serif. The released outputs include the new *Emma* errors alongside a correct constructed passive and the single credited training example. [Raw model-generated PNGs](../publication/ilm-nature/figures/prefix_recovery)

## Check on the bilingual labels

“He is my grandfather” received the correct freely generated Chinese reply. Its corpus target adds a second clause absent from the English. We then drew a separate random 96 book and 96 talk pairs from training, excluding role-rehearsal replacements. A bilingual model judged both orders equivalent on 91 book and 84 talk pairs, inconsistent on four and seven, and discrepant in both orders on one and five. It accepted eight simple controls twice and rejected 24 shuffled pairs twice. These judgments share the limitations of model-based review. The random sample identifies some faulty supervision and supports testing capacity on the many apparently aligned pairs. [Pair-level audit](evidence/prefix_recovery_v1/alignment-meaning.json)
