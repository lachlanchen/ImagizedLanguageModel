# Broad synthetic first-function transfer fails

The [registered experiment](glyph_first_function_transfer_protocol.md) is
complete. It is rejected as a writing repair; the deployed checkpoint is
unchanged. No new autonomous answer run follows this failed gate.

Calibration requests all 126,213 supported source units per round, including
multi-character units. The two training rounds accept 125,813 and 125,814
states; the withheld-base round accepts 125,810. All 4,087 supported forms occur
in each accepted round. Every retained state is verified by the full offline
source projection. These vectors are counterfactual, not natural image states.

Training completes 18,000 updates of the 27,288,576-parameter continuous
readout and decoder-input MLP. Execution takes 247.64s including 80.08s of
pressure waiting, with 5.052GiB peak allocated CUDA memory. Calibration adds
326.38s, and inherited training/pretraining costs remain separate. Independent
CPU inspection confirms all twelve actual AdamW tensor counters are 18,000 and
all frozen model tensors and contextual adapters remain bit-identical.

| Qualification | Balanced parent | Transferred writer |
|---|---:|---:|
| Withheld synthetic mean foreground F1 | 0.12617 | 0.15690 |
| Withheld synthetic form-macro F1 | 0.34716 | 0.43436 |
| Withheld synthetic fraction reaching F1 .95 | 21/125,810 | 54/125,810 |
| Actual training-first mean F1 | 0.99956 | 0.98955 |
| Actual training-later mean F1 | 0.99985 | 0.99969 |
| Cached held-wording copy mean F1 | 0.97410 | 0.95775 |

The function gate fails by a wide margin. Synthetic image mixtures and damaged
forms remain visible. The copy states here come from the earlier unadapted
reader; these numbers do not replace an actual adapted-reader copy test.

A separately registered [retrospective check](transferred_first_images_protocol.md)
uses all 128 previously observed real first states, with no new training or
source projection. On the second cohort's identical parent/refinement initial
states, first-image F1 reaches .95 for **10/32** with the transferred writer
versus **17/32** for its balanced parent. Canonical first images decrease from
six to three. Those two identical state sets are not independent replications.
On the first contextual cohort the counts are 10/32 versus 13/32; on the
unadapted-control states they are 0/32 versus 6/32. These are writer checks on
known development states, not complete-answer or fresh language accuracies.

Broader source-unit coverage and freeing the decoder-input MLP are insufficient
under this recipe. Verified source argmax alone does not make a synthetic state
a useful training context. A subsequent [margin summary](evidence/glyph_first_function_transfer_margin_v1.json)
finds a withheld median top-two gap of 0.99998, with 2.51% below 0.1 and a
minimum of 3.24e-5. Near-boundary cases exist, but these measurements do not
establish boundary proximity as the cause of failure. The synthetic states'
relevance to natural image-conditioned states remains unproven.
The result motivates collecting more real image-conditioned training states;
it does not establish that this next repair will succeed.

All sixteen fixed preview boards were actually viewed. The
[archive](evidence/glyph_first_function_transfer_v1/archive.json) retains every
binary qualification and retrospective image, physical scores, complete source
coverage/failure counts, training history and independent checkpoint audit.
Large synthetic vectors and neural weights stay local with recorded hashes.
