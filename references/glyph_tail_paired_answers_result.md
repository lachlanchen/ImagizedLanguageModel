# Contextual adaptation improves image answers but remains unreliable

Both registered conditions completed 18,000 optimizer updates and the same
72 autonomous image-answer cases. The final four contextual layers were
adapted in one condition; the other trained only the image readout and stopping
network. Both used all 3,176 admitted short-answer training questions. The
complete [paired evidence archive](evidence/glyph_tail_paired_answers_v2/archive.json)
retains all 144 output PNGs, all 52 reviewed boards, qualification pixels,
output-first readings, reference scores and checkpoint identities.

| Complete-answer measure | Contextual adaptation | Readout only |
|---|---:|---:|
| Clear literal matches, eight training controls | 8/8 | 8/8 |
| Clear literal matches, sixteen earlier development questions | 6/16 | 4/16 |
| Clear literal matches, question-erased counterparts | 0/16 | 0/16 |
| Clear literal matches, thirty-two prospectively reserved questions | 16/32 | 11/32 |
| Both different questions correct within a reserved context | 5/16 | 2/16 |
| Reserved answers reaching the 64-glyph cap | 11/32 | 14/32 |
| Bit-exact canonical complete images, reserved questions | 2/32 | 1/32 |

On the reserved questions, eleven clear correct answers were shared, five
occurred only under contextual adaptation, and sixteen were unsuccessful in
both conditions. This small, single-seed comparison is descriptive. It does
not establish a reliable general language model. The preset pilot requirement
was at least 24/32 clear complete matches and at least 10/16 contexts with both
answers correct; neither condition passed. All sixteen context pairs had
disjoint allowed reference strings. Question erasure was tested only for the
earlier development questions, not the reserved cohort.

All 36 raw-output boards were actually inspected before either condition's
reference strings were opened for scoring. Damaged or uncertain complete
streams retained null readings. Twenty-seven ambiguous stopped rows were
additionally inspected through nearest-neighbour pixel enlargements. Readings
were made by the agent, not independent human graders. Literal matches require
a natural stop, a clear complete reading and equality to an allowed reference;
canonical image equality is reported separately. An intelligible but longer
answer was not retroactively accepted by changing the reference rubric.
One of the thirty-two reserved questions had no fully supported canonical
reference in the existing physical evaluator; it remains in every denominator.

## The unresolved first decision

The training-only position audit found first-glyph mean foreground F1 of
0.893582 for contextual adaptation and 0.880031 for the control. Later glyphs,
conditioned on correct preceding images, reached 0.974584 and 0.966604. The
first decision precedes any generated-image feedback, so feedback drift cannot
explain all the failures. Later teacher-conditioned fidelity also cannot stand
in for autonomous language accuracy.

There were 870 distinct first target forms, including 449 occurring only once
as a first target. All 3,176 physical training inputs were unique, with no
contradictory labels for identical input pixels. The 13,500 question updates
averaged only 4.25 sampled sequences per question; 4,500 other updates rehearsed
copying. A sequence containing several easy teacher-conditioned continuations
can dominate the image loss while its harder first decision receives little
training. More exposure and explicit first/continuation balancing are therefore
testable repairs, not demonstrated solutions. The earlier native-unit feedback
diagnosis remains relevant to later trajectory drift but is not a complete
explanation of initial-glyph errors.

## Runtime and scope

Actual generated glyph pixels were encoded back into the causal body. The
registered runtime used no lexical classifier, tokenizer, font renderer,
canonical glyph bank, reference answer or external teacher. A cold standalone
run on the prospectively fixed first reserved input reproduced all pixels,
states, stop decisions and feedback exactly. Its system-call trace recorded
no IP connection attempt or access to the forbidden corpus, font and lexical
bank paths. This checks one runtime execution, not arbitrary correctness.

The source body retains 7,522,463,984 inherited parameters and fixed compiled
task vectors; its pretraining exposure is unknown. The experiment trained
688,128 contextual adapter parameters plus 17,848,833 writer/stopping parameters
on one 4090 D. It is not a small foundation model trained from scratch. Both
conditions failed the unchanged training-pixel qualification threshold of
0.98 mean F1 (0.963524 and 0.954784), although the held-copy thresholds passed.
Running the registered answer assay despite that failure did not waive it.

Measured generation time over all thirty-two reserved cases, including failed
ones, was 130.020 s and 144.046 s. These times exclude model loading and output
saving. They are not a matched throughput comparison with a text LLM.

The present result supports partial image-conditioned language transfer and
identifies specific remaining failures: damaged first forms, incorrect answer
selection or length, and unstable closed-loop continuation. It does not yet
support open-ended writing, unseen fonts or historical forms, a lightweight
independent foundation, or equivalence to a general-purpose LLM.
