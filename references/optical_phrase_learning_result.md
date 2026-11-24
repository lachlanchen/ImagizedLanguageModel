# Can copying a printed phrase help a reader write a sentence?

On the first new English cohort the longest-run writer recovered the date in “The Battle of Cape Finisterre occurred in July 1805.” It also copied “by how much?” into a sentence asking for a quantity. The actual replies, the single-glyph control and the matching question and source PNGs are released for each example. The two new training objectives failed their registered whole-answer qualification. [English examples](evidence/phrase_region_v1/examples.json) · [Original comparison and criteria](phrase_region_protocol.md)

## Two controlled single-GPU fits

The reader accepts a pictured question and a sequence of aligned pictured source regions. A fixed learned selector chooses a region. The recurrent writer can draw a single glyph from its learned bitmap inventory or copy 1, 2, 4, 8 or 16 visible cells before its next decision. Every emitted cell updates the state. Repeated visible origins of the same ink run share probability. Each fit used the same four complete passes through 89,263 human answers (357,052 presentations, 9,912 updates), with the original ordered schedule. The two run writers use the same architecture, initialization and greedy decoder. The single-glyph control has 805 fewer parameters. Both run systems deploy 913,932 parameters including the 47,522-parameter selector. On one RTX 4090 D, the longest-run writer consumed 22.2 kJ in 180 s and the all-path writer 22.9 kJ in 190 s, in addition to the shared selector (8.1 kJ, 70.5 s). These are training-stage device costs. [First fit](../models/phrase_region/fit.json) · [Second fit](../models/phrase_marginal/fit.json)

At each prefix the first fit teaches the longest correct run. For the second fit the objective sums the mass of every compatible path through the complete ink stream and its stop. The first Lean result gives the difference between total mass and one chosen path. The second gives the exact change in total finite-path mass under a single edge-weight perturbation in common integer units. Positive edge probabilities and normalization connect this to a derivative of log likelihood. A forward–backward implementation computes posterior edge credit. Independent autodiff agreed on all 42 parameter tensors for three real sequences (largest relative gradient error 4.52e-8). Neither identity guarantees that maximizing the training objective produces correct unseen answers. [Path proof](../formal/span_list/OpticalPhrasePath.lean) · [Edge proof](../formal/span_list/OpticalEdgeMass.lean) · [Actual gradients](evidence/phrase_marginal_v1/gradients.json)

## Independent paired passages

Each cohort contains 192 English QA2D statement questions and 192 Chinese CMRC extractive questions, two per passage per language. Training and preceding confirmation families, normalized questions and answers were excluded for the new fits. The two cohorts are disjoint. Exact below requires a stopped, whole ordered pixel sequence equal to an accepted human reference. Fact presence requires the short answer in the stopped reply and may coexist with a false statement. [First cohort](evidence/phrase_region_v1/confirmation.json) · [Second cohort](evidence/phrase_marginal_v1/confirmation.json)

| Cohort and writer | Exact English /192 | English fact /192 | Both English facts /96 | Exact Chinese /192 | Chinese fact /192 |
|---|---:|---:|---:|---:|---:|
| First, single glyph |14|44|4|0|12|
| First, longest run |3|62|11|0|29|
| Second, single glyph |11|43|5|0|3|
| Second, longest run |7|48|8|0|25|
| Second, all paths |10|33|4|0|5|

On first-cohort English the family bootstrap 95% intervals for longest minus single were −9.9 to −2.1 percentage points in exact images and +2.1 to +16.7 points in fact presence. Disabling the trained run writer’s multi-cell actions recovered ten exact images. On second-cohort English, the numbers of exact *pairs* were zero, one and one for single, longest and all paths. Among 188 English answers absent from the pictured question and source the exact counts were 11, five and ten. Of 43 compatible fact edits, both complete images matched on five, one and three pairs. The edited-answer semantic judge failed the human-reference and foil calibration; semantic equivalence across the paired edits remains unqualified. [First results](evidence/phrase_region_v1/answers.json) · [Second results](evidence/phrase_marginal_v1/answers.json) · [Blind review controls](evidence/phrase_marginal_v1/semantic.json)

## Where training changes the output

On the first cohort, now used as development, the all-path writer produced 0, 4, 4 and 8 exact English images at 0, 4,800, 7,200 and 9,912 updates. Counts of requested facts were 0, 41, 40 and 36. The 2,400-update checkpoint was unavailable. On the fresh second cohort English reference loss was 0.339 nats per glyph versus 0.341 for single-cell training, but in 176 cases with a different stopped output it preferred the human image only 19 times. It wrote no exact eight-glyph book continuation in 152 cases and no requested historical specimen in 260 cases. On a 16-image comparison Q4 Qwen3-VL-8B included the short fact on 12 queries, versus one for the all-path writer, at 4.86 kJ versus 0.107 kJ device inference energy. These outcomes do not establish a matched-quality efficiency advantage. [Trajectory](evidence/phrase_marginal_v1/trajectory.json) · [Likelihoods](evidence/phrase_marginal_v1/likelihood.json) · [Transfer](evidence/phrase_marginal_v1/transfer.json) · [Quantized comparator](evidence/phrase_marginal_v1/comparators.json)

## What happens when the correct fact is supplied?

| English input on the second cohort | Single glyph exact | All paths exact | Earlier fact-conditioned specialist exact |
|---|---:|---:|---:|
| Question and full page, ordinary selection |11/192|10/192|—|
| Question and human-marked region |11/192|11/192|—|
| Question and short human fact alone |5/192|5/192|34/192|
| Sibling question with the same fact, scored against original |0/192|0/192|0/192|
| Both original and substituted sentence, given respective fact |3/43|3/43|10/43|

The specialist included the supplied fact on 133/192 queries; changing to the sibling question left it in 126/192 replies. Only the correct question yielded any exact original sentence. The earlier specialist has a different training history and a short-fact input changes the end-to-end writers’ training distribution. Its ordinary two-stage pipeline with a *predicted* fact produced 11 normalized exact English statements on the same pages. The oracle result therefore motivates the next comparison: train a writer with explicit pictured facts and its own reader’s predicted facts on training documents, then ask whether it can preserve the fact and the requested relation on unseen pages with no supplied answer. The highest-priority criterion is autonomous paired answers, followed by grounded fact edits and transfer. [Intervention design](phrase_fact_intervention.md) · [Physical outcomes](evidence/phrase_marginal_v1/fact-interventions.json) · [Ordinary two-stage comparison](evidence/phrase_marginal_v1/comparators.json)
