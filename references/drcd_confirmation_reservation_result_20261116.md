# Frozen evaluation-only Chinese source reservation

**Source correction, 24 November 2026.** Later record checks established that the official DRCD test split was already reserved. The affected traditional-script evaluations describe observed outputs on exposed material. [Exposure audit and replacement evaluation](drcd_source_exposure_correction.md)

A new unscored source pool is reserved before the running model's new outputs
are inspected: 998 paired passages from377articles, containing3,485questions.
Every surviving eligible group is retained. No question/reference strings were
displayed, and no image, model output or score was produced. The reservation
does not establish language improvement or admit evaluation execution.
[Protocol](drcd_confirmation_reservation_protocol_20261116.md),
[registration](evidence/drcd_confirmation_reservation_registration_20261116.json),
[complete report and export check](evidence/drcd_confirmation_reservation_result_20261116.json).

The pinned official DRCD test source contains378articles,1,000paragraphs and
3,493questions. Its human annotations and Wikipedia-derived CC-BY-SA3.0 notice
are retained unchanged. The source revision is
`b944790de5af02c5fbb7cd9cb1473d27d169eebf`; both source files match their pinned
Git blob hashes and recorded SHA256s.
[Official source and notice](https://github.com/DRCKnowledgeTeam/DRCD/tree/b944790de5af02c5fbb7cd9cb1473d27d169eebf),
[primary dataset paper](https://arxiv.org/abs/1806.00920).

The declared screen covers all3,251paragraphs in both local CMRC splits,
7,017prepared Wikisource records and all10,735Chinese preference rows. Its
normalizer is used only for exclusion screening; reserved text remains in its
original traditional-Chinese form. The index contains3,267normalized known/viewed
titles,42,225document fingerprints and422,525exact64-character windows. Matching
titles, normalized documents, source windows or questions exclude the entire
DRCD article. The already viewed README example article is also excluded by
policy. This exact screen is not semantic or paraphrase decontamination, and
inherited foundation-model exposure remains unknown.

One article, containing two passages and eight questions, is quarantined. Its
known-title and exact-window reasons overlap; they do not describe two different
excluded articles. No annotation failures or additional unpaired paragraphs
occur. All6,970human answer spans in the retained pool are verified at their
original offsets. All3,485questions and their complete answer arrays equal the
authenticated raw source. There are no duplicate question or paragraph IDs.

Each retained passage has a fixed hash-selected pair of distinct questions
with disjoint normalized reference sets. The pool also keeps every other valid
question, yielding1,996fixed-pair questions within the3,485-question reservation.
No answer-length threshold, generation score or favorable-case selection is
used. An independent reparse verifies complete raw-source equality, reference
spans, pair membership, evaluation-only flags and retention of all groups from
nonquarantined articles. It does not rebuild the full overlap index or claim
independent semantic novelty.

The source reservation takes28.947seconds at0.158GiB peak process RSS. The
separate export check takes0.125seconds at0.044GiB. The source, implementation,
29normalizer resources, pool and quarantine hashes are bound. There are no
neural imports, model calls, optimizer updates, student/baseline generations or
scores. All115sources of the original live trial remain unchanged.

The ignored raw source lives under
`data/evaluation_only/drcd_confirmation_v1/source/`. The frozen pool and
quarantine are in `artifacts/drcd_confirmation_reservation_v1/`. Preserve them
as evaluation-only data: do not use them for fitting, reward/preference
calibration, hyperparameter selection or output repair. Do not open the raw
references during later blind output transcription. Preserve the original
pool hash even if new independent annotation evidence is added separately.

The next admitted model work remains the original serial white-target trial.
Its physical qualification and original autonomous pilot precede any execution
on this pool. A later confirmation run requires a separately committed image,
review, scoring, comparator, resource and decision contract. No current gate is
changed. This source addresses new-group Chinese reading; it does not replace
book continuation, historical/unencoded forms, cold independent execution or
matched-quality whole-model resource evidence.
