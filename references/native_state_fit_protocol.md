# Seen-prompt fit diagnosis for the state-alignment learner

This diagnostic is fixed while the 2,560-update optimizer is still running,
before its autonomous assessment. It cannot change the primary 160-case
assessment, its six gates, the checkpoint, training budget or learning recipe.

Select the first eight records, in the authenticated training inventory's
existing order, from each of four groups: Chinese copying, English copying,
Chinese QA and English QA. Keep all 32 records regardless of model outcomes.
They are seen training prompts, so their scores cannot establish generalization.
The learner has seen each four times when this test runs.

Evaluate three arms on every selected prompt: the frozen native source from
the whole-image 432 view; the final learner from the same 432 view and complete
1024 source; and the frozen source from the original visible text. Use the
existing generic instructions, FP32 sampled codec, BF16 body, identical
per-record latent/readout seeds and a 96-decision top-one readout with native
EOS. Original text is a privileged capacity control. Image arms receive no
transcript, annotated answer, answer history, cache state target or training
sample latent. They encode the actual PNGs afresh.

Record primary training-target exactness and annotation-alternative exactness
separately, retaining case and punctuation. Chinese normalization removes
whitespace; English normalization collapses it. Report all raw outputs and
EOS outcomes. There is no model selection or further optimization.

GPU calls wait until the main controller, all primary after-readouts and the
additional source-equivalence check have completed and the final primary
score exists. Failure on seen prompts indicates a training-fit obstruction;
success there with held-out failure instead indicates a transfer/generalization
obstruction. Neither result identifies a unique mechanism. No arm supplies
continuous image answers or completes the independent ILM goal.

CMRC 2018 and SQuAD 1.1 passages and annotations retain CC-BY-SA-4.0
attribution to Cui et al. (2019) and Rajpurkar et al. (2016). The source
model and implementation are NUS Show Lab's Show-o2, Apache-2.0. Source
pretraining may include public evaluation material; this is a local diagnostic.
