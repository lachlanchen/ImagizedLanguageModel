# Pixel likelihood and free writing

2026-11-05, written while V61 trains; its protocol is unchanged.

[MIXAR v2, Sections 3.3 and Appendices K–L](https://arxiv.org/html/2604.11575v2)
reports refinement on the model's own pixel rollouts, distinguishes readable
forms from correct continuations, and limits the transcription judge's view
to avoid question leakage. It also reports sensitivity to font changes.
These findings motivate evaluating actual free answers, not teacher-forced
loss alone. The pinned local release is not assumed identical to every
checkpoint reported in that paper. Its public code link currently returns
404 through the browser; cached source and weight hashes identify our model.

For a binary image vector Y, minimizing independent-pixel BCE gives
p_i = E[Y_i | H]. Consider a conditional distribution supported only on
1100 and 0011, equally likely. Every marginal is 0.5; uniform thresholding can
produce 0000 or 1111, neither supported. Correct marginals need not yield a
valid joint image. This elementary counterexample requires no physical analogy.

V61 therefore tests an honest likelihood baseline before adding complexity.
If free writing fails despite lower loss, a separately registered correlated
pixel decoder or rollout-aligned objective becomes justified. No current
threshold, checkpoint or objective changes follow from this note.
