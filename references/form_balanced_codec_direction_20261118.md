# Separate physical-form learning from language occurrence frequency

The failed global 64-bit codec preserves exact image identities in its saved
plans. The [saved-code analysis](evidence/binary_plan_information_20261118.json)
finds 9,581 different codes for 9,581 physical forms, no conflicting code and
no form with multiple codes. All 64 bits vary. Empirical H(form|code) is zero;
source-weighted code entropy is 9.681481 bits and the sum of marginal bit
entropies is 61.890036. Those correlated-bit statistics are not a transmitted
bit rate, semantic organization or proof of arbitrary-form capacity. A perfect
arbitrary decoder could distinguish this finite support, but the trained decoder
does not. Increasing code width is not justified by an observed collision floor.

The [frequency analysis](evidence/binary_codec_frequency_20261118.json) localizes
a different imbalance. Of 8,629 training forms, 5,821 occur once, comprising only
5,821 of 64,041 training cells. Their binary macro F1 is .823330 and exactness
.000344. At the other extreme, 110 forms occurring at least 65 times occupy
34,976 cells (54.6% of training mass); both codecs reconstruct these forms
exactly. Continuous once-occurring forms reach .953585 F1 and .034530 exactness.
These are descriptive post-hoc strata, not causal identification or new gates.
No Unicode label or original source image is changed, and no neural call occurs.

The next bounded intervention changes **only the codec training distribution**:
give each exact physical form equal exposure. A drawing subsystem should learn
rare structures as well as repeatedly printed common ones. This does not change
the natural language distribution for future contextual/prior learning. No
alphabet classifier or image lookup enters generation; hash groups are offline
sampling metadata only.

For G physical training forms with occurrence counts n_g and N source cells,
the original objective is sum_g n_g L_g/N. Use sum_g L_g/G instead. Complete-
source preflight can express it without omitting a cell through weights
w_i=N/(G n_g(i)). Actual fitting cycles one first-source representative of each
exact mask in a fixed permutation, using the same 768,000 draws and 6,000 updates
per arm. Identical source masks carry identical pixel information; retain every
original occurrence and evaluate all of them. No extra optimizer budget, latent
width, architecture, reconstruction term, precision or quality threshold changes.

Two CPU checks prove exact equal group mass, exclusion of physical validation
and agreement between weighted complete-source and uniform-form gradients in
both codec modes. The proposed comparison starts from the original common
tensors. Reuse the completed natural-frequency continuous/binary models as
unchanged controls, without re-executing them; train the new continuous and
binary pair serially under one owner. Uniform exposure may still fail to learn
spatial geometry or preserve held-form fidelity. It is a hypothesis about finite
training allocation, not a new learning theorem.

The 952-form physical partition has now been observed and is reused development,
not a new blind confirmation. Preserve all original thresholds and add an actual
.02 validation-F1 gain over the saved natural-frequency binary control. If the
new recipe fails fidelity or event cost, keep the failure and do not extend it.
If it qualifies, separately register conditional image generation; do not replace
the full QA/book, stopping, emitted-history, untouched-source, historical-form
or whole-model resource criteria. No full language fit is admitted.
