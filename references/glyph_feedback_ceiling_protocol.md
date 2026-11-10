# Matched feedback counterfactual

The physical readout now passes the genuine-copy gate but fails most complete
development answers. Before another architecture or dataset change, compare
three teacher-controlled feedback routes from exactly the same image prefill:

1. Native source units with their exact source input embeddings.
2. The first complete character of each winning source unit, fed back through
   that character's exact source input embedding.
3. The same first-character decision rule, fed back through the frozen encoder
   of its canonical glyph image.

The first contrast tests character-sized steps against native segmentation.
The second tests replacing exact character vectors with learned image vectors.
Later trajectories may diverge; their difference is an intervention on the
complete feedback policy, not a matched-prefix estimate of local encoder error.
The first source-unit winner is also recorded alongside every route. The
learned image writer and learned stop head do not choose these helper actions.

All eight original training controls and all sixteen original normal Chinese
development cases are retained, in source order. No case is selected by result.
Each route starts from a fresh prefill and must match the previously recorded
candidate's first state to maximum absolute error below 0.002 with identical
image geometry. Each run allows 64 consumed actions and one final stop query;
native units and complete glyphs have different action lengths. The full native
head is rounded BF16 then evaluated in FP32, matching the previous source
reader helper. These are deterministic greedy trajectories, not sampled text.

Native terminal units stop the helper. A partial Unicode unit, missing first
form, or unsupported canonical form terminates a character route explicitly;
there is no substitution, case rejection, pending byte repair or output repair.
Native-unit decoding retains its source tokenizer, including partial units.
Visual aliases share the primitive encoder but exact-vector feedback uses the
particular emitted character's source row; this inherent ambiguity remains a
limitation of the comparison. Complete output strings and all termination
reasons are scored after all 72 trajectories finish. Literal reference matching
is helper performance, not an independently graded semantic accuracy estimate.

This is an offline diagnostic. The temporary source vocabulary head, tokenizer,
character map and canonical glyph bank are allowed only in the helper. They
must not enter the autonomous ILM runtime or be counted as generated ILM glyphs.
All records, including training controls, are marked unusable for subsequent
fitting. No C3 input or reference enters this experiment. The C3 checkpoints
are already frozen. The full target-free prefill, subsequent states and action
choices are preserved; no hidden state is fitted or optimized here.

Execution uses the existing exclusive GPU0 mutex, deterministic arithmetic,
22 GiB allocation cap, 24 GiB runtime host reserve, 300-second pressure wait
budget and a 2,400-second total cap. Preserve partial and admission failures.
Do not launch while the C3 model process is live. Authenticate the existing
source archive rather than downloading or saving a second model copy.

This comparison is motivated by the dependence of sequential predictors on
their own previous outputs, formalized in
[Ross, Gordon and Bagnell (2011)](https://www.ri.cmu.edu/pub_files/2011/4/Ross-AISTATS11-NoRegret.pdf).
Their guarantees do not apply automatically to this frozen body or its helper.
If image-feedback teacher trajectories remain useful but the learned writer
fails, the next candidate should learn the correct state-to-image action on
training-only genuine trajectories. If exact-vector trajectories work and image
feedback fails, encoder alignment needs repair first. If both character routes
fail, additional motor fitting cannot by itself establish native language
competence under character segmentation. These are prospective interpretations,
not results.
