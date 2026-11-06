# Diagnose decoder margins in the retained correct-history predictions

This CPU-only diagnostic is specified while the adapted V81 arm is still
training. It evaluates the completed native V80 diagnosis and the completed
V81 frozen-transition control, on their original fixed 32 training questions.
It changes no running model, optimization, checkpoint, sampling rule or quality
gate. It does not select an intervention from the unfinished adapted outputs.

For every retained correct-history block, load its normalized predicted
continuous coordinates and the owning authenticated target codes. Decode both
with the same qualified frozen writing codec. The decoder returns raw logits
for white pixels; the deployed writer applies sigmoid, maps to [-1,1], then
thresholds at zero. Preserve that exact numerical classification rule.

Retain all raw decoded logits and every original block. Independently rebuild
all target rasters from the bit-packed training files. Rescore the saved GPU
images, not substituted CPU images. If CPU decoding disagrees with a saved
GPU bit, retain and count that disagreement with its logit magnitude. Never
repair the output or drop the case. Decoder-margin measurements remain
auxiliary and do not independently replay the contextual body or sampler.

For white target bit b and decoder logit l, use signed margin (2b-1)l. Report
all signed margins on erroneous saved-image pixels, separating the first block
from later blocks. Fixed bins are (-inf,-8,-4,-2,-1,-0.25,0,0.25,1,2,4,8,inf).
Also report target-code reconstruction margins, coordinate MSE, pixel errors,
foreground F1, ink/paper errors and class-balanced binary cross entropy. The
balanced diagnostic averages the means over the nonempty ink/paper classes;
it is not a new training loss. Retain empty-ink targets and all internal white
regions. Aggregate per-block means and complete counts separately; do not
substitute block averages for the existing full-answer score.

This measurement asks whether nearly correct initial writing and inaccurate
later writing have different physical error profiles. It neither proves a
unique cause nor guarantees that an auxiliary pixel loss would solve language
generation. No optimization, external service, full reader call or GPU context
is allowed. Both matched training arms and their original final verifier
continue unchanged. A final adapted result requires its own completed output
and evaluation before being included in any further comparison.
