# Lossless input-padding comparison

Specified before derivative preparation, optimization or strip-image model
calls, 7 November 2026. The completed first paired-writing experiment remains
a failed reliability screen: OCR exact match improves from 0/8 to 2/8, but
at least three of its four seen examples visibly contain incorrect characters
or omissions. Its original 30/32 training criterion is already unattainable
within that output cohort. No general language model or useful efficiency
advantage is established.

The measured setup contains a separate source of computational overhead.
All 160 prepared copy inputs place their last nonwhite pixel by row 112;
every pixel below row 255 is exactly white. Their 1024-square canvases use
4,096 native reference positions. A fixed 1024-by-256 crop uses 1,024
reference positions and preserves every source ink pixel and its coordinates.
Padding the crop back to the original dimensions with white must reproduce
every RGB pixel exactly. Reject a page if the fixed crop removes any nonwhite
pixel. Targets and question content are unchanged. This is an input-format
comparison: changing the canvas can also change frozen-VAE boundary and
global-attention features, so it is not an identical-latent attention-mask test.

Reuse the completed full-canvas control at 512 updates. Start the strip arm
from the same pinned undistilled base, not from the adapted control. Keep
rank 16/alpha 16, seed 20268321, the first 32 training copies, batch one,
accumulation one, rate 1e-4, weight decay 0.01, uniform sigma and native velocity
MSE. Keep the exact generic condition and all target PNG bytes. The 512 chosen
training IDs and sigma draws must match the original journal. Record source
and code bytes; the core model/training implementation must remain unchanged.

Prepare all 160 derivative copy inputs, then cache the selected 32 training
and all 32 validation pairs through the native VAE. Run the same two-update
execution check at the new geometry before the 512-update run. Measure actual
wall time and allocated GPU memory; theoretical position reduction is not a
measured throughput or language-efficiency result.

Infer both the unadapted strip baseline and the trained strip model on the
same first four training and four validation IDs used in the original screen,
with the same seeds, 50 native steps, guidance one and 512-by-256 output.
Retain all outputs, including errors, so the preprocessing effect can be
distinguished from the training effect. Inference reads an input-only index,
source weights, the fixed condition and optional adapter; no source answer
or question-specific language embedding enters it.

Use the same separate reference calibration and OCR convention on the same
expected strings: the original calibration is 27/32 training and 28/32
validation references, with uncertain cases retained. Inspect all raw outputs.
Do not reinterpret the eight-case screen as a complete 64-case qualification.
The 30/32 train and validation requirements remain unchanged; only a promising
screen warrants full-cohort generation. Speed on unreliable copies does not
qualify efficient language modelling. Natural QA, question interventions,
longer writing, arbitrary forms and broader language remain unproved.
