# Practical image input and output for the codec-language actor

The V68 trainer and automatic evaluator are active. Their model, protocol,
runtime loader, scoring and coordinator files must remain unchanged. This work
adds a separate image-file command that reuses the completed-actor loader only
after training and the independent autonomous audit finish.

The command admits one explicit page or binary strip. Pages retain their
grayscale pixels, EXIF orientation and white-composited alpha, with right/bottom
padding to the native 32-pixel boundary. It performs no automatic resizing,
OCR or character segmentation. Strip mode explicitly reuses the lossless
training fold. Reject inputs above 512 native visual cells, the tested
512-by-1024-page resource witness. This limit does not imply scanned-page
language capability or cover every possible aspect ratio.

Save the exact owning prompt tensor, its PNG and preprocessing receipt. The
actor receives only that tensor. Save all raw generated blocks, native stream
PNG, individual block PNGs and actual stop probabilities; retain whitespace
and distinguish capped, stopped and empty answers. Never synthesize a PNG for
an empty answer. Completed output directories are immutable.

CPU validation will compare the file interface with all 384 real prepared
normal/control prompt tensors and exercise serialization using the preserved
two-block actor execution and its real events. Additional boundary cases are
file-format validation only, with no mock actor, updates, answer scoring or
GPU allocation. Normal inference must reject unfinished training before
CUDA initialization. Actual final-model inference remains untested until the
queued audit completes; no standalone-export or language-quality claim follows
from this interface work.
