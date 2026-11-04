# Complete-line training data result

The host-only renderer passes real bilingual rendering, ordinary-space width,
deterministic-image, white-frame and explicit overflow/control/unknown-glyph
checks in the existing Python 3.12 Cairo/GI environment. The fixed default is
Noto Sans CJK JP at 24 physical pixels. A 26-pixel prototype fails the protected
height condition for a mixed-script string with descenders; the default is
fixed before corpus preparation, with no per-record shrinking.

The complete training preparation accepts 3,193/3,211 CMRC records and
9,783/12,070 Dolly records. Every original source remains intact. Eighteen
Chinese records exceed the vertical ink area. English records have 2,192
width, 77 height and 18 control/layout rejections, with IDs and reasons retained.
All eight Dolly task categories remain represented. No source is truncated or
replaced by a missing-glyph box. Future development and sealed tests are not
read or rendered by this preparation.

The 12,976 image files occupy 340,894,893 bytes and take 86.986 seconds to
prepare on CPU, with zero model calls and GPU allocations. Each file stores
losslessly packed physical image bits and the prompt's spatial block boundary.
Every retained prompt/answer/termination image passes exact pack/unpack checks;
all files have SHA-256 entries. First fixed Chinese and English PNGs are
visually inspected, including the complete English spaces and Chinese upper
strokes. The candidate cache is
`data/general_instructions/complete_lines_train_20261105/`.

The font file hash is
`b76b0433203017ca80401b2ee0dd69350349871c4b19d504c34dbdd80541690a`;
Pango is 1.52.1. Source and implementation hashes are in the cache report.
Accepted-manifest hashes are
`c5eab6220b08bda3568e4054f5e35b331582584e6952c266ec085f13fea71d9e`
(CMRC) and
`c3d9d82e73553efe55a3787b60189ac96ddee246fdf742848d0406b170dac3ac`
(Dolly). Existing source licenses and grouping limitations remain unchanged.

`CompleteLineImageData` successfully decodes actual rows from both sources
without the renderer, fonts or source text, exposing only image tensors and
the supervised-loss boundary. The example origin-of-言 prompt is a host input
render, not a generated answer or evidence of etymology competence. The new
proportional interface is a candidate for later registered training; it changes
neither V62's data nor any completed model-quality result. The original ILM
goal remains active.
