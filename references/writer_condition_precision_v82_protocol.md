# Audit possible precision loss at the writing-condition interface

Auxiliary diagnostic specified 7 November 2026, while the matched V82 loss
experiment trains. Its data are the completed V81 checkpoints and retained
correct-history states. No V82 prediction, optimization state or quality
result selects this analysis. It changes no running source or criterion.

The writing head receives FP32 contextual states but runs its condition
projection under BF16 autocast. Test whether distinct observed states become
identical at that interface, and quantify how much their pairwise differences
change. A lower-precision output projection was addressed earlier; an FP32
final layer does not by itself establish preservation of input distinctions.
This is a possible numerical limitation, not an assumed cause of corruption.

Use both completed V81 arms and every pre-block full-history condition on the
same 32 fixed training questions: 107 conditions per arm. Authenticate their
final checkpoints, completed evaluation/verification, source headers, actual
target images and state traces against the committed public result. Exclude
terminal stopping states from writing comparisons. No target label is a
model input in this audit.

On CPU, retain and compare:

1. actual saved FP32 conditions;
2. those same values rounded to BF16 and represented in FP32;
3. their FP32 trained condition projection;
4. an FP32 projection with only its input rounded;
5. FP32 accumulation with BF16-rounded input/weights/bias, rounded on output;
6. the actual local PyTorch CPU BF16 linear operation.

Report exact row-collision groups and whether they contain distinct target
pixel blocks. Retain every same-passage question pair at matching block
positions, including its target equality and distances before/after each
rounding stage. Report relative projection errors and centered numerical
spectra. More numerical rank can reflect quantization noise and must not be
called more semantic information. No collision is required for a lossy
projection; finding none cannot rule out subtler information loss.

CPU BF16 matrix accumulation is not an independent replay of the GPU kernel.
The explicit rounded-operand projection distinguishes operand/output rounding
from implementation-dependent reduction differences. No complete language
forward, sampler, optimizer, new answer or semantic-accuracy measurement is
performed. Even a collision of differing targets would identify an interface
ambiguity, not by itself prove the cause of all corrupted free answers.
Preserve all rows and provenance; make no new FP32 training change during the
ongoing matched V82 experiment.
