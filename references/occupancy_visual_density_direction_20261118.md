# A lossless spatial conditional density for visible writing

This candidate changes the physical prediction problem, not the language
acceptance criteria. The completed reader-coordinate and writer-modulation
trials do not repair the actor. Earlier joint physical likelihood produces
many readable forms but gives empty image modes and fails its own physical
conditions. Marginal pixels, conditional flow, mixtures, question assignment,
native-state regression and extra conditioning have all been tried; their
failures remain constraints, not untested alternatives.

The proposed writer models physical **occupancy** before refining strokes.
Its root chooses empty or nonempty. Each occupied region then chooses the
occupancy of its four quadrants, and only occupied quadrants are subdivided.
At the finest level, those decisions are the exact pixels. No character label,
Unicode scalar, font, stored glyph, VQ codebook or reference topology is needed
to generate an image. The decisions are discrete physical bits. This is not
a claim that the computation contains no discrete variables.

## Probability law and support

For a binary 32-by-32 image Y, let a be its nonempty indicator. For a=1,
the breadth-first occupancy traversal has N(Y) internal-region masks m_j in
{1,...,15}, with coordinates and depths d_j determined by preceding masks.
Let h be the state of a causal reader before the target enters its history.

\[
 p(Y\mid h)=
 \begin{cases}
 p(a=0\mid h),&Y=0,\\
 p(a=1\mid h)\prod_{j=1}^{N(Y)}
 q(m_j\mid h,m_{<j},d_{\le j}),&Y\ne0.
 \end{cases}
\]

Every nonempty parent has a nonzero child mask. Ancestral sampling therefore
cannot turn a declared nonempty region into an empty one. Root emptiness is
still learned; this does not remove whitespace errors by fiat. Each image
has one canonical traversal, every complete legal traversal gives one image,
each decision is normalized and the finite tree terminates within 342 events
including the root. Consequently the image law is normalized over all
2^1024 binary images. In particular, p(Y=0|h)=p(a=0|h).

This is an explicit factorization and a structural property, not a theorem
that a fitted model will draw a recognizable glyph or choose the right word.
It makes no restriction to a list of possible characters. The fixed raster
resolution remains a limitation for fine handwriting or complex historic forms.

The initial implementation uses a causal local transformer with four 192-wide
layers and cached attention. Its inputs are earlier four-bit physical masks,
prefix-determined coordinates/depths, and a contextual state. It generates its
own topology. Teacher-forced current/future masks cannot enter their own
predictors. Physical position and span length may condition a glyph; this
component alone does not model dependencies between different glyphs within
a supplied span. Complete language and stopping require the integrated actor.

## Control and discrimination from earlier work

Use an equally parameterized flat control with the same learned root and the
same physical 2-by-2 leaf masks. After a nonempty root, it emits all 256 masks
in raster order. If all preceding masks are empty, its final mask excludes
zero, so this branch also generates only nonempty images. This defines a
normalized nonempty flat law; it is not an algebraic renormalization of the old
flat writer. Both modes have identical tensor shapes and can share the exact
initial draw. The hierarchy, event order and sparse refinement are the changed
factorization, rather than access to a privileged blank label.

Earlier V50-P bounds a residual around a continuous plan; V54 models row-wise
pixel dependencies. Neither generates this lossless occupied-region tree.
The later joint writer predicts fixed eight-bit pieces and can finish with
all-zero pieces without a separate root choice. The completed white-target
objective adds margins; this candidate uses ordinary normalized likelihood,
with no margin, reward judge or source-mask selection. Its root does not prove
that empty-mode behavior was the sole cause of earlier failures. The matched
root control is necessary to distinguish a tree effect from explicit emptiness.

Compare total physical-image nats, actual generated pixels, whitespace and
event count, with unchanged complete sources and initialization. Shorter
encodings need not mean faster neural decoding, and dense trees can exceed
the old writer's 128 decisions. Do not infer efficiency before measuring it.
A fixed training-only complete-record preflight must precede any full fit.
Its outputs must improve before the unchanged complete-language and untouched
confirmation comparisons can establish useful visual language.

## Primary research and scope

[Wang et al., 2005](https://doi.org/10.1179/136821905X43927) use breadth-first
quadtrees for lossless binary image compression. That is prior representation
and compression work, not language learning. The
[Octree Transformer](https://arxiv.org/abs/2111.12480) generates hierarchical
shape sequences autoregressively. Its 3D-shape evidence does not certify glyph
semantics or consumer-GPU language training.

[QuadTok, October 2026](https://arxiv.org/html/2610.10497v1) combines quadtree
geometry and visual VQ codes for image generation under a topology supplied
before sampling. Here physical occupancy is lossless and its topology is
generated, without VQ. [OctLLM, October 2026](https://arxiv.org/html/2610.02388v1)
uses explicit octree occupancy and routed branches while retaining a pretrained
vision-language pathway; it still includes lexical language machinery and a
separate geometry completion stage. Neither work establishes independent
image-language understanding, this implementation's quality or 1,000-fold gains.

The mathematical and code checks qualify only this physical density as a
candidate component. They do not qualify a new ILM, pretrained knowledge,
generalization, historical understanding, complete stopping or efficiency.
The full engineering goal stays active.
