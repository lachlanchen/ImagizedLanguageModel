# Serpentine Visual Lattice

Date: 2026-08-13

Status: long-context design note; not part of the frozen V25 evidence contract

## Purpose

The native visual language object is an ordered stack of clean character
images,

\[
X\in[0,1]^{B\times N\times1\times32\times32}.
\]

Each slice remains readable and may contain print, handwriting, a historical
form, an unknown sign, or damage. For long books, direct global attention over
`N` is too expensive. The stream can instead be folded into a continuous 2D
field without assigning a character ID.

## Exact geometry

A `256 x 256` lattice has 65,536 positions. The sequence coordinate `t` maps
to a serpentine page coordinate

\[
r=\lfloor t/256\rfloor,
\qquad
c=\begin{cases}
t\bmod256,&r\text{ even},\\
255-(t\bmod256),&r\text{ odd}.
\end{cases}
\]

Every consecutive pair is therefore adjacent on the page, including row
turns. The mapping has an exact inverse and carries an explicit validity mask
for a partially filled final row.

Three related representations must not be confused:

| Representation | Shape for 65,536 cells | Role |
|---|---:|---|
| clean visual stream | `65536 x 1 x 32 x 32` | canonical input/output evidence |
| native flat page | `1 x 8192 x 8192` | exact inspectable fold |
| overview page | `1 x 2048 x 2048` | lossy 8-pixel peripheral view |
| learned retinal lattice | `C x 256 x 256` | efficient long-context state |

The raw native stream contains 67,108,864 scalar pixels, about 128 MiB in
BF16 before activations. It should be streamed through the retina in chunks,
not materialized repeatedly inside every layer. The retina first maps each
clean `32 x 32` slice to a continuous state `z_t in R^C`; those states are then
folded into `Z in R^(C x 256 x 256)`. This preserves more visual information
than making an `8 x 8` thumbnail the sole record of a complex Han form.

## Input and output

Input has two equivalent views:

1. canonical 3D visual time: `N x 1 x 32 x 32`;
2. folded 2D context: an exact page or learned `C x R x C_col` lattice.

Output follows the same geometry. The cell-causal writer first draws one flat
2D image `y_t in [0,1]^(1 x 32 x 32)`. Generated cells are stacked into
`Y in [0,1]^(K x 1 x 32 x 32)`, reread in order, and folded into a page for
display. A later block writer may draw a fixed grid of several cells at once,
unfold it into ordered slices, and reread every slice. The block page must not
silently replace the high-resolution cell stream.

## Causal long-context rule

A normal 2D convolution over a filled training page can see later cells and
would invalidate next-cell prediction. Long-context processing must therefore
obey one of these equivalent restrictions:

- masked convolutions whose mask follows the serpentine order;
- a causal scan over the folded coordinates with 2D local state; or
- a hierarchy in which global memory contains only completed earlier blocks,
  while the current block is read causally at full resolution.

The proposed single-4090 route is the third option. Keep a short full-resolution
foveal stream, cache completed retinal blocks on the 2D lattice, and build a
multiscale prefix memory from strictly completed blocks. This makes work grow
approximately linearly with character count rather than quadratically. It also
matches reading: recent writing is sharp, older context is peripheral but not
discarded.

V25 remains the 64-cell causal proof. This lattice is the next long-context
extension after V25 demonstrates that natural Chinese image cells contain
measurable predictive language signal.
