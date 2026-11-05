# Continuous writing: perturbation diagnostic

Registered 5 November 2026 before execution. This is a CPU-only diagnostic of
the frozen, qualified V34 codec, motivated by the encoder–decoder robustness
limitation in [Sigma, Appendix B](https://arxiv.org/html/2610.02665v1#S0.SSx2).
It changes no actor, training source, checkpoint, development gate or final
evaluation. It does not test language, semantic similarity or correct answers.

Use exactly the 64 physical training blocks in the completed V65 manifest,
SHA-256 `2fc36c1da4344351dd0f97fa4f928fd99f9b5ca7e16b01c50ad439ecbbdbc50c`.
Re-read their original, hash-checked training-cache files and verify their
saved reference PNGs. No development or sealed images are opened.

Use the original V34 EMA, pinned by `load_qualified_writing_codec`, through
the committed `CodecBlockWriter`. Native encoding is white-positive, with
two continuous 768-dimensional vectors per physical 32-by-64 image block.
There are no character labels, vocabulary entries or retrieval candidates.

For four fixed independent directions per block (CPU generator seed
2026110507), remove the mean and the component along the mean-centered code,
then rescale each direction to unit coordinate RMS. These are tangent
directions to the centered constant-norm surface at the source code. Evaluate
`LayerNorm(z + sigma * direction)` for sigma in
`[0, 0.01, 0.03, 0.1, 0.3, 1.0]`, with epsilon `1e-5`.
LayerNorm's epsilon makes the surface description approximate; actual latent
MSE is measured rather than inferred from sigma. No perturbation is selected
by its output quality. Source-code and re-normalized-code baselines are both
retained. Sigmoid outputs use the existing zero threshold in signed pixels.

Retain all continuous perturbed codes, raw decoder scores and reference
images. Report foreground F1, exact blocks, latent MSE, continuous pixel MSE
and the proportion of thresholded pixels that change from the source-code
reconstruction. Every perturbation is compared with its own original block.
At sigma 0.01, report the range of directional pixel-MSE / latent-MSE ratios
within each block. This is a finite-difference sensitivity diagnostic, not a
Jacobian eigenspectrum, intrinsic dimension or posterior susceptibility.

There is no acceptance threshold or hyperparameter selection. A post-run
result must preserve all levels and cases, disclose the use of training
images, and keep reconstruction separate from autonomous writing. The
registered complete-actor preflight remains the next execution step; this
diagnostic cannot qualify that actor or justify a new optimizer by itself.
