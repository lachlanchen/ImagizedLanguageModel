# Equal-bit scalar precision for a generated visual plan

The [completed spatial-capacity comparison](spatial_capacity_comparison_result_20261118.md)
narrows the current physical failure. Its common convolutional drawing network
reaches .996424 validation F1 with continuous features, but only .943690 with
one bit per feature. It is now reasonable to test the precision allocation of
the quantized plan. More epochs of either completed recipe are not justified.

The selected change retains a 4-by-4 field and 256 total bits, allocating eight
four-level features per site instead of sixteen two-level features. This is
a tradeoff between feature dimension and scalar precision at fixed bit budget;
it does not isolate precision independently of dimension. Each codec has
1,768,393 parameters. All 1,766,209 shared convolutional initial parameters,
source order, exposure, reconstruction loss and optimizer schedule remain fixed.

Use finite scalar quantization from [Mentzer et al., ICLR 2024](https://arxiv.org/abs/2309.15505),
with the exact even-level bounding expression in Appendix A.1. With
`a = 3(1-.001)/2`, define

    u = a tanh(z + tan(.5/a)) - .5
    q = round(u) / 2

Actual decoder features belong to {-1, -.5, 0, .5}. The continuous paired control
receives u/2 with the same parameters and bounding function. Backpropagation
uses straight-through rounding, a biased surrogate rather than the derivative
of a discontinuous quantizer. No commitment penalty, learned codebook, entropy
penalty, noise curriculum, renderer, OCR or lexical classifier is added.

Serialize each integer index `2q+2` in two least-significant-first bits, in
site-raster/channel order. All 256-bit patterns decode to valid scalar fields;
the serial plan still takes thirty-two eight-bit decisions. The generator
therefore retains the earlier normalized joint law

    P(Y|H) = sum_B P(B|H) P_R(Y XOR D(B)|H,B).

For every B, XOR is a bijection over all binary images. Full support comes from
the physical residual, not a claim that 256 bits can encode every 1,024-bit
image exactly. The deterministic encoded training branch gives an upper bound
on marginal image NLL. At inference, the prior must actually generate B and the
residual network must generate R; neither receives the future target image.
There is no learned residual fit or successful conditional-language result yet.

This design adopts an established image representation method; it is not a new
physics theorem or proof of cheaper language learning. [PIXEL](https://arxiv.org/abs/2207.06991)
provides prior evidence for language representations from rendered images, and
[PIXAR](https://aclanthology.org/2024.findings-acl.874/) studies autoregressive
pixel-language generation. Neither establishes this model's quality or the
requested consumer-GPU parity. The proposed contribution still requires actual
independent image answers, causal pixel feedback and coherent book continuation.

First require structure, bit serialization, actual quantized forward values,
real gradient/update/reset, common CNN identity and target-encoder exclusion
checks. Then prospectively register one complete physical comparison against
the saved spatial-256 controls, retaining every original fidelity and resource
rule. No passed component may replace the complete-language acceptance criteria.

If the codec passes all conditions, separately bind and qualify the conditional
plan prior and physical correction on authentic complete training records.
If it fails, preserve the final models and all outputs; inspect the observed
remaining error before selecting another bounded intervention. No unchanged
extension, snapshot selection or language-fit admission follows from failure.
