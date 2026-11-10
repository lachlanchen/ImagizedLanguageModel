# Dense motor-coordinate flow is not a successful glyph prior

The168,366,080-parameter conditional flow fails both prospective rendering
requirements. Clean conditions give mean selected foreground F1 **0.906284**,
with only5.823%of forms reaching0.95and no pixel-exact glyph. The fixed45%
mixtures give **0.715449**, with no form reaching0.95or pixel exactness.
The original neural decoder and the verified language checkpoint are unchanged;
no full language inference uses this failed motor.

The reference-best of four samples, computed only as an oracle diagnostic,
gives0.910727 on clean conditions and0.735920 on mixtures. Thus the present
failure is not explained by selection alone: the sampled candidates themselves
usually lack the required physical quality. Re-encoding real pixels for
selection does not make those pixels correct.

Training8,000updates takes129.67s at3.313GiB peak allocation; qualification
takes85.10s. Final velocity MSE is0.860810. Neither falling loss nor the removal
of a narrow output-rank ceiling establishes learned glyph modes. All selected
and candidate images are retained, and both fixed64-triplet boards were
actually viewed. Clean outputs lose strokes; mixtures remain composites with
wrong parts. Individual human grades for every glyph are not claimed.

This result rejects the specified dense flow/schedule, not all flow matching.
The mathematical objective describes a conditional joint distribution, but the
finite model has not learned the sharp endpoint forms. The subsequent
[spatial motor experiment](spatial_semantic_motor_protocol.md) reuses an
existing32x32image-flow U-Net and the preserved decoder's continuous ink plan,
introducing explicit image structure instead of increasing dense semantic
capacity. That follow-up is a new experiment, with no result claimed here.

[Flow protocol and conditional-path equations](semantic_motor_flow_protocol.md) ·
[All measurements and raw sampled pixels](evidence/semantic_motor_flow_v1/)
