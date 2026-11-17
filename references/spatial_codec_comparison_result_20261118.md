# Spatial 64-bit plans improve held-form F1 but remain unqualified

The two spatial codecs complete all 6,000 updates and fail six of ten fixed
conditions. They retain the equal-form source order and all 1,766,209 shared
convolutional initial parameters, changing the two global bottleneck maps to
pointwise spatial maps. Each new codec has 1,767,365 parameters; each saved
global control has 2,030,465. The effect concerns this architecture including
its parameter difference, not uniquely isolated locality.

| Physical measurement | Global continuous | Spatial continuous | Global binary | Spatial binary |
|---|---:|---:|---:|---:|
| Training pooled F1 | .997578104 | .994859878 | .943666840 | .922577200 |
| Validation pooled F1 | .923484240 | .929688379 | .820592454 | .847344074 |
| Validation macro-form F1 | .903684546 | .910131441 | .801056845 | .827283493 |
| Validation exact cells | 424 | 480 | 0 | 0 |
| Validation macro exact rate | .002100840 | .004201681 | 0 | 0 |

All columns use the same 4,907 physical-validation cells and 952 forms.
Spatial binary F1 gains .026751621 over its saved global counterpart, passing
the prospective .02 improvement condition. It still fails every absolute
fidelity requirement, macro exactness, continuous retention and the common
continuous ceiling. Both new modes have zero source false blanks and false ink.
The binary source mean of eight plan events plus actual correction-tree events
is 48.657046, passing the unchanged complete-source event condition against
97.302489. Count savings and a small F1 gain do not qualify a writer.

The positive validation effect coexists with worse training pooled F1. Keep
that tradeoff and zero binary exactness. Spatial structure helps this finite
comparison but is not sufficient. The result does not prove a universal
64-bit limit, identify optimization uniquely or justify more unchanged epochs.
No conditional language fit is admitted.

## Execution and complete record

Both weighted complete-source preflights have finite nonzero bottleneck/drawing
gradients, actual updates and exact resets. Both update-three optimizer replays
are exact. Actual sampled image order equals the saved global equal-form order;
the old controls are not re-executed. Source arrays, prior models, frozen
language readings and all criteria remain unchanged.

The [independent CPU audit](../scripts/export_spatial_codec_comparison.py)
recounts every one of the 275,792 persisted initial/final reconstructions,
exact XOR recoveries and correction event counts, every one of 12,000 retained
updates and both final checkpoint hashes. It verifies all shared initial CNN
values and source-order equality without another neural fit or decoding pass.
The [complete result](evidence/spatial_codec_comparison_result_20261118.json)
and fixed [validation](evidence/spatial_codec_comparison_20261118/held_fixed_reconstructions.png)
and [training](evidence/spatial_codec_comparison_20261118/train_fixed_reconstructions.png)
plates preserve the original pixels, cases, controls and failed conditions.
These are supplied-image reconstructions, not answers or semantic scores.

The monitored neural run takes 261.088 seconds, .524 GiB peak allocation,
zero pressure wait and 47.892 GiB minimum sampled available host memory.
The 48-GiB entry reserve and 24-GiB runtime reserve remain distinct. The
supervisor records 270.271 seconds. CPU recount takes 38.037 seconds at
1.716 GiB peak RSS. All owned model processes terminate; foreign GPU1 remains
unchanged. No contextual foundation is loaded, and no whole-model efficiency
or language-learning claim follows from this physical-component cost.

This spatial 64-bit recipe is terminal. Preserve its limited architectural
improvement together with the failed absolute quality. A later capacity test
must retain spatial geometry, data allocation, loss and budget, declare extra
plan events and keep every original quality/resource condition. Do not install
this failed codec, change thresholds or select an intermediate checkpoint.
