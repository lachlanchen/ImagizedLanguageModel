# Scalar-plan comparison is interrupted and unqualified

The 128-value continuous control completes all 6,000 updates, but the hard-code
arm stops during its complete-source preflight, before any optimizer update.
Its final quality is unmeasured. This is not a completed paired comparison or
evidence that the quantized network cannot learn.

The control reaches validation pooled F1 .978102816, macro-form F1 .965529914
and 34/952 exact forms, below the unchanged .98 continuous ceiling. It is worse
than the saved 256-value continuous control's .996423850. Trading sixteen one-bit
features for eight four-level features changes feature dimension as well as
precision; the result does not isolate a universal scalar-quantization effect.

All 8,825,344 initial scalar values, across all 68,948 source images, round to
zero. The preflight requires a nonzero gradient of the expansion weight alone.
For an affine expansion `Wq+b`, `dL/dW = (dL/da)q^T`; it is zero when every q is
zero. This guard is incompatible with this initialization. It does not establish
zero encoder or total drawing-network gradient. Those separate norms were not
retained, so neither should be invented. The independent saved-array diagnosis
uses the common initial continuous values and makes no new neural call.

The interrupted checkpoint retains a stale `updates=6000` counter from the
preceding continuous arm. Its actual binary weights are byte-identical to
initialization, and no binary update ledger, initial/final generation or final
checkpoint exists. Preserve the original failure file and this explicit
correction; there were **zero retained hard-code updates**. Future multi-arm
failure handlers must initialize counters before the preflight.

The [independent CPU audit](../scripts/audit_scalar_codec_interruption.py)
checks all 137,896 saved control reconstructions and correction-event arrays,
all 6,000 retained control updates, source order, common CNN initialization,
checkpoint identity and actual terminal ownership. It preserves the
[complete evidence](evidence/scalar_codec_interrupted_result_20261118.json).
The hard-code failure is not silently converted into a zero score. All original
sources, rules, completed controls and reserved language data remain unchanged.

No automatic or unchanged recovery is selected: the completed continuous control
already misses its fixed ceiling. Fixing the admission guard would not remove
that measured limitation. No conditional-language fit, deployment or whole-model
efficiency follows. The next proposal must address an observed remaining failure
and retain all original language and physical criteria.
