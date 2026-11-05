# V69 independent mean-control verification correction

The original capture, fitting implementation, calibration, raw outputs and
first verifier remain unchanged at commit `5a5d407`. Capture completes all
896 prompt forwards and exactly replays all 192 original first-stop decisions
and output blocks. CPU fitting completes before its verifier reports a
training-mean discrepancy.

The discrepancy is in the verifier's independent NumPy reference: it sums
512 FP32 vectors in FP32 before converting to FP64 for normalization. The
maximum difference from the retained mean is 3.1365884e-6, above the original
2e-6 tolerance. The producer's declared Torch arithmetic reconstructs the
stored mean exactly. Independent FP64 NumPy accumulation reduces the maximum
difference to 3.4977544e-7.

`scripts/verify_conditional_readout_v69_r2.py` therefore accumulates this
independent mean reference in FP64. The numerical tolerance remains 2e-6.
All other calibration, normal-equation, source-pixel, encoder, physical-output,
score and bootstrap checks remain unchanged. The correction changes no
prediction or reported experimental score. The original failed verifier is
retained because its hash is part of the frozen producer's provenance.
