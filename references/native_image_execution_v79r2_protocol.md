# Execution-checker scalar-hash repair

The first V79 execution stopped while hashing the scalar input-scale buffer:
viewing a zero-dimensional FP32 tensor as bytes is invalid in this PyTorch
version. It had not reached checkpoint, autonomous-feedback, final immutability
or CPU-verification completion. Its partial directory and raw error remain
unchanged; it is not a completed pass or a language result.

Rerun the exact architecture, fixture, seed, numerical settings, no-update
boundary and all acceptance thresholds from native_image_execution_v79_protocol.md.
Only flatten tensors before byte-view hashing in the separate revised checker.
Use a fresh native_image_execution_v79r2_20261106 directory, pin the original
checker/protocol alongside this repair, and retain all originally required
checks. No model objective, weights, target, threshold or language claim changes.
