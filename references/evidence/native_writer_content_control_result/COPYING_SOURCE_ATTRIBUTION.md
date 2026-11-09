# Synthetic copying cohort

Input and reference strings originate from `copy_cases()` in
`ilm/image_edit/paired_data.py`, using deterministic seed 20268320. Chinese
strings contain 4–12 sampled characters; English strings contain 2–4 sampled
words. A shared uniqueness check separates the training and validation
strings. The copying cases contain no encyclopedia or book passages.

The source renderer produces full input pages and separate answer strips.
`scripts/prepare_visual_flow_pages.py` places each original answer strip on
a white 1024-square canvas without rescaling its writing. The registered
evaluation uses the first 32 training copying cases and all 32 validation
copying cases, with equal Chinese/English counts in each split.

This folder preserves the original rendered inputs, padded references and
actual neural-generated answers byte for byte. `report.json` binds them to
the complete fixed cohort, image-only inference records, actual decoder
tensors and separate offline OCR calibration. This copying cohort is a
writing-fidelity control, not a natural-language comprehension benchmark.
