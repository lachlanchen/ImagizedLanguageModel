# Complete image-only editor baseline

All 20 normal prompts and 20 matched blank controls are retained. There is no local training and no qualified ILM result. OCR exact match is 0/12 copy requests and 0/8 questions; reference-page OCR calibration is 18/20. Read the [result and limitations](../../image_edit_baseline_result.md).

The four galleries retain every complete raw output in original case order, with reference, normal and blank columns:

- [Cases 00–04](all_cases_00_04.png).
- [Cases 05–09](all_cases_05_09.png).
- [Cases 10–14](all_cases_10_14.png).
- [Cases 15–19](all_cases_15_19.png).

`raw/` contains all prompts, references and answers; `codec/` contains every prepared page's reconstruction. [The export receipt](result.json) authenticates the original PNG bytes and checks all generated/codec images against retained raw BF16 decoder RGB. The raw tensors and weights remain in ignored local artifacts. This does not replay the neural model independently. Five accompanying reports retain the source identity, costs and separate offline OCR observations.
