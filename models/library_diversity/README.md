# Writing from bilingual book images

The broad and repetition-control models read aligned 32 × 32 glyph cells and write a bitmap sequence with a learned stop. Each has 1,688,480 neural parameters and shares the existing optical repertoire. Qwen was used only in evaluation. The optional nearest-neighbor writer also uses a 613 MB train-only memory.

To translate the pictured Proust question into English, run from the repository root with the project Python environment and a fresh output directory.

~~~bash
python -m scripts.write_ordered_translation --model library-broad \
  --image publication/ilm-nature/figures/library_diversity/request-740-input.png \
  --cells 8 --to en --output /tmp/ilm-proust-question
~~~

The writer produced **“But who is he?”** including its quotation marks. Inspect `answer.png` and the stop receipt. Use `--model library-repeat` for the matched control. `--text` renders a typed input to the same grid and `--to zh` selects Chinese output. The models accept 1–96 occupied source cells, including spaces. Generation is limited to the existing output bitmap inventory.

For the fixed positive source-contrast diagnostic add `--decode background-half`. Development selected ordinary decoding. For associative recall, use `--decode associative`. Both experiments and the undistorted PNGs are documented in the [full result](../../references/visual_library_writing_result.md). The ordinary and memory writers each earned three source-only credits on the memory panel of 96 requests.

The training library is prepared locally from the 21 named ZhJpBook works. Preparation and evaluation also use the existing project corpus and rehearsal artifacts. Run the following stages in order on the project workstation to reproduce the fits and rebuild the optional memory.

~~~bash
python -m scripts.prepare_visual_library
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_visual_library
CUDA_VISIBLE_DEVICES=0 python -m scripts.evaluate_visual_library
CUDA_VISIBLE_DEVICES=0 python -m scripts.calibrate_library_contrast
CUDA_VISIBLE_DEVICES=0 python -m scripts.evaluate_library_contrast
CUDA_VISIBLE_DEVICES=0 python -m scripts.build_library_memory
CUDA_VISIBLE_DEVICES=0 python -m scripts.evaluate_library_memory
~~~

Memory archives and complete answer panels are stored locally under `artifacts/library_memory_v1/` and `artifacts/library_diversity_v1/`. The published receipts identify the exact local inputs and weights.
