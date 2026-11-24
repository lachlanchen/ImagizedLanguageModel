# Classical Chinese image continuation

Give the writer a horizontal strip or row-major grid of 32 × 32 black-on-white glyph cells. It returns a new PNG. The default writer uses a visual transformer, 5,501 training bitmaps, a pointer to the visible cells and a suffix-count table. The inference path accepts an unencoded bitmap and can copy it from the prompt. It has no OCR or text-model dependency. It uses up to 64 recent cells and a user-specified output length. Free writing frequently repeats short phrases.

Run from the repository root in the project PyTorch environment. The first command uses a genuine held prefix and requires no training corpus.

```bash
python -m scripts.write_book_prefix \
  --input references/evidence/book_prefix_v1/KR1h0004-prefix.png \
  --steps 16 --output continuation.png

scripts/with_ilm_gpu python -m scripts.write_book_prefix --device cuda \
  --text '子曰學而時習之不亦說乎' --output continuation.png
```

Use `--cells N` to exclude padding at the end of a rectangular grid. Typed input uses the Noto CJK font expected by the project renderer. For a scan or a handwritten sample, align cells and binarize to this interface. Evaluation covers the canonical printed font.

## Workstation GPU command

The project runtime on this workstation is `/home/lachlan/miniconda3/bin/python`. Use its full path when running from the older `incoder` shell. The launcher finds the user-installed driver compatibility libraries in batch shells as well as login shells. The November 24 training run used the loaded 595.84 kernel driver with matching CUDA and NVML libraries. An unwrapped system NVML client installed for 595.99 reported a version mismatch. The older environment's PyTorch can execute CUDA operations, but its torchvision is incompatible with PyTorch and it lacks safetensors.

```bash
scripts/with_ilm_gpu /home/lachlan/miniconda3/bin/python -m scripts.write_book_prefix \
  --device cuda --input references/evidence/book_prefix_v1/KR1h0004-prefix.png \
  --steps 16 --output continuation.png
```

## Reproduction

Install the three CoNLL-U files and the accompanying LICENSE.txt from the treebank's r2.16 release in `data/raw/ud_classical_kyoto_r216/`. The release reassembles unpunctuated text by volume, removes title records and reserves entire works. Rendered images, count training features and training caches are constructed locally. The upstream distribution credits the Institute for Research in Humanities, Kyoto University and its contributors. Its bundled LICENSE.txt specifies CC BY-SA 4.0. The derived corpus examples and suffix-count data retain that attribution and licence. [Versioned data](https://github.com/UniversalDependencies/UD_Classical_Chinese-Kyoto/tree/r2.16) · [Licence](https://creativecommons.org/licenses/by-sa/4.0/)

```bash
python -m scripts.prepare_book_prefix
python -m scripts.fit_book_counts
scripts/with_ilm_gpu python -m scripts.train_book_prefix
python -m scripts.package_book_prefix
scripts/with_ilm_gpu python -m scripts.evaluate_book_prefix
python -m scripts.inspect_book_prefix
```

A completed checkpoint with a matching training receipt is reused. For an independent fit, use a fresh output directory with `ILM_BOOK_PREFIX_RUN=/absolute/path/to/run` for all six commands. That directory holds its own `artifacts`, `evidence` and `model` subdirectories. For inference from those new weights specify `--bundle /absolute/path/to/run/model`. Training and evaluation use GPU 0 and the project lock. The released PNG route also works on CPU. Exact outputs from CPU and GPU agreed on the first reserved passage in each book. [Results and limitations](../../references/book_prefix_result.md)
