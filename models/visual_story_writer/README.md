# A visual writer for pictured questions and source pages

Use `balanced` to request a historical specimen from a labelled page. Use `page` to reproduce the general reading experiment. These checkpoints share an optical reader and an image-copying writer with a learned bitmap palette. Both run locally on CPU or on one CUDA GPU. The deployed student needs no MiniMind, Qwen, network or SQLite connection for a pictured request. The reference library is used to construct new catalogue pages.

From the repository root, write into a fresh directory.

```bash
python -m scripts.write_visual_story --mode balanced \
  --input models/visual_story_writer/examples/yan/oracle-zh.json \
  --output /tmp/ilm-yan-answer

python -m scripts.write_visual_story --mode page \
  --input models/visual_story_writer/examples/c30/input.json \
  --output /tmp/ilm-reading-answer

python -m scripts.write_visual_story --mode page --question "Where did Todd visit?" \
  --passage "Todd visited the city." --output /tmp/ilm-my-page
```

The CLI writes `answer.png`, the selected evidence in `fact.png`, and the stop flag, cell count and pixel checksum in `receipt.json`. The typed-input path rasterizes modern script before the visual model sees it. A pictured manifest provides one question PNG and one or more region PNGs, with cell counts. Use the examples' 32×32 aligned cell layout. For an unencoded historical shape, supply its pixels in a region. Raw scan layout recognition requires separate preprocessing.

To reproduce the local training and comparisons with the retained source data and earlier model releases:

```bash
python -m scripts.build_origin_library
python -m scripts.prepare_visual_story_writer
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_visual_story_writer
CUDA_VISIBLE_DEVICES=0 python -m scripts.evaluate_visual_story_writer forms evaluate judge
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_balanced_visual_story
CUDA_VISIBLE_DEVICES=0 python -m scripts.evaluate_visual_story_writer judge-balanced
python -m scripts.audit_form_pairs
```

The preparer checks MCTest.zip and c3-m-dev.json under `data/raw/narrative_writer/` against the reservation, and uses the retained disjoint passage indices and earlier writer materials. The text baseline checks the pinned sibling MiniMind checkout and official full-SFT checkpoint. The semantic evaluation uses the local, hash-checked Qwen judge. Single-example inference needs only the released student assets and the repository's Python environment. [Data and checksums](../../references/chinese_origin_library.md) · [Primary MCTest files](https://mattr1.github.io/mctest/data.html) · [Primary C³ files](https://github.com/nlpdata/c3/tree/1f41682ebb684d63afe7c583c8f38e137fc8a826) · [Full outcome and limitations](../../references/visual_story_writer_result.md)
