# Directional reading of pictured questions

The released model accepts a pictured question and pictured passage regions. It selects a source strip, generates a glyph stream and saves both images. The signed fit improves English evidence selection. The complete system still produces inaccurate or repetitive answers. The image interface expects ordered 32 × 32 glyph cells. Typed input uses the repository renderer. [Evaluation](../../references/directional_reading_result.md)

From the repository root, reproduce an English response with the checked CPU path. Choose a new output directory for each run.

~~~bash
python -m scripts.write_directional_reading --mode signed --device cpu \
  --input models/directional_reading/examples/mlqa-en/input.json \
  --output /tmp/ilm-english-reading
python -m scripts.write_directional_reading --mode signed --device cpu \
  --input models/directional_reading/examples/mlqa-zh/input.json \
  --output /tmp/ilm-chinese-reading
~~~

The output directory contains `answer.png`, `fact.png` and a receipt with the stop decision, selected region, cell count and physical hashes. The main English and Chinese examples include all three model outputs. The additional `mlqa-en-credited` and `mlqa-zh-credited` manifests reproduce the stopped English sentence “Dickinson wrote the poem.” and the Chinese question’s answer “26.50%”. The four 言 directories contain historical test outputs. The shared reader, writer, visual inventory and earlier whole-span ranker must be present. The CLI verifies the parent weight digests. No teacher process is required. [Shared reader and writer](../fact_channel/README.md) · [Whole-span ranker](../fact_span_ranking/README.md)

For training, prepare the pinned parent corpus and previous reservations, then place the official MLQA v1 archive at `data/raw/mlqa/MLQA_V1.zip`. The preparation script verifies both MLQA and DROP source checksums. Use a separate checkout and fresh experiment output paths when reproducing the two fits. Existing matching fits are reused. The separate model-judged benchmark requires the pinned local Qwen models. [Source split](../../scripts/prepare_directional_reading.py) · [Training protocol](../../references/directional_reading_protocol.md)

~~~bash
python -m scripts.prepare_directional_reading
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_directional_reading unsigned signed
CUDA_VISIBLE_DEVICES=0 python -m scripts.evaluate_directional_reading
CUDA_VISIBLE_DEVICES=0 python -m scripts.probe_directional_reading
CUDA_VISIBLE_DEVICES=0 python -m scripts.audit_directional_candidates
CUDA_VISIBLE_DEVICES=0 python -m scripts.diagnose_directional_answers
CUDA_VISIBLE_DEVICES=0 python -m scripts.review_directional_reading
python -m scripts.audit_directional_split
python -m scripts.figure_directional_reading
~~~
