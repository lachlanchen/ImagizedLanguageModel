# Writing an answer from a pictured conversation

Both checkpoints read a pictured passage and question and emit the answer one 32 × 32 glyph at a time. A follow-up question can carry the preceding human exchange, followed by a vertical bar and the current question. The input may contain up to 4,096 passage cells and 320 query cells. The generation cap is 64 decisions including stop. The optical reader and the shared visual language prior run locally without a text language model. The training source supplies the correct response and, in one condition, the annotated English evidence. Standard inference receives the full story and query. [Study and measurements](../../references/evidence_views_result.md) · [Protocol](../../references/evidence_views_protocol.md)

On 498 whole English stories the full-story checkpoint answered both questions on 63, the evidence-trained checkpoint on 51. Both scored zero exact on 116 questions with no accepted contiguous input phrasing. Use the full-story checkpoint to reproduce the stronger English result.

## Inference

Run from the repository root and choose a fresh output directory. The typed interface renders the two strings before inference. Open the resulting `answer.png`. The machine-readable receipt gives the output length and stop status.

```bash
python -m scripts.write_open_exam --arm whole --model-dir models/evidence_views \
  --passage 'The library was founded in 1891. The reading room opened in 1903.' \
  --question 'When did the reading room open?' --output /tmp/ilm-library
```

For bitmap inputs, supply `--passage-image`, `--question-image` and the occupied cell counts `--passage-cells` and `--question-cells`. Padding comes after the final occupied cell in a row-major grid. The documented weights expect aligned glyph cells. Use `--device cuda` on the project GPU. The `whole` checkpoint uses whole-story training throughout. Both arms use the released [visual prior and palette](../visual_discourse/).

The figure examples include the actual question excerpt and generated reply. To reproduce the complete pictured conversation from the local official CoQA development data, run the following command. The receipt compares the CPU bitmap with the published GPU output.

```bash
python -m scripts.replay_evidence_view_case --case 1 --output /tmp/ilm-evidence-case
```

## Reproduction

Install the project dependencies and the Noto Sans CJK, Noto Serif CJK and DejaVu Sans fonts. Prepare the earlier Chinese examination cache and the official CoQA development file and evaluator used by [the reference scorer](../../scripts/score_dialogue_ink_references.py). The preparation script fetches and verifies the additional public training and external validation files as needed. The CoQA corpus includes source-specific licenses. The C³ data is distributed for non-commercial research. Raw stories remain in the local corpus cache. [CoQA source and terms](https://stanfordnlp.github.io/coqa/) · [C³ source and terms](https://github.com/nlpdata/c3) · [DROP](https://huggingface.co/datasets/ucinlp/drop)

```bash
python -m scripts.prepare_open_exam
python -m scripts.prepare_branch_ink
python -m scripts.prepare_evidence_views
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_evidence_views whole --run-dir artifacts/views_replication/whole
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_evidence_views highlight --run-dir artifacts/views_replication/highlight
CUDA_VISIBLE_DEVICES=0 python -m scripts.evaluate_evidence_views --model-dir artifacts/views_replication --output artifacts/views_replication/scores
CUDA_VISIBLE_DEVICES=0 python -m scripts.measure_evidence_view_transfer --model-dir artifacts/views_replication --output artifacts/views_replication/scores
python -m scripts.audit_evidence_views --work artifacts/views_replication/scores --report artifacts/views_replication/scores
```

Resume the same training command after an interruption. It checks the corpus, initial weights, objective, example order and optimizer state before resuming. Keep the two runs sequential on a shared GPU.
