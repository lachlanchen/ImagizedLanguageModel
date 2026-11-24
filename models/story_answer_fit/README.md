# Writing from a pictured question and a page

These weights select the evidence for a frozen visual answer writer. Both fits used 10,084 real questions and complete human replies. The single-path selector trained on the strongest evidence location for each answer. The marginal selector trained on the probability summed over all available locations. Both choose one strip at inference and draw their own answer. The released 57,984-parameter heads use the shared optical reader, ranker and writer. The full inference pipeline has 1,135,306 parameters. [Experiment and limitations](../../references/story_answer_fit_result.md)

Run the first illustrated story on CPU from the repository root. The output directory must be new. The manifest supplies the question and section as PNGs with ordered 32 × 32 glyphs. The output includes answer and evidence PNGs, pixel hashes and a stop flag.

```bash
python -m scripts.write_story_answer --mode marginal \
  --input models/story_answer_fit/examples/story-1/input.json \
  --output /tmp/story-1-marginal
```

Use `story-2/input.json` to change the question on the same pictured section. Use `--mode single` for the matched model. You can also pass `--question` and one or more `--passage` strings to render your own inputs. Each model runs locally without a teacher. The included comparisons use quantized Qwen models solely during evaluation. [Shared model dependencies and parent preparation](../optical_competition/README.md)

To reproduce training, use the pinned FairytaleQA archive and parent optical corpus described in the linked preparation, a fresh experiment cache and one CUDA GPU. Each stage checks the prior data or parameter hashes. The first check evaluates the analytic derivative on real human questions. The held scorer evaluates answer probabilities given human continuations. The evaluator separately generates complete answers from images with no supplied continuation.

```bash
python -m scripts.prepare_story_answer_training
CUDA_VISIBLE_DEVICES=0 python -m scripts.check_story_answer_objective
python -m scripts.prepare_story_answer_fit
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_story_answer_fit
python -m scripts.audit_story_answer_split
CUDA_VISIBLE_DEVICES=0 python -m scripts.evaluate_story_answer_fit evaluate judge
CUDA_VISIBLE_DEVICES=0 python -m scripts.score_story_answer_probabilities
python -m scripts.figure_story_answer_fit
```
