# Subword conversation baseline

This 817,506-parameter encoder–decoder Transformer reads and generates text using a 4,096-symbol byte-level BPE vocabulary. It trained from scratch on the same sequence of 256,000 CoQA examples as the visual writers. It has two encoder blocks, two decoder blocks, four attention heads and a tied output embedding. A training-only evidence loss supervises the source interval.

```bash
python -m scripts.answer_dialogue_subword --passage 'Kim bought a book.' --question 'What did Kim buy?'
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_dialogue_subword \
  --model artifacts/subword-refit --evaluation artifacts/subword-refit-eval
```

Use the registered CoQA split and produce the direct visual decisions first for the matched control evaluation. The CLI accepts a passage, a question and optional text history. CPU decoding uses full precision. Benchmark decoding used GPU bfloat16 and required an explicit end marker. The original model weights, tokenizer, training trace, energy measurements and hashes accompany this file. [Visual writer and data](../dialogue_ink/README.md) · [Full comparison](../../references/dialogue_efficiency_result.md)
