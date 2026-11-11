# Text reference for the compact visual model

Use the local `qwen3:1.7b` archive that supplies the compact model's inherited
contextual weights. The text model uses its original tokenizer, vocabulary
head and chat template. Match all 44 normal visual inputs by recovering the
original prompt strings and verifying that each re-renders to the exact
registered pixels. Book prompts end at the same tile as their image inputs.
Generation reads the prepared prompts. The scoring stage opens the references.

Run Qwen3 in non-thinking chat mode with greedy generation and a 128-token
limit. Use one fixed instruction for each task. For QA, ask for the shortest
complete extract from the passage. For book prefixes, ask for a continuation
of roughly 100 characters in the same style. These system instructions and
the text model's chat formatting are part of the baseline recipe. The visual
model receives the registered input tiles. See [Qwen3](https://qwenlm.github.io/blog/qwen3/)
and the [Ollama chat API](https://docs.ollama.com/api/chat).

Score an exact complete answer when the raw response ends naturally, fits
within 128 characters and matches the canonical reference after whitespace
removal. Count both correct answers in each of the sixteen QA pairs. For
books, record exact source matches and stopping. Assess other coherent
continuations in the later visual-model review. Record cap hits and retain
all original outputs.

The initial measurement runs the quantized text model on two CPU cores at
low priority, with zero GPU model residency. It supplies a task quality
reference. A GPU speed or efficiency comparison requires measurements of
both routes on the same RTX 4090. Comparison with Qwen3 8B also remains a
separate experiment.

## Measured reference

The Q4_K_M model produced all 44 responses. Eight of 32 QA responses matched
the complete canonical span, and both questions matched in two of sixteen
contexts. Of the 24 misses, 14 responses were strict substrings of the
canonical span and three contained the canonical span. The remaining seven
had neither relation. These counts use whitespace-normalized strings.
Alternate answers still require a contextual correctness judgment.

None of the twelve book outputs exactly reproduced the original continuation.
Eleven ended naturally and one reached the token cap. Coherence remains
unrated. The 44 CPU requests took 454.36 seconds in total, including model
loading and reuse of prompt prefixes. The service reported zero GPU model
residency in every case. See the [run evidence](evidence/compact_visual_span_qwen_reference_20261111.json).
