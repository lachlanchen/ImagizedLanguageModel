# Answer a pictured question about a person's role

The optional participant scorer learns where the requested person's name begins and ends on a page. It receives the role from the pictured question reader. Both neural backbones and the glyph writer run locally. In the repository environment, run

```sh
PYTHONPATH=. python scripts/answer_visual_fact.py \
  --relation-head models/pictured_relation/pictured-relation.npz \
  --participant-head models/participant_margin/participant.safetensors \
  --writer artifacts/role_binding_v1/writer --word-order object \
  --passage-text '新曲《松窗》由周至作词，苏恬作曲。' \
  --question-text '谁是《松窗》的作曲人？' --output answer.png
```

For image input, pass `--passage passage.png --question question.png` in place of the two text arguments. Each input is a strip of 32 × 32 glyph cells, printed in black on white. Omitting the participant checkpoint selects the earlier reader. See the [training and comparison](../../references/participant_margin_result.md) for the measured performance on different passages.

The likelihood-only comparison is also released. Substitute `models/participant_margin/likelihood.safetensors` for the participant checkpoint in the command above. It achieved 79/128 complete new passage exchanges and answered 28/48 human questions in both requested orders. The portable update `participant-scorers.zip` includes both weights. Extract it over the full pictured fact model and question update. Use `--participant-head model/likelihood.safetensors` or `--participant-head model/participant.safetensors` from that directory.
