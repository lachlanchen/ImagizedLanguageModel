# A local reader and writer for pictured Chinese facts

Give this model an image of a short Chinese passage and an image of a question about it. It can identify the people, works, institutions and places involved in twelve kinds of fact. It writes an answer as a strip of glyph images. The released bundle includes the visual reader, a factual prediction head, an induced sentence writer, a separate autoregressive image writer, and original examples.

Extract the model archive. Install Python with PyTorch, Transformers, safetensors, Pillow, NumPy and fontTools. From the extracted directory run

```sh
python -m scripts.answer_visual_fact --reader reader --model model \
  --passage examples/film_0_passage.png --question examples/film_0_question.png --output answer.png
```

For the credit in a different word order, add `--writer writer --word-order object`. This invokes a separate generative model that writes one physical glyph at a time. The default induced writer has one complete statement pattern per relation. To return only the selected participant, use `--direct`. The experimental `--writer writer --word-order natural` permits freer phrasing and can add unsupported information. Both neural backbones run locally. Use `--device cuda` on a compatible GPU.

Each image input is a horizontal grayscale strip with one glyph per 32 × 32 cell and black ink on white. The sum of passage and question lengths must fit in 509 cells. For ordinary typed input use `--passage-text` and `--question-text`. Font rendering precedes inference. Raw book page segmentation is outside this entry point.

The twelve relations are authorship, lyrics, composition, singing, directing, screenwriting, starring, producing, father, mother, school and ancestral origin. Both the predictions and the copied names can be wrong. The examples preserve real returns, including incorrect answers after changing two names in a film note. The upstream language models are the CKIP Chinese QA reader and UER Chinese GPT-2. DuIE 2.0 research data are obtained separately from the organizers.
