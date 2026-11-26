# Read a pictured request for an original form and meaning

On the supplied example page, ask for the seal form of 遘 first and its original gloss second. The answer contains the original historical specimen and the English wording “Meaning to meet or to come across.” The command renders the typed question into a grid, reads the page and question as images, and outputs the actual selected pixels.

```sh
python -m scripts.write_origin_values --read-order \
  --prompt '查「遘」，篆書字形在前，本義在後。' \
  --page models/origin_value_reader/example/page.png \
  --geometry models/origin_value_reader/example/geometry.json --output seal-first.png
cmp seal-first.png publication/ilm-nature/figures/visual_order/0001-new-ordered.png
```

Changing the request to `查「遘」，本義在前，篆書字形在後。` reverses the order. Its reference output is [the second raw strip](../../publication/ilm-nature/figures/visual_order/0001-new_swap-ordered.png). For an image request, pass `--question question.png` and provide its occupied cell count in the page geometry. The reader takes no text input internally. The page must supply the requested gloss and specimen in labelled rows. [New model, original outputs and limitations](../../references/visual_order_result.md)

The fixed split, training-only wording variations and entirely held phrasing are defined in the two protocols. Reproduce from the prepared Hanziyuan family shards and tile bank. Each learned model and scoring pass preserves its original output receipt.

```sh
python -m scripts.train_origin_instruction
python -m scripts.evaluate_origin_instructions generate
python -m scripts.evaluate_origin_instructions replay
python -m scripts.fit_visual_order
python -m scripts.evaluate_visual_order generate
python -m scripts.evaluate_visual_order replay
python -m scripts.audit_visual_order_transfer
```
