# A historical form from the wrong dictionary entry

On two-entry dictionary cards, the image-stream writer began all 1,136 standard answers with the requested headword. Of the 1,129 answers it finished, 561 used that word’s attested historical form and 568 used the other word’s form from the correct stage. For example, a request for the bronze form of 耳 elicited the bronze form of 乃. This is an error in attributing evidence to a word. Both glyphs are readable. [Figure 79](../publication/ilm-nature/figures/Figure_79_general_entry_writer.png) · [Independent physical recount](evidence/general_entry_writer_20261119/audit.json)

| Complete answers | Before adaptation | Shuffled questions | Matched questions |
|---|---:|---:|---:|
| Standard dictionary requests / 1,136 | 0 | 38 | 284 |
| Held-out paraphrase / 1,136 | 0 | 46 | 260 |
| Independent Chinese requests / 96 | 0 | 2 | 19 |
| English requests / 1,136 | 0 | 46 | 74 |
| Erased questions / 1,136 | 0 | 59 | 19 |
| Both DRCD passage answers / 1,042 | 21 | 16 | 12 |
| Canonical natural DuReader answers / 571 | 16 | 16 | 15 |
| Natural references requiring absent content / 300 | 0 | 0 | 0 |

Matched supervision improved whole standard answers over shuffling by 21.65 percentage points (95% interval 19.37–23.94, resampling 142 cards). The increase was 17.71 points (8.33–28.13) across 96 independently written requests on 24 cards. None of the 142 cards had all eight answers correct. The matched checkpoint failed the registered retention criterion for both modern corpora. The released specialist passage reader remains the stronger passage baseline. [Registered protocol](general_entry_writer_protocol.md) · [Complete results](evidence/general_entry_writer_20261119/result.json)

Both fits started with the same 628,104-parameter writer and 7,715-glyph writing memory. Each saw 267,600 dictionary presentations and 33,480 rehearsal examples from each of two natural corpora. Matched training took 115.3 seconds on one RTX 4090 D and peaked at 0.69 GiB of allocated PyTorch memory. The training cost excludes the common initialization. All 568 evaluated historical shapes were absent from entry training and writing memory. Each requested shape was available to copy from the pictured evidence.

## What changes when the distractor is removed?

The frozen writer reproduced its original 1,136 standard outputs byte for byte using the original evaluation precision and feature bank. Swapping the two intact records changed 277 responses but gave a similar 282 exact answers. Presenting only the requested record raised exact answers from 284 to **1,104 of 1,136**. It recovered 820 and lost none. Every one of the 1,119 stopped answers used the correct historical shape. Supplying only the distractor gave zero exact answers. An initial replay at different numerical settings failed the identity check and was excluded before testing the interventions. The cropped condition supplies the correct record location. Learning to locate that record from the prompt is the remaining problem this test isolates. [Probe design](entry_order_binding_probe.md) · [Fixed-weight results](evidence/general_entry_writer_20261119/order_probe.json)

## Resolving a visible headword

An optional locator reads the question and card as images, searches for the pictured headword between the card’s printed brackets, and crops only when there is one match. It uses no answer key, field labels or stop information. On the same standard questions this raised exact stopped outputs from **284 to 1,103 of 1,136**. It recovered 819 answers and lost none. It also raised complete answers on the held-out paraphrase from 260 to 1,094 and on independently worded Chinese from 19 to 60 of 96. All 1,136 English queries selected the correct record, yet only 249 answers were exact. Reading the headword and understanding what is asked are distinct capabilities. The locator changed no natural passage inputs and preserved the 19 erased-question successes. This component requires an explicit matching headword and the pictured bracket layout. [Protocol](visible_record_selection_protocol.md) · [All five conditions](evidence/general_entry_writer_20261119/selection.json)

A separate recount compared each emitted picture with all eight stopped answers on its original card. Of the 887 English errors after location, 214 answered the wrong lexical field, 217 used the wrong stage, and 200 did both. Each of those 631 images was an exact answer to a different question about the correct headword. Another 138 matched none of the card answers, and 118 reached the output limit. Among the 33 remaining standard Chinese errors, 17 reached the limit and 16 matched none of the card answers. [Physical error recount](evidence/general_entry_writer_20261119/located_errors.json)

To use a previously pictured card directly, pass `--locate-entry` with `--checkpoint-dir artifacts/general_entry_writer_v1/paired` to `python -m scripts.read_grounded_glyph_answer` alongside the two PNGs and output directory. The locator is optional. Its exact-pixel assumption leaves handwriting variation and differently rendered fonts open.

## Ask about a local entry

With the fitted checkpoint and verified local Hanziyuan cache available, this command retrieves an entry and gives the question and card pixels to the writer. It produces the source, prompt and answer PNGs, raw answer pixels, a post-generation reading of identifiable shapes and a provenance receipt. A question PNG can be supplied with `--question-image` and an optional occupied-cell count. Each cell is 32 pixels square.

```bash
python -m scripts.ask_visual_dictionary --char 言 --stages oracle bronze \
  --question '请写出言的本义和金文字形。' --output /tmp/ilm-yan-bronze
```

The writer returned “言的本义是talk，金文作[B02975]。” A second request for pronunciation and the oracle form returned “言的读音是yán，甲骨文作[J04903]。” Bracketed identifiers stand for actual historical bitmaps in the output. Both runs used exactly the same pictured entry. Re-entering the bronze question as an image reproduced the full stopped answer byte for byte. The two forms were absent from the entry training set. The headword and facts had appeared with seal S01648 and Liushutong L06986. These are source-grounded responses to one-entry questions; writing an independent history of 言 requires further evidence. [Bronze answer](evidence/general_entry_writer_20261119/yan_bronze.png) · [Oracle answer](evidence/general_entry_writer_20261119/yan_oracle.png) · [Hashes and source labels](evidence/general_entry_writer_20261119/figure.json)
