# Predicting a Chinese expression one drawing at a time

A 2.37-million-parameter model learned to condition its drawings on a pictured Chinese passage. On 17,686 new missing-expression cases, it wrote 136 complete stopped answers exactly with one committed history, versus zero after the same training with whole passages shuffled. Retaining four histories raised the result to 176. The existing uniform writer recovered 547, and a training-only local count baseline recovered 960. The new architecture remains experimental.

## What the model learned

Each passage supplies 24 pictured glyph cells on either side of a gap. A language state reads the passage and measurements of the last completed drawing. A separate recurrent network predicts twelve groups of four physical ink bits. The renderer turns those 48 bits into a 32-by-32 image. A binary decision can stop the writer at each glyph boundary. The deployed computation for a research checkpoint requires only those weights and the pictured prompt.

The first comparison used 142,878 unique training windows four times per arm. With the correct alignment, 70,705 of the 70,744 emitted cells exactly matched familiar glyph images. The planner wrote 106 complete expressions whose four target characters were all absent from the visible window. The first newly correct answer against the uniform writer was 众目睽睽 before the words 之下，犯罪嫌疑人. The first lost answer illustrates a different outcome. Given 新浪娱乐讯正在 before a gap and 筹拍中的新版《水浒》 after it, the planner wrote 千可万别. The reference was 紧锣密鼓.

| Retained answer histories | Exact first glyph | Two-glyph prefix | Three-glyph prefix | Complete answer |
| --- | ---: | ---: | ---: | ---: |
| One | 1,283 | 277 | 165 | 136 |
| Four | 1,132 | 310 | 217 | 176 |
| Existing uniform writer | 1,221 | 584 | 552 | 547 |

All 17,686 original outputs, lengths and stops reproduced with the one-history decoder. Four-history search gained 49 answers and lost nine. The 95% paired source-group interval was 0.15 to 0.31 percentage points. The aligned planner predicted 1,283, 1,699, 2,799 and 3,172 correct glyphs when supplied the correct preceding images. These are independent position counts. Only 136 passages were correct at all four positions. Later glyph choice remains difficult even when previous drawing errors are removed.

## Continued learning and teacher supervision

On a second source partition of 17,723 gaps, the same planner initially wrote 191 complete answers. Four additional passes with real answers yielded **435**. Adding the stronger writer's conditional ink probabilities at equal exposure yielded **383**. The teacher-assisted fit gained 40 and lost 92 relative to source-only training, with a 95% paired interval of −0.43 to −0.17 percentage points. The stronger writer itself recovered 540 and local counts recovered 977. The source-only fit wrote 303 answers in which all four characters were absent from the prompt. Seven matched pairs had two correct completions. The teacher-assisted fit solved four pairs and returned its ten most common answers on 14,999 of 17,723 inputs. Neither new model qualified for release.

The first two fits together took 57.8 seconds on one RTX 4090 D and peaked at 290.7 MiB allocated GPU memory. The matched additional fits took 71.0 seconds including the offline teacher in both arms, with 322.5 MiB peak allocated GPU memory. These are incremental training costs. The source preparation, earlier reader and motor training, evaluation, and a quality-matched token-model comparison would be needed for a claim about total efficiency.

## Transfer to ordinary passage questions

In the fixed CMRC collection the 1,788 held-out answers span 425 passage groups. Only 680 answers fit in the eight output slots used for idioms. When supplied the true 48 landmark bits per character, the frozen renderer reconstructed 23,170 of 34,080 glyphs but only 219 complete answers. Just 183 answers met both the length and canonical-rendering conditions. This is a test of one input path to the renderer. Other latent patterns could in principle draw a missed image.

A training-inventory diagnostic helps explain the difference. The idiom training answers contained 2,335 distinct physical glyphs with distinct landmark signatures. Adding the QA training answers expanded the inventory to 3,975 images, with 14 ambiguous signatures shared by 32 images. Given the **true landmarks** of each held-out answer, choosing the most frequent training bitmap per signature recovered 221 answers using idioms alone and 1,612 using both training sources. All 1,788 held-out reference answers occurred as an exact contiguous sequence of image cells in their passage. That measures availability. Selecting the correct sequence still requires reading the question. A joint model can therefore be tested on selecting exact source ink for ordinary questions, generating absent forms for idioms, and learning when to stop over both tasks.

## Running the research checkpoint

From the repository root, the command below runs the source-only eight-pass checkpoint on CPU and produces an image, its original binary tiles, and a stop receipt. It also accepts separate black-on-white PNG inputs via `--before-image` and `--after-image` with 32-pixel square cells in reading order. An optional input-cell count omits padded cells. The current research weights are local training artifacts. The released general project entry point continues to use the verified joint-reference weights.

```bash
PYTHONPATH=. python -m scripts.infer_glyph_planner \
  --weights artifacts/glyph_planner_transfer_v1/reference/weights.pt \
  --before-text '' --after-text '之下，犯罪嫌疑人手持利刃' \
  --output /tmp/ilm-planner-answer
```

The standalone CPU entry point reproduced the original GPU pixels, lengths and stops for both published examples, through both typed and image input, with one and four answer histories. To reproduce the fits in a prepared workspace with the original pictured corpus and initial weights, run `scripts.train_glyph_planner` with `train` then `evaluate`, followed by `scripts.search_glyph_planner`, then `scripts.distil_glyph_planner` with `train` and `evaluate` as Python modules. Each stage retains its original results and requires a fresh destination for a reproduction.

[Physical outputs and numerical receipts](evidence/glyph_planner_20261118/) · [Initial experiment](glyph_planner_protocol.md) · [Search experiment](glyph_planner_search_protocol.md) · [Teacher comparison](glyph_planner_transfer_protocol.md)
