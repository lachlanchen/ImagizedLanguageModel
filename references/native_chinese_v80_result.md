# Full native context does not qualify the trained image writer

The completed native-context model fails the original Chinese development
quality gates. Its 128 normal-question answers have **0/128 exact reference
images**, mean foreground F1 **0.3650**, and **0/64** pairs with both reference
images correct. These are physical-image measurements, not independently
measured semantic accuracy. The answers are generated autonomously from prompt
pixels, with no supplied answer code or length.

Erasing the question reduces Chinese foreground F1 to **0.2540**. The paired
gain is **0.1110**, with a 95% article-bootstrap interval of **[0.0858, 0.1402]**
(64 articles, 2,000 resamples, seed 20266831). Thus question content affects
the output, but this sensitivity does not establish correct answers. All
streams stop; learned stopping also does not establish the correct answer
length. Three of the six original gates pass: mean question gain, a positive
lower confidence bound, and stopping. Exact answers, foreground F1 and paired
correctness fail. The complete 384-output audit includes normal and erased
questions for 128 Chinese and 64 English cases. English was not trained in this
run and has 0/64 exact normal images and foreground F1 0.0586.

Training completed 40,000 updates and 160,000 example draws over all 3,193
Chinese training records. Active training time was 12.62 hours; peak allocated
GPU memory was 18.298 GiB. The complete actor contains **7,583,221,875
parameters**, including **46,899,713 trainable parameters**. Most weights are
externally pretrained and frozen. This is not a lightweight model trained
from scratch, and memory fit is not evidence of language quality.

The [four scheduled final training outputs](evidence/native_chinese_v80_update40000/all_outputs_and_targets.png)
also have 0/4 exact images, with foreground F1 0.6037 and first-block F1
0.8319. All four stop; output lengths are 4/1/2/2 blocks versus targets of
6/1/2/3. Each example received 56 optimization exposures. An unblinded visual
inspection finds a readable “伦敦” with pixel defects, while longer responses
are damaged or truncated. This is not a separate semantic score, and these
four seen examples provide no generalization evidence. Different cases and
exposure counts prevent treating the saved training snapshots as a matched
learning curve.

Three completed command-line calls exercise image input, the same input
rendered from text, and an image with an additional rendered instruction.
The first two have identical input pixels and bit-identical generation.
All three fail the one owning copy reference, with foreground F1 0.2370,
0.2370 and 0.2272 respectively. These are interface checks on one copy item,
not three independent copy questions or a successful copy benchmark.

The [evidence receipt](evidence/native_chinese_v80_result.json) retains the
training report, complete audit report, all original output hashes, CPU
verification and standalone-call verification. CPU verification independently
reconstructs prompt layouts and scores all 384 physical outputs; aggregate
statistics share the producer's aggregation implementation. It does not replay
the entire inherited neural forward pass. The development set has been reused;
there are no sealed evaluation examples in this result.

The [fixed 32-case diagnosis](native_corpus_fit_v80_protocol.md) compares free
generation with correct-history prediction and supplied-code reconstruction.
It is running on the completed checkpoint. Until these controls finish, the
relative contributions of writing-code prediction, feedback error and
generalization remain unresolved. Retaining full context has not established
a usable ILM, efficiency over a conventional language model, or general
historical-script understanding.
