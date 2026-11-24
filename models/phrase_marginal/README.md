# Writing complete answers from pictured questions

This checkpoint reads a pictured question and source regions and emits the answer as 32 × 32 glyph cells with a learned stop. It uses the same image writer as the longest-run model and credits all ways of writing the target during training. The 866,410-parameter writer and shared 47,522-parameter learned selector run independently on CPU or GPU. The input currently requires aligned cells. A new source glyph can be copied; spelling from memory uses the learned bitmap bank.

On 192 fresh English questions the writer reproduced ten full human answer images. In one case it wrote “Napoleon offered public support to secondary education.” In another it omitted “online” from the statement that a publication began publishing online in 2012. On 192 fresh Chinese reading questions it produced zero exact images. [Examples](examples) · [Scores, factual interventions and transfer](../../references/optical_phrase_learning_result.md)

From the repository root, run either packaged question with its full source PNGs. Choose an unused output directory.

~~~bash
python -m scripts.write_region_writer --phrase-checkpoint models/phrase_marginal \
  --input models/phrase_marginal/examples/1/input.json --output /tmp/ilm-napoleon
~~~

The output contains the chosen region, actual answer PNG and a stop/hash receipt. The second example is in the adjacent examples/2 directory. [PNG input specification](../region_writer/README.md) · [Four CPU inference checks](../../references/evidence/phrase_marginal_v1/png-replay.json)

To prepare the bound training data and reproduce the fit using the released selector and original corpus preparation:

~~~bash
python -m scripts.prepare_region_writer
CUDA_VISIBLE_DEVICES=0 python -m scripts.prepare_writer_regions
python -m scripts.prepare_phrase_confirmation
python -m scripts.prepare_marginal_confirmation
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_marginal_phrase \
  --model-dir artifacts/marginal-reproduction/model --run-dir artifacts/marginal-reproduction/fit
~~~

Training validates the sequence order, input and selector identities. The public receipt records the weight hash, exposure, wall time, allocated memory and device energy. The comparer uses the two previously released writers and the original locally prepared corpora. [Fit receipt](fit.json) · [Formal identity](../../formal/span_list/OpticalEdgeMass.lean) · [Implementation](../../ilm/visual_lm/marginal_phrase.py)
