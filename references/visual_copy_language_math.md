# The image probability used in V57

Let the sensor alphabet be all binary 32-by-32 images, not a list of written
characters. Its finite size is 2^1024. A prompt is a sequence of observed
images X_1,...,X_L. Host text rendering is one possible source of these
observations, but neither Unicode IDs nor strings enter the student.

At answer step t, the causal reader produces state h_t from the complete
prompt and already emitted answer images. Source states k_i come from
positions in the observed prompt. A query may attend only to source positions
already visible to that query. This also matters for natural continuation:
packing a longer training sequence must not make its future images available.

The position distribution and gate are

    a_ti = softmax_i((W_q h_t)^T (W_k k_i) / sqrt(d)),
    lambda_t = sigmoid(w_g^T h_t + b_g).

The copy component is a distribution over images:

    q_t(Y) = sum_i a_ti * 1[Y = X_i].

If the same image appears at several positions, those masses are added.
There are no separately learned identities for these positions. Exact bit
packing is an equality implementation, not a learned visual codebook. Since
each source image is one possible Y, summing q_t(Y) over all Y gives one.

The V55 raster branch is another normalized distribution r_t(Y), a product of
1,024 causal Bernoulli factors within the image. Their mixture is

    p_t(Y | continue) = lambda_t q_t(Y) + (1-lambda_t) r_t(Y).

It also sums to one. The log probability is computed with logaddexp; the copy
term uses logsumexp over matching visible source images. An all-masked copy
term has zero mass and a finite backward path. The free branch still gives
positive finite probability to any binary image under finite logits, including
forms absent from the prompt and every character encoding.

A stop hazard s_t = sigmoid(w_s^T h_t+b_s) completes the answer law:

    p_t(stop) = s_t,
    p_t(Y, continue) = (1-s_t) p_t(Y | continue).

For a reference answer Y_1,...,Y_T, its sequence probability is the product
of the T continue/image probabilities and the stop probability at T+1.
Teacher forcing exposes only Y_<t to a query at t. Gold length determines
training masks, not an input to deployed inference. Finite-prefix event
normalization does not prove that an unconstrained infinite rollout terminates
almost surely; inference therefore records a fixed output cap separately.

The natural-language objective uses the same next-image law at causal
positions in public Chinese streams. Its motor exercises deliberately reveal
the target's image field and train the raster branch alone. Such exercises
verify faithful synthesis and cannot count as prompt understanding.

The primary decoder makes a hierarchical greedy choice: stop, copy or free;
then a copied image mode or raw-pixel beam output. It does not find the global
mode of the mixture. A proper training law alone does not establish fluent
generation, calibrated semantics or efficient learning. These remain empirical
questions, tested here with raw held-out answers and question interventions.

The design applies established copy/generation ideas to observed image
rasters; it makes no priority claim for a pointer-generator or pixel-language
modelling. See [See et al. (2017)](https://arxiv.org/abs/1704.04368) and the
existing pixel-language references in the manuscript. Source selection from
a visible paragraph is also narrower than general free-form language.
