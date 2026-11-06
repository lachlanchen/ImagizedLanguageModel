# Visual conditioning and the still-unproved answer-writing step

Primary-source follow-up, 7 November 2026. These papers inform the next
experiment; their results are not local ILM measurements.

[FlowInOne, Sections 5.1 and G](https://arxiv.org/html/2604.06757v2)
is close to the desired input interface: instructions are rendered visually,
and an image-conditioned flow produces images. Its evaluated tasks are
generation and editing, rather than free Chinese answer writing. The authors
report 1,108.40M trainable flow parameters plus a 103.65M text-image VAE, with
322.58M visual-encoder and 83.65M image-VAE parameters frozen. The advertised
1.2B size therefore excludes those frozen components. Training uses inherited
CrossFlow weights, 240,000 updates, global batch 512 and approximately 240
A100 GPU-hours. It supports investigating a pretrained visual generative
prior, but does not demonstrate an independently trained general language
model on one 4090. The [author code](https://github.com/CSU-JPG/FlowInOne)
releases paired-image training and inference; no additional model or service
has been installed for this review.

[One Image, No Tokens, Sections 3.2–3.5](https://aclanthology.org/2026.mrl-1.3.pdf)
compares visual Chinese input with index embeddings using a shared GPT-2
decoder. Despite its title, the output remains next-character classification
over 3,002 categories, including two special symbols. The visual branch also
receives patch-level and global character-class supervision. Its reported
input-representation advantage does not establish image-output writing,
unrestricted novel glyphs or a training route without symbolic supervision.
Its English comparison also cautions against extrapolating a Chinese input
advantage to every script. This is relevant input evidence, not an answer to
the output failure observed in this repository.

The author-published [Klein LoRA guide](https://huggingface.co/blog/black-forest-labs/flux-2-klein-lora)
distinguishes the base checkpoint used for training from the four-step
distilled checkpoint used for inference. Its consumer-GPU claims concern
style/edit adaptation, not language learning. The current prepared baseline
uses the distilled checkpoint without local optimization. Any subsequent
paired-image training needs a separate specified experiment, correct source
time/latent conventions and measured memory/quality; a generic flow loss on
the inference checkpoint is not automatically a reproduction of that guide.

The practical hypothesis is to reuse an already learned visual generator
and adapt its image-conditioning path. Accurate codec reconstruction is only
the first prerequisite. Complete autonomous new-string copying, correct
held-out answers, paired-question sensitivity and actual cost at matched
quality remain necessary. Neither a paper title nor a smaller reported
trainable parameter count closes those requirements.
