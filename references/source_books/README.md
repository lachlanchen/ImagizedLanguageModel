# Local Source-Book Registry

This directory registers the local books used to design and evaluate the
image-native word-origin model. It does **not** make the books a redistributable
training corpus.

`catalog.json` records the intended research role of each work. Running the
registry script creates:

- ignored relative symlinks under `library/`, avoiding a second roughly 1 GB
  copy of the same archives;
- `manifest.json`, with source size, SHA-256, media type, PDF page metadata,
  and the local link target; and
- a failure if a source is absent, changed, or would overwrite a nonmatching
  local file.

Register or verify the current workstation collection:

```bash
PYTHONPATH=. python scripts/register_reference_books.py \
  --source-dir /home/lachlan/Downloads

PYTHONPATH=. python scripts/register_reference_books.py \
  --source-dir /home/lachlan/Downloads --check
```

The source binaries and extracted text stay local and git-ignored. Rights are
currently unverified, so these works may be used for private source comparison,
layout research, and evaluation, but they must not enter a released dataset or
redistributable checkpoint until their licenses are documented.

For extraction experiments, the existing tools in the sibling `ZhJpBook`
repository can produce auditable OCR/text sidecars. Such text is offline
teacher metadata only. It must be removed before an ILM student batch is formed;
the deployed student receives writing images and continuous visual states.
