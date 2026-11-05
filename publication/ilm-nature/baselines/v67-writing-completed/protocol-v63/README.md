# Question-conditioned image-language training

This methods companion explains the registered V63 computation and completed
implementation preflight. It contains no V63 autonomous-language result.
The main article, existing figures and seventeen-slide deck retain their
completed V62 evidence.

- `methods.tex` / `methods.pdf`: model, objective, controlled comparison,
  implementation checks, inference and independent evaluation requirements.
- `question_assignment_method.pdf`, `.svg`, `.png`: the training mechanism,
  using exact observed images from the first frozen training pair.
- `question_assignment_method_editable.pptx`: one slide with editable text,
  boxes, connectors and the unmodified training-image regions.
- `question_assignment_method_editable.pdf`: rendered slide for review.
- `figure_sources.json`: source identities, displayed pixels and diagram
  object geometry. Energies are symbolic; answer images are observed targets.

The figure retains the publication's Lamp+LCD-derived layout, palette and
type hierarchy. The deck preserves its existing theme and geometry. The
ArtifactTool runtime is unavailable on this workstation, so the native
PowerPoint fallback matches the established project workflow.

From the repository root, reuse the existing Python environment:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python \
  publication/ilm-nature/protocol-v63/generate_method_figure.py
cd publication/ilm-nature/protocol-v63
TEXINPUTS=..: xelatex -interaction=nonstopmode -halt-on-error methods.tex
TEXINPUTS=..: xelatex -interaction=nonstopmode -halt-on-error methods.tex
```

Rendering and byte-hash verification are recorded under `references/`.
The protocol is `references/question_assignment_v63_protocol.md`; source images
come from the pinned CMRC training cache. The figure builder performs no model
inference. Rebuilding requires the existing pinned training cache for its four
displayed image regions. Dataset attribution: CMRC~2018, CC-BY-SA-4.0; external MIXAR foundation
pretraining is separately attributed in the note.
