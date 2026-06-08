# AgInTi Figure Generation Brief

Purpose: create publication-safe PNG figures for an image-native language model paper.

Style:
- Clean academic diagram.
- White or very light background.
- Flat boxes, clear arrows, no decorative gradients.
- Labels must be large enough for a two-column paper.
- Figures should compile as PNGs in LaTeX.

Figures:

0. `ilm_v_yan_readme_hero.png`
   - README/paper hero figure.
   - Use an AgInTi-generated abstract background with bronze vessel / oracle bone / neural circuit mood.
   - Overlay all readable labels deterministically.
   - Use real local ziyuan glyph exemplars for `言` (YAN, U+8A00): oracle, bronze, seal, modern.
   - Show the core idea clearly: image input -> ILM-V -> image output answer.

1. `architecture_overview.png`
   - Show the flow:
     - Input canvas: book page / oracle glyph / cuneiform-like sign
     - Visual encoder
     - Visual memory transformer
     - Latent generator: masked diffusion / inpainting
     - Output image
   - Bottom row: training objectives:
     - masked visual LM
     - image instruction tuning
     - glyph evolution
     - readability critics

2. `training_curriculum.png`
   - Six phases:
     - data pipeline
     - glyph autoencoder
     - evolution generator
     - visual page LM
     - instruction images
     - multiscript expansion
   - Include a small note that one/two RTX 3090 GPUs are enough for early phases.

3. `zhong_evolution_example.png`
   - Timeline for Chinese character `中`.
   - Panels: oracle, bronze, seal, Liushutong, modern.
   - Use real local glyph exemplars when available.

4. `aginti_artifact_loop.png`
   - Show research -> AgInTi figure brief -> deterministic PNG generation -> LaTeX paper.

Current implementation:
- The PNGs are generated deterministically by `publication/ilm-image-native/generate_figures.py`.
- `aginti_yan_background.png` is the AgInTi-generated no-text background used by the README hero.
- This keeps the paper reproducible even if an interactive image agent is not available.
