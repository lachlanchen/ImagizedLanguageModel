# AgInTi Figure Generation Brief

Purpose: create publication-safe PNG figures for an image-native language model paper.

Style:
- Clean academic diagram.
- White or very light background.
- Flat boxes, clear arrows, no decorative gradients.
- Labels must be large enough for a two-column paper.
- Figures should compile as PNGs in LaTeX.

Figures:

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
- This keeps the paper reproducible even if an interactive image agent is not available.

