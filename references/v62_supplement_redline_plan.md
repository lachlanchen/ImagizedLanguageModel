# V62 supplement: compiled revision record

Prepared 5 November 2026 before TeX authoring. The completed transcript-bounds
revision has a preserved V62-primary baseline and a clean compiled supplement,
but lacked a compiled change-marked PDF because `latexdiff` was unavailable.
The article source is unchanged relative to that baseline. V63 still trains;
this package adds no new language result and does not alter its comparison.

Scope:

- Preserve all 258 currently tracked publication files byte-for-byte, except
  a short navigation addition to `publication/ilm-nature/README.md`.
- Add a reproducible Git-baseline redline builder, a flattened change-marked
  supplementary TeX/PDF pair and its source/build/visual verification receipt.
- Use the baseline commit recorded in `baselines/v62-primary/baseline_manifest.json`
  (`8fb7ce0a7867bd5d9dad00d891446d34bb0c5c64`) for every old TeX dependency.
  Do not silently use current files as missing baseline inputs.
- Mark the already completed auxiliary reader/transcript-sensitivity addition
  in Supplementary Note 21. Preserve all current numbers, claims, figures,
  bibliography, article sources, deck and primary PDF aliases.
- Reuse the installed TeX/Perl tools. The portable official `latexdiff` 1.4.0
  script, commit `57d0ec532c41eb73645804d7f67667336da8bd01`, is kept under
  ignored `artifacts/external/`, without modifying the training environment.

Allowed new files: `publication/ilm-nature/build_supplement_redline.py`,
`publication/ilm-nature/verify_supplement_redline.py`,
`redlined_supplementary_information.tex`,
`redlined_supplementary_information.pdf`, evidence receipts and this plan's
verification note. The only allowed existing publication edit is its README.
No reviewer-response letter applies to this internal revision record.

Acceptance: authenticate the baseline/tool, recursively expand literal TeX
inputs, require unchanged shared style/old figure bytes, generate CFONT/SAFE
markup, compile with XeLaTeX, inspect every rendered redline page and the
changed subsection/table at readable resolution, and check for clipping,
unresolved references or overfull boxes. Hash-verify that all other current
publication files remain unchanged. Keep a separate locally verified Nutstore
redline package; preserve the five existing primary LATEST aliases. A compiled
redline is a review artifact, not a replacement for the clean supplement.
