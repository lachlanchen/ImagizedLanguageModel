# Compact visual sequence experiment in the research article

This research-draft revision starts from the current historical-forms article.
Its stable reading and review PDFs have eighteen and thirty-four pages. The
previous historical-forms package and all existing figure files remain intact.
No reviewer response is involved. The current paper describes the larger
7.522B contextual experiment; it needs a distinct account of the smaller
complete-sequence experiment and its actual generated-language results.

## Scope and evidence

Use the completed teacher cache and compact fit in
`artifacts/visual_sequence_teacher_v1` and `artifacts/compact_visual_span_fit_v1`.
The fit completed 8,172 updates over 13,785 sequences and 2,380,620 state
targets, with five retained checkpoints. Its 1.409B contextual foundation and
fixed V34 codec remain externally pretrained and project-trained components,
respectively. The image modules learn from the complete visible sequences;
native alignment and the larger text teacher supervise training offline.
This is transfer of inherited language knowledge to a visual interface.

Wait for all five fixed generations, all opaque image readings and their
authenticated scores before adding answer counts or learning curves. Use
`references/compact_visual_span_learning_analysis.md`, preserving all 32 QA
questions, sixteen paired contexts, twelve book prefixes and endpoint controls.
Retain the first registered QA pair and first book output at every checkpoint.
Their identities were fixed before training. No reference repair, selected
successes, or inferred readability can supply missing evidence.

Report the linear flow-head restriction from
`references/compact_visual_flow_rank_result.md` in its stated mathematical
scope. Its 384-dimensional invariant noise subspace is a property of the
continuous velocity model. The decoder projection result concerns
preactivations. Neither measurement is a language result, a hardware-exact
sampler identity, or sufficient evidence about the cause of a particular
incorrect answer. A corrected architecture remains a separate experiment.

## Allowed changes

- Add compact sequence Results and Methods sources, included from
  `publication/ilm-nature/article_content.tex`. Explain the tile stream,
  768-dimensional codec, learned span grouping and continuous feedback,
  2,048-dimensional contextual states, learned span length/stop decision,
  conditional flow and emitted-pixel rereading.
- Give the six implemented training objectives and their actual equal-weight
  sum: relative feedback-vector error, update-count cross entropy, boundary
  binary cross entropy, output-length cross entropy, flow velocity error and
  teacher-context cosine loss. Distinguish reference-image training history
  from generated-image history at inference.
- Update `frontmatter.tex` and the relevant introduction/discussion paragraphs
  to state the compact experiment's observed scope and outcome. Preserve the
  earlier contextual cohort's 16/32 versus 11/32 clear matches and the later
  18/32 versus 19/32 result as different cohorts and metrics. Do not pool them
  with the new canonical-exact cohort or imply a new from-scratch learner.
- Add a clean editable PowerPoint figure with the measured fixed-checkpoint
  counts and pointwise passage-group intervals, and the same fifteen raw
  answer images. Use a separate raw-output figure if one figure would make
  the writing unreadable. Keep native chart data and labels editable. Show
  stop/cap and empty-response status without repairing glyphs.
- Export corresponding PDF/PNG figures and captions. Use generated-language
  scores for language curves; show training losses and resource costs in
  separate panels with their own units. All values come from authenticated
  reports, including the native-text reference's unchanged exact scores.

## Verification and delivery

Create the figures only after the final score files exist. Check chart workbook
values against the analysis CSV and raw-image hashes against the generations.
Inspect the slide and PDF renders for alignment, font substitution, readable
labels and intact glyph images. Record the actual design and rendering tools.

Build the reading, review and redlined manuscript PDFs against the recorded
pre-revision Git baseline. Inspect every changed page and caption in the
rendered PDFs. Use bounded CPU builds; run the full package builder only after
the active model jobs have exited and the project mutex is available. The
existing supplementary package and composite deck remain unchanged unless a
specific dependency requires a separately documented update.

After verification, sync a new versioned package to the owned Nutstore ILM
project folder and update only the two owned latest manuscript aliases after
checking their previous hashes. Verify local copies; cloud propagation needs
separate evidence. Commit and push the completed revision. The current step
records this plan; it does not claim that the revised article or figures exist.
