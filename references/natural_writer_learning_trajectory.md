# Measuring complete answers over writer training

Register steps 0, 9,000, 18,000, 27,000 and 36,000 before the current fit
starts. Step 0 is the fixed parent. Intermediate files contain only the
writer weights that change. Their step, parent, preparation and file hashes
bind them to the same fit. Step 36,000 remains the final candidate. The
intermediate measurements provide a trajectory for the paper.

At each step, use the same 32 reserved image prompts and the actual learned
stopping and emitted-pixel feedback. Hold the frozen reader, contextual body,
encoder and decoder fixed. Use the existing complete-answer definition and
preserve all streams, including malformed output, premature stops and caps.
Review the actual images before reference scoring. The registered parent and
final runs supply their fixed endpoints. See the
[complete-answer protocol](natural_answer_assay_protocol.md).

The paper figure should plot complete answers and paired-context success
against actual optimizer steps on this cohort. A companion panel should show
the same prospectively chosen raw image outputs at every step. Record the
physical first-image score and training loss separately with their own units.
Editable PowerPoint data and the TeX figure must agree with the measured
reports. Intermediate generation and figure production await the fit.
