# Complete-record writer-conditioning preflight

This is a training-only support comparison, not a successor language fit.
It implements the [layer-conditioning candidate](writer_layer_conditioning_direction_20261118.md)
after the failed [reader image-consequence test](reader_writing_consequence_result_20261118.md).
Preserve all failed coordinate and language gates. Do not install either
calibrated reader: use the exact original final learned-context actor in both
arms, with its original reader, codec and length/stop head unchanged.

Bind the same complete 513-, 846- and 1,042-glyph training records. Observe each
whole source once with reference-image history and known span/update counts;
retain every predicting state, target latent and binary target. Each context
precedes its own target. There are 512, five and 35 target glyphs, respectively.
The cache is a fixed input for both writer trials; no source embedding table,
teacher head or target-conditioned prefix state is loaded. This does not test
generated feedback, segmentation, independent reading or stopping.

Use the common explicit layer loop for both arms. The control has no added
modulation; the candidate has three zero-initialized scale/shift maps applied
after its three existing transformer blocks. Require bit-equal initial
candidate/control velocities and generated images. Separately record the
common loop versus the original TransformerEncoder's velocity discrepancy;
require relative RMS at most 1e-5, without changing the old execution or scores.
All new maps must receive finite nonzero gradients, and every old writer
gradient must match its control initially within 1e-5 relative RMS.

Freeze the whole actor outside the velocity writer, including its length head.
Both trials start from the same actual final weights and receive exactly
twelve AdamW updates: four passes through the three whole records in registered
order. Use learning rate 2e-4, weight decay .01 and global gradient clipping
at one. At each update accumulate the complete valid-glyph mean flow gradient
over microbatches of sixteen spans before one step. No partial record is an
update. The reference latent is interpolated with a standard Gaussian at an
independent uniform time. Seed each update with 202611187+update index; preserve
noise/time hashes and verify their equality across arms.

Evaluate all 552 target positions before and after fitting, with supplied
lengths, eight Euler steps and per-span noise seed
202611188+100000*record index+span index. Preserve every raw latent endpoint,
binary output and canonical pixel/whitespace metric. No selection, source
pixel repair, new sampling setting or additional unchanged updates follows
inspection. Reset both trial writers exactly to their own initial hashes;
verify the original saved checkpoint, complete frozen actor and length head
unchanged. Retain no trained successor for deployment.

Support requires all computational checks, candidate pooled F1 at least .05
above initialization and .02 above control, no case-level F1 deterioration
against control, and no increase in pooled false blanks or false ink against
either initialization or control. These are physical support conditions,
not readability or language criteria. A failure stops this intervention at
this budget. A pass permits designing a separately registered full-source
comparison; it does not qualify the reader, admit an automatic full fit or
alter any autonomous language score.

Commit/push implementation, CPU checks, protocol and source registration before
neural execution. One GPU0 mutex owner; two CPU threads; entry 48 GiB host,
20 GiB free GPU and 48 GiB disk at low PSI; runtime host reserve 24 GiB and
maximum allocated GPU 22 GiB. Monitored deadline 1,200 seconds, pressure-wait
allowance 300 seconds; external deadline 1,260 seconds with thirty-second grace.
No automatic retry. Keep foreign services, all original outputs and the full
independent visual-language goal unchanged.
