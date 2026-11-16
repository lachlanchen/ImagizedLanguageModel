# The unchanged cell policy does not preserve the new preference source

The registered CPU inventory covers all 10,735 prepared preference pairs and
all 6,118 distinct code points in system, prompt, chosen and rejected fields.
It finds missing forms, clipped isolated renders and formatting collapsed to
blank. The unchanged cell policy is therefore not a faithful importer for
this complete source. No preference fit is admitted.

This is a new-source preparation result. It does not diagnose or repair the
earlier language trials, which did not train on this preference collection.
The running white-target comparison and every original frozen reading remain
unchanged. [Protocol](chinese_preference_raster_protocol_20261116.md),
[registered inputs](evidence/chinese_preference_raster_registration_20261116.json),
[result and recount](evidence/chinese_preference_raster_result_20261116.json).

| Code-point outcome under the existing target font | Distinct forms | Occurrences |
| --- | ---: | ---: |
| Visible inside the cell | 5,142 | 10,302,078 |
| Clipped isolated render | 4 | 10,897 |
| Unsupported by this font face | 964 | 13,281 |
| Legacy policy maps to blank | 8 | 646,421 |

The font is the existing Noto Sans CJK SC face 2 at 26 pixels, with a 32-by-32
cell and baseline y=26. Each supported visible form is also drawn on a
128-by-128 canvas at an integer-translated origin. Every reference crop equals
the original cell. Clipping counts ink outside that cell, rather than inferring
it from a nominal font box. Frame contact is recorded separately.

The vertical bar U+007C occurs 10,878 times and loses one ink pixel below its
cell. Combining grave U+0300, acute U+0301 and macron U+0304 occur 7, 11 and 1
times; their isolated shaped renders also extend beyond the cell. The sample
plate shows the combining renders with dotted-circle scaffolding. Drawing a
combining mark separately does not establish faithful rendering of its source
grapheme. Those marks need contextual shaping rather than a repaired image or
an inferred missing character.

Unsupported forms include other scripts, emoji, Latin variants and variation
selectors. The source contains 788 U+FFFD replacement characters. Their presence
in the authenticated upstream file is preserved; no missing intended content
is guessed. These counts concern one preparation font, not the model's physical
image support or a requirement for a fixed deployed character inventory.

At pair level, the overlapping issue counts are:

| Source issue in at least one writing field | Pairs |
| --- | ---: |
| Formatting mapped to an ordinary blank cell | 10,047 |
| Cell frame touched | 3,998 |
| Unsupported code point | 758 |
| Clipped isolated render | 287 |
| Combining mark needs layout review | 129 |

The eight blank-mapped forms include ordinary space, tab, line feed, carriage
return, nonbreaking space, zero-width space, nonjoiner and joiner. In particular,
273,550 line feeds become indistinguishable from ordinary spaces under this
policy. That does not prove that every affected answer is semantically damaged;
it prevents a claim that paragraph, table, code or joining structure is preserved.

Only 556 rows satisfy the conservative inventory filter together with the
earlier structural check: 507 prospective training rows and 49 prospective
calibration rows. Of those, 320 and 32 respectively have both answers within
128 code points; all 556 fit 513. These are descriptive counts, not qualified
training examples. Silently retaining this remainder would replace the proposed
complete collection with a much smaller task. All 10,735 raw pairs remain intact.

There are also 22 groups of identical nonblank cell rasters, including cross-script
homographs and the two cropped accent renders. Equal-looking forms do not imply
equal language meaning, and not every alias is a renderer defect. Appearance
alone need not identify a unique Unicode spelling; context and uncertainty
remain relevant to any optional transcription system.

The inventory takes 10.444 seconds at 0.609 GiB peak process RSS. An independent
recount of all source rows and four field-frequency tables takes 1.049 seconds
at 0.069 GiB. The complete labelled source-font sample plate was actually viewed;
this is not blind model-output adjudication. Input, implementation, shared-font,
inventory and sample hashes are verified. There are no neural calls, optimizer
updates, full-row image renders or new language scores. All 115 sources bound
to the live model comparison remain unchanged.

Before any preference fit, register and qualify an explicit preparation policy
for complete pages or lines, contextual shaping, supported font selection and
visible structural layout. Preserve original source text and report unsupported
cases rather than substituting or truncating them. This audit is terminal for
the unchanged cell policy; do not tune its font size, rerun it unchanged, discard
its failures or repair old actor outputs. Complete-image fidelity and semantic
preference labels still need separate qualification. The existing model's final
writing and language evaluation retains priority.
