# Occupancy hierarchy learns pixel agreement but fails physical support

The registered comparison is terminal. Both 2,261,074-parameter heads complete
96 whole-record updates from identical initial tensors, using the same actual
pre-target state/image cache. Tree F1 improves from .056570593 to .455979811;
flat F1 changes from .247534314 to .004023445. Three of six fixed support
conditions fail. There is no new trained successor, full-fit admission or
language score.

| Complete training record | Positions | Final flat F1 | Final tree F1 |
|---|---:|---:|---:|
| complete_513_book | 512 | 0.004313913 | 0.455571809 |
| complete_qa | 5 | 0.000000000 | 0.378807947 |
| longest_native_record | 35 | 0.000000000 | 0.469458792 |

Neither final arm has an exact source mask. Both emit ink at all four real
source spaces. Zero false blanks means only that a generated mask is nonempty;
flat outputs can satisfy that condition with a single pixel. The tree exceeds
the unchanged flat-gain requirement and its rotated-state NLL difference is
.707847 nat/image, above the .5 support floor. Neither fact qualifies readable
writing or correct answers. The three sources were used for fitting; there is
no held-out generalization claim.

All six complete-record gradient probes have finite nonzero contextual-leaf,
root and mask-output gradients. Every actual event's cached/parallel probability
replay passes, with a maximum error of 9.54e-7 nat. All 96 updates per arm reset
exactly; source actor/cache and original language readings remain unchanged.
The [CPU export](../scripts/export_occupancy_density_preflight.py) independently
recounts every one of the 2,208 generated images and decodes every retained
physical trace to its exact output pixels. It verifies the persisted probability
replay arrays without another neural evaluation. Conditional NLL values remain
measurements of the original bound neural execution.

The [post-hoc ink-count diagnosis](evidence/occupancy_density_ink_mass_20261118.json)
uses only those saved arrays. Source glyphs average 164.824 ink pixels; final
flat outputs average 1.772 and tree outputs 418.797. Among 548 nonempty source
positions, 547 flat outputs have at most four ink pixels, while 366 tree outputs
contain more than twice their source ink. Those descriptive bins are not new
acceptance rules. Source/root aliases, true spaces and all prior scores stay
unchanged. Opposite mass errors and correct numerical replay support an actual
conditional-writing failure; they do not identify its unique cause.

The source-image NLL decreases on every case in both arms. For books, it falls
from 838.653 to 187.046 nats/image in flat mode and from 307.375 to 210.242 in
tree mode. Thus lower teacher-forced likelihood loss still coexists with
unusable greedy writing. The mathematical factorization remains normalized;
this outcome does not make teacher forcing an invalid probability objective.
It rejects the tested finite-budget model/optimization recipe.

The run takes 162.945 monitored seconds with no pressure wait, .221 GiB peak
allocated GPU memory and 53.419 GiB minimum sampled available host RAM. The
outer supervisor records 180.180 seconds, including that interval. These are
head-only costs. The 1,465,381,142-parameter source actor and its external
pretraining, previous fitting and reused cache preparation remain part of the
provenance. No lightweight foundation or matched-quality efficiency is proved.

The [bound result](evidence/occupancy_density_preflight_result_20261118.json)
and [all twelve actual image sheets](evidence/occupancy_density_preflight_20261118/)
preserve the complete outcomes. The earlier lossless/event-cost result remains
valid but cannot be promoted to language learning. Do not extend this trial,
select a different snapshot or relax its failed conditions. The
[next candidate](global_visual_choice_direction_20261118.md) targets a global
whole-image decision before local physical prefixes. It remains unimplemented
and requires its own source, gradient/update/reset and real-writing qualification.
