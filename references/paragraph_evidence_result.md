# Reading the right paragraph

The frozen date writer originally returned the correct actor with the year from a competing paragraph. A picture-based selector trained on 14,446 assignments took 16.4 seconds of optimization on one RTX 4090 D. It has 61,925 parameters. The writer it serves has 86.6 million parameters and inherits a pretrained Chinese language backbone. All comparisons used supplied paragraph boundaries.

| All replies correct | Whole page | Literal glyph match | Learned selection | Annotated paragraph |
|---|---:|---:|---:|---:|
| Distinct subjects, 48 boards | 8 | 13 | 13 | 13 |
| Same subject in different roles, 34 boards | 2 | 2 | 8 | 10 |
| Independently edited years, 45 boards | 5 | 16 | 15 | 16 |
| Earlier date questions, 64 boards | 15 | 42 | 40 | 42 |
| Competing quotations, 64 boards | 0 | 34 | 28 | 34 |

On shared-subject questions the selector chose the requested paragraph 394/408 times. Literal matching succeeded 306 times. Learned selection gained 94 and lost six. The independently edited distractor left the selected writer output identical on 526/540 pairs. On distinct subjects literal matching found every source. Of 139 wrong responses with the learned selector, 123 came from the correct paragraph. Neither possible paragraph yielded the accepted answer in those 123 cases.

The first recovered response to a question about the actor in 《天外仙》 gave 陈秀丽 and corrected the year from 2011 to 2005. The first remaining error with the correct paragraph gave 曲一线 as author of 《高考1号文件》 and stopped before reporting 2009.

On 302 eligible human questions the selector found the answer-bearing section 279 times, compared with 270 for literal matching. The date writer returned zero exact answers from the annotated section and six from whole pages. A separate extractive reader returned 198 and 202 respectively, and 190 with learned selection. The span reader returns an interval and has no learned stop. A reference had to appear exclusively in one section for inclusion. Natural pages were split at a sentence boundary before locating the answer. Shared-subject factual examples are controlled replacements and can contain awkward wording.

The Lean development bounds routing error by selection errors plus writer errors on the annotated paragraph. A separate finite-margin result shows when perturbations cannot change the winning score. Neither establishes distributional generalization. All 2,878 recorded choices and five sets of complete image replies were recounted. The historical single-page controls reproduced the original writer's pixels on CPU, including its mistakes. [Actual results and image examples](evidence/paragraph_evidence_20261124) · [Protocol](paragraph_evidence_protocol.md) · [Offline inference](../models/paragraph_evidence/README.md)
