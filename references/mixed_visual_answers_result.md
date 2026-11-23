# Training a glyph writer to answer different kinds of questions

One further epoch of mixed supervision changed what the date-trained writer could do with the same pictured passages. Full-page answers on the existing CMRC and DRCD sets rose from 7/359 to 114/359. The independent span reader retains the stronger score of 242/359. When the correct section was supplied on 302 eligible questions, the mixed writer rose from zero to 115 exact answers. With learned section selection it returned 111.

| Additional simplified-Chinese questions | Accepted answers / 256 | Both questions correct / 128 passages |
|---|---:|---:|
| Date writer | 5 | 1 |
| Mixed writer | 79 | 15 |
| Mixed writer with question erased | 1 | 0 |
| Full-page span reader | 156 | 49 |

The questions on each page were written by people and their acceptable answers were disjoint. The mixed writer gave a different reply on 127/128 question switches. Only the 15 fully correct pairs demonstrate answering both requests. Source and eligibility screens left no further DRCD pairs for this additional test. The original 167 DRCD questions remain reported separately (49 correct). An unknown amount of source exposure during upstream pretraining prevents a claim of entirely unseen text.

On the first exact additional reply, the question asked which music society 邓寄尘 joined as a composer. The emitted pixels read 羊城音乐社. On the catalogue plates, exact responses rose from 18 to 190 of 192 with familiar wording and from zero to 54 of 192 with reworded questions. The known 言 card produced the attested oracle and bronze shapes in either requested order. This tests retrieving and arranging supplied shapes. Each of the 190 exact historical captions included a source bitmap absent from the 7,990-image fixed inventory and a connective bitmap absent from the catalogue page. All 54 correct rephrasings and all four known-card answers also required both sources of ink. Complete learned-selection results rose from 13 to 23 of 48 distinct-subject date boards, eight to 14 of 34 shared-subject boards and 28 to 30 of 64 quotation boards. Earlier routed dates regressed from 40 to 39 complete boards out of 64.

Of the 177 errors on the additional human questions, 94 began with a glyph outside every accepted first position. Another 83 began with an accepted glyph. Twenty-two continued beyond a complete accepted prefix and 16 stopped or diverged at a proper prefix of the answer. The visual span reader recovered 98 of the writer's misses. The writer recovered 21 answers the reader missed. These counts motivate supervision of the answer's source location and its endpoint inside the writer.

The fit used 20,271 examples, 1,014 updates and 292,570 answer positions. Optimization took 61.7 seconds with 4.01 GiB peak allocated on one RTX 4090 D. The 86.6-million-parameter writer inherited a pretrained Chinese causal backbone and earlier visual training. It emits from a physical bitmap inventory and glyphs available on the page. From-scratch language learning, unrestricted glyph synthesis and a matched 1,000-fold efficiency gain remain open.

[Measured outputs and unmodified pixels](evidence/mixed_visual_answers_20261124) · [Offline model instructions](../models/mixed_visual_answers/README.md) · [Training protocol](mixed_visual_answer_protocol.md) · [Formal question-switch condition](../formal/span_list/PicturedQuestionSwitch.lean)
