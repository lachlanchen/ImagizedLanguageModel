# What a visual writer can add to a Chinese sentence

Given 嗣复曰： 洵直无邪，臣知之。, the correct modern prefix 杨嗣复说 ：窦洵直, and a boundary before 无, the visual writer drew 没有 and stopped. Neither output glyph appeared in the classical source. It also failed in the opposite direction. After 派使 in another passage, it supplied 者 where the recorded insertion was 者前. Both examples use actual output pixels from held-out chapters and reproduce on CPU. [Figure 52](../publication/ilm-nature/figures/Figure_52_visual_phrase.png)

## What the model received

The classical sentence, modern prefix and a constant translation instruction were presented as glyph images. The experiment supplied the alignment-derived source boundary and the fact that an insertion was needed. At each new cell the writer predicted twelve groups of four physical ink measurements. Its inherited neural renderer drew the complete cell. The measured pixels of that emitted image updated the recurrent state and a binary head estimated whether the phrase had ended. The 32-glyph limit applied to every request. Target length and subsequent modern text were withheld. A separate model received the modern prefix and one identical constant source at every training and test site. [Training and acceptance protocol](visual_phrase_protocol.md)

The 48,000 original training pairs and 16,384 additional NiuTrans pairs contained 296,837 complete aligned insertions and 100,726 distinct insertion types. Each arm saw the same 627,438 insertion glyphs per epoch, for three epochs. The fits updated 584,241 parameters of a 2,000,566-parameter network in 2,016 steps apiece. Source-conditioned and prefix-only training took 19.9 and 18.3 seconds, and peaked at 368 and 317 MiB of allocated CUDA memory, rounded upward, on one RTX 4090 D. These are incremental fit costs. The first training launch and first diagnostic launch exited at the process resource guard before model work began. Fresh child processes ran with the original limits. The study includes shared inherited weights and preparation costs outside the fit times. [Conditioned fit](evidence/visual_phrase_20261118/source_fit.json) · [Prefix-only fit](evidence/visual_phrase_20261118/prefix_fit.json)

## Where the source helped

| Exact stopped whole phrase | Discovery, 5,376 sites | Replication, 5,413 sites |
|---|---:|---:|
| Prior writer, required to begin drawing | 220 | 197 |
| New writer trained on modern prefix alone | 157 | 150 |
| New writer trained on source and modern prefix | 337 | 268 |
| Local whole-phrase counts | 868 | 808 |

Conditioning on the classical source improved the exact rate over prefix-only training on both chapter cohorts. The 95% chapter-bootstrap intervals for the gains were 2.70–4.05 and 1.68–2.70 percentage points. Local counts led the conditioned writer by 8.80–10.95 and 8.84–11.15 points. The conditioned writer alone solved 218 sites missed by counts. Counts alone solved 1,289. Two discovery and seven replication requests had no fully renderable sentence pair and could supply no insertion sites. The comparison uses one recorded modern translation per source and a deterministic edit alignment. [Discovery](evidence/visual_phrase_20261118/discovery_report.json) · [Replication](evidence/visual_phrase_20261118/replication_report.json)

All target glyphs were absent from the classical source in 582 of the 605 conditioned successes. Only 20/5,213 multi-glyph targets were recovered and no unseen training-insertion type was recovered across 3,281 such sites. The conditioned writer stopped on 9,922/10,789 requests. In 293 natural exchanges, likelihood favored the recorded phrase assignment 219 times and 163 times with reversed classical sources and mirrored boundaries. The prefix-only fit favored 126 correct assignments in both orientations. Independently producing both complete phrases succeeded in four pairs. These distinctions matter for an application that must write an answer from a prompt.

## Why a wider first-glyph search cannot close the gap

At 9,516 of 10,184 incorrect sites the first emitted glyph already differed from the reference. Given the correct landmarks, the same frozen renderer could reproduce 10,783 complete references exactly. At 330 sites the output stopped on an initial part of the reference. At another 170 it began with the complete reference and continued writing. Each split's most frequent physical output occurred more than 2,400 times. The bitmap archives were independently indexed by scanning the original edit actions and recounted. [Physical recount](evidence/visual_phrase_20261118/audit.json)

At the actual source and modern prefix, the model gave the emitted first-glyph code greater probability than the reference code on 9,453 of 9,516 first-glyph errors. The median log-probability gap favoring the emitted code was 4.81 nats in discovery and 4.77 in replication. All erroneous pairs had different measured codes. Exact maximization of the existing first-glyph density can therefore rescue at most 63 of those references. This is a bound on recovering that particular reference by optimizing the existing first-step objective. It does not bound alternative training or sequence objectives. [Probe protocol](visual_phrase_first_glyph_protocol.md) · [Discovery scores](evidence/visual_phrase_20261118/discovery_onsets.json) · [Replication scores](evidence/visual_phrase_20261118/replication_onsets.json)

## Reproduction and decision

The new writer missed the predeclared count-control, unseen-phrase and stop criteria. Its research weights and CPU inference command accompany the original release. Full-sentence testing with model-selected sites awaits a qualified phrase generator. [Run the example](../models/visual_edit_zh/README.md#draw-a-missing-modern-phrase-at-a-supplied-boundary)

From the repository root, after preparing the parent editor and frozen alignment rosters, run the following commands with the original NiuTrans data available. Raw outputs, scores and weights are stored in `artifacts/visual_phrase_writer_v1/`. The report command writes the figure sources and recount under the run directory. The scoring command writes each cohort’s ordered score array and summary beside its generated images.

```sh
export PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python scripts/train_visual_phrases.py
python scripts/evaluate_visual_phrases.py
python scripts/report_visual_phrases.py
python scripts/score_visual_phrase_onsets.py
```

The next comparison should train a predictor of the first glyph's physical measurements using local source and modern context at several lengths. A rare visual context can borrow evidence from a shorter context, and a frequent one can learn its own distribution. Compare a smoothed physical-pattern model with this conditioned writer and local whole-phrase counts at the same sites. It must draw from successive physical patterns without a catalogue of complete glyphs or answers. Keep the rest of the phrase writer fixed in the first comparison, score complete phrases as well as first cells, and account for the storage and runtime of every conditional count. A first-cell improvement alone would not qualify the system for sentence translation.
