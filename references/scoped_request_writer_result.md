# When the same local ink asks different questions

Four questions can have different answers while sharing every centered nine-glyph neighborhood, including multiplicity and boundary context. On such questions the previous local writer emitted one identical image for all four requests in **every group**. A writer with relative-position attention answered all four correctly in **1,060 of 1,136 groups** under familiar wording. Changing two clause cues and the wording around the fields reduced the count to zero. [Figure 81, with the original images](../publication/ilm-nature/figures/Figure_81_scoped_request_writer.png)

| Writer | All four, familiar cues / 1,136 | All four, new cues / 1,136 | Independent English / 96 | Independent Chinese / 96 | DRCD pairs / 1,042 | DuReader / 571 |
|---|---:|---:|---:|---:|---:|---:|
| Local | 0 | 0 | 48 | 61 | 13 | 14 |
| Global attention, no positions | 0 | 0 | 53 | 59 | 10 | 14 |
| Global relative attention | 1,060 | 0 | 60 | 56 | 12 | 14 |

## Why the local writer produced one answer

Each question cell influences two local convolutions with radii one and three. The resulting feature at a position depends on nine glyph cells. The model pools question features and attends over their set. Its copy bias can inspect up to three preceding cells, which the same nine-cell neighborhood already specifies. For two uniformly padded questions sharing the same multiset of these windows and the same passage, each emitted bitmap receives the same total probability given any common output history, in exact arithmetic. Greedy decoding therefore gives the same complete image. The measured outputs reproduced that prediction across 2,272 groups covering two constructions. [Architecture](../ilm/visual_lm/grounded_glyph_decoder.py) · [Physical neighborhood verification](evidence/english_request_writer_20261119/local_window_equivalence.json)

Four direction- and distance-sensitive attention heads let the writer relate a printed field to a more distant clause. The unordered control used the same parameters, supervision, updates and global value aggregation. On the familiar cues, the ordered writer gave 4,396 of 4,544 exact responses and all 16 answers for 264 of the 284 tested headwords. Its four-question gain over unordered attention was 93.31 percentage points (95% card-clustered interval 90.49–95.95). The ordered and unordered writers each had 694,152 parameters, 181,761 of them trainable. Their fits took 82.9 and 78.8 seconds on one RTX 4090 D and each stayed below 0.44 GiB peak allocation. These incremental costs exclude their shared checkpoint training. [Model](../ilm/visual_lm/scoped_question.py) · [Registered comparison](scoped_request_writer_protocol.md) · [Audited groups and intervals](evidence/scoped_request_writer_20261119/audit.json)

## What failed to transfer

With the unfamiliar construction, 1,813 of 4,544 outputs exactly expressed both the unwanted fact and the unwanted script for the correct word. We exchanged the field wrappers, known-information cue and requested-information cue separately and kept all 284 words, four layouts, historical rasters and weights fixed.

| Wording change from the familiar construction | Complete four-question groups / 1,136 |
|---|---:|
| None | 1,060 |
| Only the text surrounding each field and its conjunction | 784 |
| Only “Please show” to “Could you provide” | 246 |
| Only “I already have” to “I can supply” | 16 |
| Both clause cues, original field wrappers | 0 |
| Both clause cues and field wrappers | 0 |

The complete eight-way comparison also includes the two remaining combinations. The architecture could use long-range order but depended heavily on particular expressions of what was already known. Independent English increased to 60/96 and independent Chinese fell to 56/96. No writer recovered a canonical DuReader answer among the 300 requiring an absent content glyph. Flexible cue interpretation and general composition remain open. [All eight conditions and paired effects](evidence/scoped_request_writer_20261119/cue_exchange.json)

## Run the writer

With the local fitted checkpoint, entry index and historical raster cache, the following command answers a typed English question by imaging it before inference.

```bash
python -m scripts.ask_visual_dictionary --char 言 --stages oracle bronze \
  --checkpoint-dir artifacts/scoped_request_writer_v1/ordered \
  --question 'What was the original meaning of 言? Show the bronze form.' \
  --output /tmp/ilm-yan-english
```

It wrote the gloss *talk* and copied the attested bronze bitmap B02975. Re-entering the same 57-cell question as a PNG reproduced the identical answer bytes. The headword and gloss had appeared in earlier dictionary training, with other forms. This illustrates an input and output supported by the local model. [Receipt](evidence/scoped_request_writer_20261119/yan_cli.json)
