# Reading an edited page

This 87.1-million-parameter research writer accepts one Chinese passage and two questions, each supplied as an image strip of 32-by-32-pixel cells. Questions may be Chinese or English. It writes a single Chinese response, with a limit of 48 cells. Its learned pointer can copy unfamiliar marks from the supplied passage. Its other output choices come from a learned distribution over a stored print alphabet.

From a trained checkout, run the following command with a directory containing the four example input images.

```bash
examples=references/evidence/role_change_writer_20261124/examples
PYTHONPATH=. python -m scripts.write_ordered_pictured_reply --mode page \
  --page "$examples/page0.png" --questions "$examples/q0.png" "$examples/q1.png" \
  --output reply.png
```

The command defaults to CPU and takes an optional `--device cuda`. Change the page to `page1.png` to ask the identical questions about the second version. For the packaged release, extract `bilingual-novel-writer.zip` and `role-change-update.zip` into separate directories. From the extracted update directory, run the installer and the supplied page runner. The example assumes the bilingual archive was extracted in a sibling directory named `parent`.

```bash
python scripts/install_frame_update.py --base ../parent/writer --update . --output role-writer
PYTHONPATH=. python -m scripts.write_ordered_pictured_reply --mode page \
  --writer role-writer --separator examples/separator.png \
  --page examples/page0.png --questions examples/q0.png examples/q1.png \
  --output reply.png
```

The installer verifies all restored weight files. The returned stop flag reports whether the writer ended its answer. See the paired-page experiment for role accuracy and complete-statement results. [Evaluation](../../references/role_change_writer_result.md)
