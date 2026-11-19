# Read and write a participant as a glyph image

This research checkpoint accepts a pictured passage and a pictured question. It can reorder a Chinese clause so that the participant's name appears last. A name can be an inscription outside the 7,715 shapes in the model's writing memory. The writer can copy that image from the source and continue in modern Chinese. The two examples use genuine bronze forms of Qin and Chu in invented statements about a bronze mirror.

Unzip the bundle, install the dependencies in `requirements.txt` if needed, and run this command from the unpacked directory. The default device is CPU.

```bash
python read.py --passage-image examples/bronze1_premise.png \
  --question-image examples/bronze0_query.png --output answer
```

Open `answer/answer.png`. For these inputs the correctly emitted sentence reads 没有铜镜的是秦。, with the original bronze Qin form. The other passage in `examples/bronze0_premise.png` provides a counterexample: the writer puts the wrong participant after the correct affirmative predicate. For wrapped pages supply `--passage-cells` or `--question-cells` to exclude final grid padding. Input cells are 32 by 32 pixels in row-major order.

Typed inputs use the Noto Sans CJK Regular collection, face 2. Set `--font` to its path if it is installed elsewhere. Any inscription can instead be supplied directly in a PNG cell.

```bash
python read.py --passage-text '秦有铜镜，楚没有铜镜。' \
  --question-text '请把秦的情况写成一句话，把姓名放在“的是”后面。' \
  --output printed
```

The archive includes the complete recurrent inference model, original shape memory, example PNGs and source. A PyTorch installation with CUDA also supports `--device cuda:0`. No network access or language-model service is used during inference. Each command writes the emitted image, packed one-bit cells and a receipt with the stop decision. Choose a new output directory for each call.

Training included task adaptation on OCNLI, which its authors distribute under CC BY-NC 2.0. The released results distinguish constructed role-writing tests from human-authored inference. The writer can lose the role on unfamiliar questions, and its ability to write open-ended answers remains limited. [OCNLI dataset](https://github.com/CLUEbenchmark/OCNLI)
