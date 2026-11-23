# Reproducing the pictured fact experiment

In the repository, the trained field head and frame table are in `artifacts/role_binding_v1/genuine/`, the free writer in `artifacts/role_binding_v1/writer/`, and the full local distribution in `artifacts/role_binding_v1/visual-fact-reader-writer.zip`. The preceding visual reader, image writer, prepared proposition boards, DuIE archive and CMRC/DRCD datasets are the dependencies of a fresh fit. The archive alone is sufficient for inference with the Python packages listed above.

Run the proof from the repository root with Lean 4.12.0. Then prepare the corpus and fit and evaluate the models in order in the project Python environment.

```sh
lean formal/role_binding/RoleBinding.lean
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
python scripts/prepare_visual_facts.py
python scripts/train_visual_facts.py
python scripts/evaluate_visual_facts.py
python scripts/train_fact_realizer.py
python scripts/evaluate_fact_realizer.py
python scripts/probe_visual_fact_transfer.py
python scripts/package_visual_facts.py
```

The training scripts write new checkpoint directories and preserve existing fits. Use a fresh experiment checkout with the required input artifacts for a reproduction. The inference CLI accepts separate model and reader directories. [Mathematical assumptions, results and controls](../../references/role_binding_result.md)
