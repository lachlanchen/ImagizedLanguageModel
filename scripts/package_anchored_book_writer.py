"""Bundle only changed writer tensors. Reuse the verified pictured reader and parent."""

import json
from zipfile import ZIP_DEFLATED, ZipFile
from safetensors.torch import load_file, save_file
import torch
from scripts.train_anchored_book_clauses import PUBLIC, WORK, PARENT, ROOT
from scripts.train_causal_glyph_writer import record, sha


def write_update(base_folder, trained_folder, base, trained, folder):
    changed = {}
    for name, target in trained.items():
        assert (
            sha(trained_folder / name) == target
            and sha(base_folder / name) == base[name]
        )
        if target == base[name]:
            continue
        old, new = [load_file(str(d / name)) for d in (base_folder, trained_folder)]
        assert old.keys() == new.keys()
        for key in new:
            if not torch.equal(old[key], new[key]):
                changed[name + "/" + key] = new[key]
    delta = folder / "frame-update.safetensors"
    save_file(changed, str(delta))
    record(
        folder / "frame_manifest.json",
        dict(base=base, trained=trained, update_sha=sha(delta)),
    )
    return changed, delta


def package():
    torch.set_num_threads(2)
    folder = WORK / "release"
    folder.mkdir(exist_ok=True)
    trained = json.loads((PUBLIC / "fit.json").read_text())["weights"]
    base = json.loads((PARENT.parent / "page_fit.json").read_text())["weights"]
    changed, delta = write_update(PARENT, WORK / "page", base, trained, folder)
    paths = {p.name: p for p in (delta, folder / "frame_manifest.json")}
    paths["FRAME_README.md"] = ROOT / "models/anchored_book_writer/README.md"
    for n in (
        "scripts/install_frame_update.py",
        "scripts/write_ordered_pictured_reply.py",
        "ilm/visual_lm/anchored_clause_writer.py",
        "ilm/visual_lm/ordered_evidence_writer.py",
    ):
        paths[n] = ROOT / n
    for p in (PUBLIC / "examples").glob("*.png"):
        paths["frame_examples/" + p.name] = p
    from scripts.prepare_ordered_book_replies import PUBLIC as DIRECT

    paths["frame_examples/separator.png"] = DIRECT / "examples/separator.png"
    archive = folder / "anchored-book-update.zip"
    with ZipFile(archive, "x", compression=ZIP_DEFLATED, compresslevel=2) as z:
        for name, p in sorted(paths.items()):
            z.write(p, name)
    record(
        PUBLIC / "release.json",
        dict(
            archive=archive.name,
            sha256=sha(archive),
            bytes=archive.stat().st_size,
            changed_tensors=len(changed),
            files={n: sha(p) for n, p in paths.items()},
        ),
    )


if __name__ == "__main__":
    package()
