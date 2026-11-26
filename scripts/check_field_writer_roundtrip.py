"""Compare free translation ink to the actual input picture from the reverse request."""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image

from scripts.replay_field_writer_archive import ARCHIVE, EVIDENCE, MODES


def cells(blob, count):
    with Image.open(BytesIO(blob)) as image:
        bitmap = np.asarray(image.convert("L")) < 128
    height, width = bitmap.shape
    if height % 32 or width % 32 or count > (height//32)*(width//32):
        raise ValueError("Invalid published grid geometry")
    return bitmap.reshape(height//32,32,width//32,32).transpose(0,2,1,3).reshape(-1,32,32)[:count]


def main():
    digest = sha256(ARCHIVE.read_bytes()).hexdigest()
    if json.loads((EVIDENCE/"evaluation.json").read_text())["archive"] != digest:
        raise ValueError("Archive fingerprint changed")
    with ZipFile(ARCHIVE) as z:
        rows = json.loads(z.read("request-outcomes.json"))
        matched = {mode:[] for mode in MODES}
        for i, row in enumerate(rows):
            peer = rows[i ^ 1]
            if (row["reference"],row["source"],row["work"]) != (peer["source"],peer["reference"],peer["work"]):
                raise ValueError("Opposite requests do not share the same passage pair")
            reference = cells(z.read(f"{i^1:04d}/input.png"), len(row["reference"]))
            for mode in MODES:
                answer = row["answers"][mode]
                raw = z.read(f"{i:04d}/{mode}.png")
                if sha256(raw).hexdigest() != answer["png"]:
                    raise ValueError("Answer image fingerprint changed")
                if answer["stop"] and np.array_equal(cells(raw, answer["cells"]), reference):
                    matched[mode].append(i)
    evidence = dict(requests=len(rows), archive=digest, exact_stopped_ids=matched,
                    exact_both_directions={k:sum(i%2==0 and i+1 in ids for i in ids) for k,ids in matched.items()},
                    program=sha256(Path(__file__).read_bytes()).hexdigest())
    (EVIDENCE/"paired_reference_ink.json").write_text(json.dumps(evidence, indent=2)+"\n")
    print(matched)


if __name__ == "__main__":
    main()
