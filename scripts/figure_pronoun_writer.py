"""Assemble native chart counts and crop only the exterior blank input cells."""

from PIL import Image
from scripts.prepare_asked_gloss import read, dump
from scripts.prepare_pronoun_writer import PUBLIC


def main():
    data = read(PUBLIC / "result.json")
    selected = [
        "held_book-object_person-correct",
        "held_book-motion-correct",
        "held_book-motion-miss",
    ]
    by_image = {
        r["image"]: r for r in read(PUBLIC / "held_replies.json") if "image" in r
    }
    for name in selected:
        r = by_image[name]
        s = r["window"]
        left = len(s) - len(s.lstrip(" "))
        right = len(s.rstrip(" "))
        with Image.open(PUBLIC / "samples" / (name + ".png")) as im:
            assert im.width == 32 * len(s)
            im.crop((32 * left, 0, 32 * right, im.height)).save(
                PUBLIC / (name + ".png")
            )
    keys = ("held_book", "held_section")
    series = [
        [data["pairs"][s]["correct"][arm] for s in keys]
        for arm in ("parent", "aligned", "exchanged")
    ]
    table = [["Passages", "Original", "Aligned", "Exchanged", "Measure"]]
    for subset, title in [("previous", "Earlier 539"), ("fresh", "New 29")]:
        for key, label in [
            ("classroom", "Four questions"),
            ("writable", "Locations given"),
        ]:
            table.append(
                [title]
                + [
                    str(data["transfer"][arm][subset][key])
                    for arm in ("parent", "aligned", "exchanged")
                ]
                + [label]
            )
    dump(PUBLIC / "figure.json", dict(series=series, table=table))


if __name__ == "__main__":
    main()
