"""Recover the displayed input and answer cells from evaluated PNGs without re-typesetting."""

import io
import zipfile
from PIL import Image
from scripts.prepare_prose_fields import PUBLIC
from scripts.write_prose_fields import load_png, trim
from scripts.write_chinese_card_reply import display


def main():
    with zipfile.ZipFile(PUBLIC / "entity-examples.zip") as z:

        def page(group, i):
            return load_png(io.BytesIO(z.read(f"{group}/{i:04}-page.png")))

        def answer(group, i, name):
            with Image.open(io.BytesIO(z.read(f"{group}/{i:04}-answer.png"))) as im:
                im.save(PUBLIC / (name + ".png"))

        a = page("native-multi", 0)
        display(trim(a[0]), 40).save(PUBLIC / "restaurant-question.png")
        display(trim(a[3:].reshape(-1, 128)), 36).save(PUBLIC / "restaurant-prose.png")
        for i in (0, 1):
            p = page("native-multi", i)
            display(trim(p[2]), 35).save(PUBLIC / ("name-" + str(i) + ".png"))
            answer("native-multi", i, "reply-" + str(i))
        p = page("authored-multi", 5)
        display(trim(p[3:].reshape(-1, 128)), 18).save(PUBLIC / "park-prose.png")
        answer("authored-multi", 5, "park-reply")


if __name__ == "__main__":
    main()
