"""Construct a self-contained cross-typeface writer and check its GPU and CPU answers."""

import json
import shutil
from scripts.package_mixed_visual_answers import package
from scripts.train_typeface_writer import WORK,PUBLIC,ROOT


def release():
    result=json.loads((PUBLIC/"result.json").read_text())
    tag="gain" if "gain" in result["examples"] else ("success" if "success" in result["examples"] else "error")
    examples=PUBLIC/"examples"
    for target,source in (("human_page","page"),("human_question","question"),("human_answer","mapped")):
        shutil.copyfile(examples/(tag+"_"+source+".png"),examples/(target+".png"))
    package(WORK/"mapped",PUBLIC,ROOT/"models/typeface_writer/README.md","typeface-writer.zip")


if __name__=="__main__":
    release()
