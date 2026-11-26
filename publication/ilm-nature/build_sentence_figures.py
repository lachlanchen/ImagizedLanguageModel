"""Rebuild the six editable manuscript panels in an isolated staging directory."""
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

PAPER=Path(__file__).resolve().parent
ROOT=PAPER.parents[1]
sys.path.insert(0,str(ROOT))
from build_full_context_comparison_article import environment, NODE, SKILL


def main():
    with TemporaryDirectory(prefix="ilm-draft-figures-",dir=ROOT/"artifacts") as work:
        stage=Path(work)
        (stage/"node_modules").symlink_to(NODE/"node_modules")
        source=shutil.copy2(PAPER/"pictured_path_figures.mjs",stage)
        subprocess.run([str(NODE/"bin/node"),source,str(ROOT),str(stage),str(SKILL),sys.executable,"drafts"],
                       env=environment(stage),check=True)
        shutil.copy2(stage/"output/Sentence_reconstruction.pptx",PAPER/"figures")
        shutil.copy2(stage/"presentation_validation.json",ROOT/"artifacts/emitted_history_v1")


if __name__=="__main__":
    main()
