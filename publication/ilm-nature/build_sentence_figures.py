"""Rebuild the six editable manuscript panels in an isolated staging directory."""
import argparse
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
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--actions",action="store_true")
    cli.add_argument("--support",action="store_true")
    args=cli.parse_args()
    actions=args.actions or args.support
    edition="actions" if actions else "drafts"
    stem="Independent_optical_actions" if actions else "Sentence_reconstruction"
    corpus="independent_optical_actions_v1" if actions else "emitted_history_v1"
    if args.support:
        edition,stem,corpus="support","Source_supported_writing","source_supported_writing_v1"
    with TemporaryDirectory(prefix="ilm-draft-figures-",dir=ROOT/"artifacts") as work:
        stage=Path(work)
        (stage/"node_modules").symlink_to(NODE/"node_modules")
        source=shutil.copy2(PAPER/"pictured_path_figures.mjs",stage)
        subprocess.run([str(NODE/"bin/node"),source,str(ROOT),str(stage),str(SKILL),sys.executable,edition],
                       env=environment(stage),check=True)
        shutil.copy2(stage/"output"/(stem+".pptx"),PAPER/"figures")
        shutil.copy2(stage/"presentation_validation.json",ROOT/"artifacts"/corpus)


if __name__=="__main__":
    main()
