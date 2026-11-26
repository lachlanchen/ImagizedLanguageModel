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
    cli.add_argument("--restoration",action="store_true")
    cli.add_argument("--roles",action="store_true")
    cli.add_argument("--referents",action="store_true")
    cli.add_argument("--full-image",action="store_true")
    cli.add_argument("--alignment",action="store_true")
    cli.add_argument("--phrases",action="store_true")
    cli.add_argument("--editor",action="store_true")
    cli.add_argument("--sequence-preference",action="store_true")
    args=cli.parse_args()
    actions=args.actions or args.support
    edition="actions" if actions else "drafts"
    stem="Independent_optical_actions" if actions else "Sentence_reconstruction"
    corpus="independent_optical_actions_v1" if actions else "emitted_history_v1"
    if args.support:
        edition,stem,corpus="support","Source_supported_writing","source_supported_writing_v1"
    if args.restoration:
        edition,stem,corpus="restoration","Visual_strip_restoration","visual_strip_restoration_v1"
    if args.roles:
        edition,stem,corpus="roles","Visual_role_reading","natural_pictured_roles_v1"
    if args.referents:
        edition,stem,corpus="referents","Visual_referent_answers","visual_referents_v1"
    if args.full_image:
        edition,stem,corpus="full-image","Full_image_role_writing","full_image_roles_v1"
    if args.alignment:
        edition,stem,corpus="alignment","Ordered_glyph_alignment","ordered_alignment_v1"
    if args.phrases:
        edition,stem,corpus="phrases","Covered_glyph_phrases","covered_phrases_v1"
    if args.editor:
        edition,stem,corpus="editor","Glyph_draft_editor","glyph_draft_editor_v1"
    if args.sequence_preference:
        edition,stem,corpus="sequence-preference","Visual_sequence_preference","visual_sequence_preference_v1"
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
