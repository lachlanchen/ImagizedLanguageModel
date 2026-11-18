"""Join frozen contextual judgments and print the sampled real source/answer images."""
import argparse
import shutil
from pathlib import Path

from scripts.evaluate_sentence_choice import WORK, WRITERS, load, roster
from scripts.read_local_ink_phrases import score
from scripts.render_conditional_ink_gallery import render
from scripts.report_sentence_choice import EVIDENCE


def gallery():
    reading = WORK/'reading'
    for name in ('binding.json','result.json','ratings.txt','roster.json'):
        shutil.copyfile(reading/name,EVIDENCE/('reading_'+name))
    split=load(reading/'roster.json')['split']
    render(path=EVIDENCE/'reading_gallery.pdf',reading=reading,cohort=roster(split),directory=WORK/'evaluation'/split,
           versions=[('assignment','Paired assignment'),('likelihood','Likelihood'),('frozen','Frozen writer'),
                     ('local','Whole-phrase counts')])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    form=parser.add_mutually_exclusive_group(required=True)
    form.add_argument('--score',type=Path)
    form.add_argument('--gallery',action='store_true')
    args=parser.parse_args()
    if args.score:
        score(args.score,folder=WORK/'reading',arms=WRITERS)
    else:
        gallery()
