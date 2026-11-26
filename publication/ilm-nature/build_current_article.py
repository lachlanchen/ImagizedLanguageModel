"""Build both public PDF aliases from the one current manuscript and refresh the figure audit."""
from pathlib import Path
from tempfile import TemporaryDirectory
import os
import shutil
import subprocess
import sys

PAPER = Path(__file__).resolve().parent


def main():
    with TemporaryDirectory(prefix="ilm-article-") as tmp:
        subprocess.run(["latexmk","-pdf","-halt-on-error","-interaction=nonstopmode",
                        "-outdir="+tmp,"manuscript.tex"],cwd=PAPER,check=True)
        for filename in ("manuscript.pdf","visual_language_paper.pdf"):
            dest=PAPER/filename
            staging=dest.with_suffix(".pdf.tmp")
            shutil.copyfile(Path(tmp)/"manuscript.pdf",staging)
            os.replace(staging,dest)
    subprocess.run([sys.executable,str(PAPER/"verify_visual_output_provenance.py")],cwd=PAPER,check=True)


if __name__=="__main__":
    main()
