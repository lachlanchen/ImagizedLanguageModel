#!/usr/bin/env python3
"""Build ink-mass/full-page redlines against the preserved 41-slide baseline."""

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASELINE = ROOT / "artifacts/ink_mass_publication_20261108/baseline"
QA = ROOT / "artifacts/ink_mass_publication_20261108/redlines"
TOOL = ROOT / "artifacts/external/latexdiff-1.4.0/latexdiff"
TOOL_SHA = "bbf7567f48ae6d52887ebce0a390eac0718508b5e1e3ee158bffd4280dca8723"
INPUT = re.compile(r"\\input\{([^{}]+)\}")
REVISION = "b5501de2e925ec86c25f726b1261bb7f65da5825"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def prepare_baseline():
    """Recreate the immutable manifest from Git on a fresh checkout."""
    names = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", REVISION,
        "--", "publication/ilm-nature"], cwd=ROOT, text=True).splitlines()
    files = {}
    for name in names:
        relative = str(Path(name).relative_to("publication/ilm-nature"))
        if (relative.startswith("figures/") or relative == "nature_reading.sty"
                or (len(Path(relative).parts) == 1 and relative.endswith(".tex"))):
            files[relative] = sha(subprocess.check_output(["git", "show", REVISION + ":" + name], cwd=ROOT))
    manifest = dict(source_commit=REVISION, files_sha256=files)
    path = BASELINE / "baseline_manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("baseline manifest differs from immutable Git inputs")
    else:
        BASELINE.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    manifest = prepare_baseline()
    revision = manifest["source_commit"]
    if sha(TOOL.read_bytes()) != TOOL_SHA:
        raise ValueError("pinned official latexdiff 1.4.0 required")
    for name, expected in manifest["files_sha256"].items():
        raw = (BASELINE/name).read_bytes() if (BASELINE/name).exists() else subprocess.check_output(["git", "show", revision+":"+str((HERE/name).relative_to(ROOT))], cwd=ROOT)
        if sha(raw) != expected:
            raise ValueError("preserved baseline changed: "+name)

    def git_bytes(name):
        return subprocess.check_output(["git", "show", revision+":"+
            str((HERE/name).relative_to(ROOT))], cwd=ROOT)

    for name in manifest["files_sha256"]:
        if name.startswith("figures/") or name == "nature_reading.sty":
            if git_bytes(name) != (HERE/name).read_bytes():
                raise ValueError("previous figure/style changed: "+name)

    def expand(name, reader, hashes, ancestors=()):
        if name in ancestors or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("cyclic or external TeX input")
        raw = reader(name)
        hashes[name] = sha(raw)
        def child(match):
            path = str(Path(name).parent/match[1])
            if not Path(path).suffix:
                path += ".tex"
            return expand(path, reader, hashes, (*ancestors,name))
        return INPUT.sub(child, raw.decode())

    # Put title and abstract in the document body before comparison. Otherwise
    # latexdiff treats macro-definition changes in the preamble as unmarked.
    def inline_frontmatter(source):
        # Normalize a layout-only bibliography penalty in both inputs. Markup
        # inside a TeX integer assignment is invalid and has no scientific role.
        source = re.sub(r"\\interlinepenalty=\d+", lambda _: r"\interlinepenalty=5000", source)
        source = source.replace(r"\balance", r"\relax")
        for macro in ("PaperTitle", "PaperSummary"):
            token = "\\newcommand{\\"+macro+"}{"
            start = source.find(token)
            if start < 0:
                continue
            begin = start+len(token)
            depth, end = 1, begin
            while depth:
                if source[end] == "{" and source[end-1] != "\\":
                    depth += 1
                elif source[end] == "}" and source[end-1] != "\\":
                    depth -= 1
                end += 1
            body = source[begin:end-1]
            source = source[:start]+source[end:]
            source = source.replace("\\"+macro, "{"+body+"}")
        return source

    results = {}
    for document, output in (("manuscript_review.tex","redlined_manuscript"),
            ("supplementary_information.tex","redlined_supplementary_information")):
        folder = QA/output
        folder.mkdir(parents=True, exist_ok=True)
        old_hashes, new_hashes = {}, {}
        old = inline_frontmatter(expand(document,git_bytes,old_hashes))
        new = inline_frontmatter(expand(document,lambda n:(HERE/n).read_bytes(),new_hashes))
        for name, value in old_hashes.items():
            if manifest["files_sha256"].get(name) != value:
                raise ValueError("expanded Git input is not in baseline: "+name)
        (folder/"old.tex").write_text(old)
        (folder/"new.tex").write_text(new)
        marked = subprocess.run(["perl",str(TOOL),"--encoding=utf8",
            "--type=CFONT","--subtype=SAFE","--floattype=IDENTICAL",
            "--math-markup=0","old.tex","new.tex"],cwd=folder,
            capture_output=True,text=True,check=True)
        (folder/"latexdiff.log").write_text(marked.stderr)
        notice = (r"\noindent{\color{blue}\footnotesize Change-marked copy: "
            r"additions relative to the b5501de 41-slide writing baseline are blue.}"
            r"\par\medskip"+"\n")
        source = marked.stdout.replace(r"\begin{document}",
            r"\begin{document}"+"\n"+notice,1)
        source = "\n".join(line.rstrip() for line in source.splitlines()) + "\n"
        target = HERE/(output+".tex")
        target.write_text(source)
        env = dict(os.environ,TEXINPUTS=str(HERE)+os.pathsep+os.environ.get("TEXINPUTS",""))
        build = subprocess.run(["latexmk","-pdf","-interaction=nonstopmode",
            "-halt-on-error","-outdir="+str(folder),str(target)],cwd=HERE,
            env=env,capture_output=True,text=True)
        (folder/"build.log").write_text(build.stdout+build.stderr)
        if build.returncode:
            raise RuntimeError("redline compilation failed: "+str(folder))
        log = (folder/(output+".log")).read_text()
        if any(x in log for x in ("Overfull","undefined references","undefined citations","LaTeX Error")):
            raise ValueError("redline layout/reference error: "+str(folder))
        shutil.copy2(folder/(output+".pdf"),HERE/(output+".pdf"))
        if source.count(r"\DIFadd{") == 0:
            raise ValueError("no visible added text")
        results[output] = dict(old_inputs_sha256=old_hashes,new_inputs_sha256=new_hashes,
            added_text_markers=source.count(r"\DIFadd{"),
            tex_sha256=sha(target.read_bytes()),pdf_sha256=sha((HERE/(output+".pdf")).read_bytes()))
    receipt = dict(baseline_commit=revision,baseline_manifest_sha256=sha((BASELINE/"baseline_manifest.json").read_bytes()),
        latexdiff_sha256=TOOL_SHA,markup="CFONT / SAFE / IDENTICAL floats; math-markup=0",
        abstract_inlined_before_comparison=True,bibliography_interline_penalty_normalized=5000,
        redline_final_columns_unbalanced=True,
        trailing_line_whitespace_normalized=True,
        article_redline_layout="matching line-numbered single-column review source",
        baseline_figures_and_style_unchanged=True,
        documents=results,builder_sha256=sha(Path(__file__).read_bytes()))
    (HERE/"evidence/ink_mass_writing_redline_build.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print("Built both redlines against "+revision)


if __name__ == "__main__":
    main()
