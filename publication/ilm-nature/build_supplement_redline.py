#!/usr/bin/env python3
"""Compile a change-marked supplement from its immutable Git source baseline."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TOOL_SHA = "bbf7567f48ae6d52887ebce0a390eac0718508b5e1e3ee158bffd4280dca8723"
INPUT = re.compile(r"\\input\{([^{}]+)\}")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latexdiff", type=Path, default=ROOT/"artifacts/external/latexdiff-1.4.0/latexdiff")
    parser.add_argument("--qa-dir", type=Path, default=ROOT/"artifacts/v62_supp_redline_qa_20261105")
    args = parser.parse_args()
    args.qa_dir.mkdir(parents=True, exist_ok=True)
    tool = args.latexdiff.resolve()
    if digest(tool.read_bytes()) != TOOL_SHA:
        raise ValueError("pinned official latexdiff 1.4.0 source required")
    manifest = json.loads((HERE/"baselines/v62-primary/baseline_manifest.json").read_text())
    revision = manifest["source_commit"]
    def git_bytes(name):
        return subprocess.check_output(["git", "show", revision+":"+str((HERE/name).relative_to(ROOT))], cwd=ROOT)
    def current_bytes(name):
        return (HERE/name).read_bytes()
    def expand(name, reader, hashes, ancestors=()):
        name = str(Path(name))
        if name in ancestors or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("cyclic or external TeX input")
        raw = reader(name)
        hashes[name] = digest(raw)
        def included(match):
            child = str(Path(name).parent/match[1])
            if not Path(child).suffix:
                child += ".tex"
            return expand(child, reader, hashes, (*ancestors, name))
        return INPUT.sub(included, raw.decode("utf-8"))
    old_hashes, new_hashes = {}, {}
    document = "supplementary_information.tex"
    old = expand(document, git_bytes, old_hashes)
    new = expand(document, current_bytes, new_hashes)
    if old_hashes[document] != manifest["files_sha256"][document]:
        raise ValueError("Git source differs from the named preserved baseline")
    if git_bytes("nature_reading.sty") != current_bytes("nature_reading.sty"):
        raise ValueError("shared style changed; requires an explicit separate revision plan")
    figure_names = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", revision,
        "--", "publication/ilm-nature/figures"], cwd=ROOT, text=True).splitlines()
    for name in figure_names:
        relative = str(Path(name).relative_to("publication/ilm-nature"))
        if git_bytes(relative) != current_bytes(relative):
            raise ValueError("baseline figure bytes changed: "+relative)
    old_main = expand("manuscript.tex", git_bytes, {})
    new_main = expand("manuscript.tex", current_bytes, {})
    if old_main != new_main:
        raise ValueError("current unit permits a supplement-only revision")
    (args.qa_dir/"old_expanded.tex").write_text(old)
    (args.qa_dir/"new_expanded.tex").write_text(new)
    command = ["perl", str(tool), "--encoding=utf8", "--type=CFONT", "--subtype=SAFE",
        "--floattype=IDENTICAL", "--math-markup=0", "old_expanded.tex", "new_expanded.tex"]
    marked = subprocess.run(command, cwd=args.qa_dir, text=True, capture_output=True, check=True)
    (args.qa_dir/"latexdiff_stderr.log").write_text(marked.stderr)
    notice = (r"\noindent{\color{blue}\footnotesize Change-marked review copy: "
        r"additions relative to the V62 primary baseline are shown in blue.}\par\medskip"+"\n")
    source = marked.stdout.replace(r"\begin{document}", r"\begin{document}"+"\n"+notice, 1)
    if source.count(r"\DIFadd{") < 1:
        raise ValueError("no visible added text was marked")
    target = HERE/"redlined_supplementary_information.tex"
    target.write_text(source)
    env = dict(os.environ)
    env["TEXINPUTS"] = str(HERE)+os.pathsep+env.get("TEXINPUTS", "")
    env["SOURCE_DATE_EPOCH"] = subprocess.check_output(["git", "show", "-s", "--format=%ct", "HEAD"], cwd=ROOT, text=True).strip()
    env["FORCE_SOURCE_DATE"] = "1"
    compiled = subprocess.run(["latexmk", "-xelatex", "-interaction=nonstopmode", "-halt-on-error",
        "-outdir="+str(args.qa_dir.resolve()), str(target)], cwd=HERE, env=env, text=True, capture_output=True)
    (args.qa_dir/"compile_stdout.log").write_text(compiled.stdout+compiled.stderr)
    if compiled.returncode:
        raise RuntimeError("redline compile failed; inspect "+str(args.qa_dir/"compile_stdout.log"))
    log = (args.qa_dir/target.with_suffix(".log").name).read_text()
    forbidden = ("Overfull", "undefined references", "There were undefined", "LaTeX Error", "Undefined control sequence")
    if any(text in log for text in forbidden):
        raise ValueError("redline has unresolved compile/layout warnings")
    output = target.with_suffix(".pdf")
    shutil.copy2(args.qa_dir/output.name, output)
    version_run = subprocess.run(["perl", str(tool), "--version"], text=True, capture_output=True, check=True)
    version = version_run.stdout+version_run.stderr
    receipt = dict(baseline_commit=revision, baseline_manifest_sha256=digest((HERE/"baselines/v62-primary/baseline_manifest.json").read_bytes()),
        old_input_sha256=old_hashes, current_input_sha256=new_hashes,
        article_expanded_source_unchanged=True, baseline_figures_unchanged=len(figure_names),
        shared_style_unchanged=True, latexdiff_source_sha256=TOOL_SHA, latexdiff_version=version.strip(),
        markup="CFONT / SAFE / IDENTICAL floats; math-markup=0", added_text_markers=source.count(r"\DIFadd{"),
        output_sha256={p.name: digest(p.read_bytes()) for p in (target, output)},
        builder_source_sha256=digest(Path(__file__).read_bytes()),
        clean_supplement_changed=False, primary_manuscript_changed=False, new_language_result=False)
    (HERE/"evidence/supplement_redline_build.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
