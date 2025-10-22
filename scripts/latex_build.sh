#!/usr/bin/env bash
set -euo pipefail

# Build a LaTeX file while routing intermediate files to a hidden folder
# named ".<basename>" next to the .tex. Keeps only .tex and .pdf in place.

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 path/to/file.tex" >&2
  exit 1
fi

TEX_PATH=$1
TEX_DIR=$(dirname "$TEX_PATH")
TEX_BASE=$(basename "$TEX_PATH" .tex)
AUX_DIR="$TEX_DIR/.${TEX_BASE}"

mkdir -p "$AUX_DIR"

pushd "$TEX_DIR" >/dev/null

if pdflatex --help 2>&1 | grep -q -- "-aux-directory"; then
  # MiKTeX: send aux/log/toc/etc. to AUX_DIR; PDF stays in TEX_DIR
  pdflatex -interaction=nonstopmode -halt-on-error -aux-directory="$AUX_DIR" "$TEX_BASE.tex"
  pdflatex -interaction=nonstopmode -halt-on-error -aux-directory="$AUX_DIR" "$TEX_BASE.tex"
else
  # TeX Live fallback: put everything in AUX_DIR, then move the PDF back
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$AUX_DIR" "$TEX_BASE.tex"
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$AUX_DIR" "$TEX_BASE.tex"
  if [[ -f "$AUX_DIR/$TEX_BASE.pdf" ]]; then
    mv -f "$AUX_DIR/$TEX_BASE.pdf" "$TEX_BASE.pdf"
  fi
fi

popd >/dev/null

echo "Built $TEX_DIR/$TEX_BASE.pdf; intermediates in $AUX_DIR"

