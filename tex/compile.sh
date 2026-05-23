#!/bin/bash
# Compile results.tex to PDF (two passes for cross-refs) and clean aux files.
set -e
cd "$(dirname "$0")"

TEX=${1:-results.tex}
BASE=${TEX%.tex}

pdflatex -interaction=nonstopmode -halt-on-error "$TEX" > /dev/null
bibtex "$BASE" > /dev/null || true
pdflatex -interaction=nonstopmode -halt-on-error "$TEX" > /dev/null
pdflatex -interaction=nonstopmode -halt-on-error "$TEX" > /dev/null

rm -f "$BASE.aux" "$BASE.log" "$BASE.out" "$BASE.toc" "$BASE.fdb_latexmk" "$BASE.fls" "$BASE.synctex.gz" "$BASE.nav" "$BASE.snm" "$BASE.vrb" "$BASE.bbl" "$BASE.blg"
echo "Built: $BASE.pdf"
