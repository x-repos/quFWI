#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# Main manuscript
latexmk -pdf -interaction=nonstopmode main.tex
latexmk -c main.tex  # clean aux files, keep main.pdf

# Supplementary information (separate PDF)
latexmk -pdf -interaction=nonstopmode supplementary.tex
latexmk -c supplementary.tex  # clean aux files, keep supplementary.pdf

# Cover letter
latexmk -pdf -interaction=nonstopmode cover_letter.tex
latexmk -c cover_letter.tex  # clean aux files, keep cover_letter.pdf
