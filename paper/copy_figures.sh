#!/bin/sh

# syntax ./copy_figures.sh main.tex arXiv/

for i in `./ltxclean.pl $1 | grep includegraphics | cut -d "{" -f2 | cut -d "}" -f1 | awk '{print $1 ".pdf"}'`; do cp $i $2; done
