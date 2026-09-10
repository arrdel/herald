# Build the submission. Requires pdflatex and bibtex on PATH.
#
# The build machine used for the experiments has no TeX installation, so this Makefile is the
# handoff: run it anywhere with a standard TeX Live and it produces main.pdf. Everything it needs
# beyond a base installation (geometry, natbib, fancyhdr, times, booktabs, amsmath, amssymb, amsthm,
# graphicx, xcolor, bm, tikz/pgf, hyperref, url, lineno) ships with any full distribution.

MAIN = main

.PHONY: all clean pages watch

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex appendix.tex references.bib tmlr.sty tmlr.bst $(wildcard figures/*.pdf)
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN)
	-bibtex $(MAIN)
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN)
	pdflatex -interaction=nonstopmode -halt-on-error $(MAIN)

# The page limits are a submission requirement, so make them checkable rather than eyeballed.
pages: $(MAIN).pdf
	@pdfinfo $(MAIN).pdf 2>/dev/null | grep Pages || \
	 echo "install poppler-utils for pdfinfo, or open the PDF"

clean:
	rm -f $(MAIN).aux $(MAIN).log $(MAIN).bbl $(MAIN).blg $(MAIN).out \
	      $(MAIN).pdf $(MAIN).toc $(MAIN).fls $(MAIN).fdb_latexmk
