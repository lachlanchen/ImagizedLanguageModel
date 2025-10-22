TEXSRC := $(wildcard *.tex)
PDFS   := $(TEXSRC:.tex=.pdf)

.PHONY: all clean

all: $(PDFS)

%.pdf: %.tex
	../scripts/latex_build.sh $<

clean:
	rm -f *.aux *.log *.out *.toc *.bbl *.blg *.synctex.gz
	# Keep pdf/tex; remove hidden aux dirs
	rm -rf .*/

