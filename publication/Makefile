TEXSRC := $(shell find . -maxdepth 2 -name '*.tex')
PDFS   := $(TEXSRC:.tex=.pdf)

.PHONY: all clean

all: $(PDFS)

%.pdf: %.tex
	../scripts/latex_build.sh $<

clean:
	# Remove common aux files recursively within publication/
	find . -type f \( -name '*.aux' -o -name '*.log' -o -name '*.out' -o -name '*.toc' -o -name '*.bbl' -o -name '*.blg' -o -name '*.synctex.gz' \) -delete
	# Remove hidden aux directories like .<name> next to each tex
	find . -mindepth 1 -type d -name '.*' -prune -exec rm -rf {} +
