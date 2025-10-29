JOB = {target}_report
LATEX = pdflatex -interaction=nonstopmode
BIBTEX = biber -l zh__pinyin --output-safechars
DVIPDF = dvipdfmx

ifeq ($(LATEX), latex)
	DODVIPDF = $(DVIPDF) $(JOB)
endif

ifdef PATH
	MAKE = make
	RM = rm -f
else
	MAKE = mingw32-make
	RM = del
endif

doc:
	$(LATEX) $(JOB)
	$(LATEX) $(JOB)
	$(LATEX) $(JOB)
	$(DODVIPDF)

clean:
	$(RM) $(JOB).aux $(JOB).bbl $(JOB).bcf $(JOB).blg $(JOB).dvi \
		$(JOB).lof $(JOB).log $(JOB).lot $(JOB).run.xml $(JOB).toc \
		$(JOB).maf $(JOB).mtc $(JOB).mtc0 \
		$(JOB).out $(JOB).out.bak missfont.log texput.log sec/*.aux

distclean: clean
	$(RM) $(JOB).pdf

# vim:ts=4:sw=4