# Minimal Makefile for the TroopAI ADK documentation site.
# Drives Sphinx for the EN build.

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR      = .
BUILDDIR       = build

.PHONY: help html linkcheck livehtml clean

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

html:
	SPHINX_LANG=en $(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html/en" $(SPHINXOPTS)

linkcheck:
	$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck" $(SPHINXOPTS)

livehtml:
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" --watch ../src

clean:
	rm -rf "$(BUILDDIR)"
