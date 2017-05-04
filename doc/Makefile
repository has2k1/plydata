# Makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = plydata
SOURCEDIR     = .
PAPER         =
BUILDDIR      = _build

# For make theme
theme = united
brand-primary = \#d04715
headings-font-weight = bold
font-family-sans-serif = '"Lato", "proxima-nova", "Helvetica Neue", Arial, sans-serif'

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo "  theme    to create a custom sphinx boostrap theme"
	@echo "  readme_images to generate images used in README.rst"

.PHONY: help Makefile theme

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

theme:
	../tools/build_theme.sh --theme $(theme) \
	   --brand-primary '$(brand-primary)' \
	   --headings-font-weight $(headings-font-weight) \
	   --font-family-sans-serif '$(font-family-sans-serif)'
