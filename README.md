# hp_contention

This repo contains research material and a Springer LaTeX paper template under `springer_paper_template/`.

The template has been adjusted to use `fig.pdf` directly instead of relying on automatic EPS conversion. That avoids the `repstopdf` toolchain issue that commonly breaks local builds.

## Build The Paper

From the repo root:

```bash
cd springer_paper_template
latexmk -pdf sn-article.tex
```

Output:

- `springer_paper_template/sn-article.pdf`

Clean generated files:

```bash
cd springer_paper_template
latexmk -C
```

## macOS Setup

The least painful setup on macOS is full MacTeX plus LaTeX Workshop in VS Code.

1. Install Homebrew if it is not already installed.
2. Install MacTeX:

```bash
brew install --cask mactex-no-gui
```

3. Restart your terminal, then verify the tools exist:

```bash
which pdflatex
which latexmk
which bibtex
```

4. In VS Code, install the `LaTeX Workshop` extension.
5. Open this repo, then open `springer_paper_template/sn-article.tex`.
6. Build with the VS Code LaTeX Workshop build command, or run:

```bash
cd springer_paper_template
latexmk -pdf sn-article.tex
```

If `latexmk` is missing after MacTeX install, your shell may not have refreshed yet. Open a new terminal and try again.

## Notes

- Prefer PDF, PNG, or JPG figures for `pdflatex`.
- Avoid EPS unless you intentionally want to maintain an EPS conversion toolchain.
- The Springer template may still emit warnings about float anchors, font substitutions, and box layout. Those warnings do not currently block PDF generation.
