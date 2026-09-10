# TMLR submission

`main.tex` (12 pages of main content, the limit above which TMLR relaxes its two-week review
deadline) and `appendix.tex`, with `references.bib`.

## Building

The experiment host has no TeX installation, so the PDF committed here was produced with a static
[tectonic](https://tectonic-typesetting.github.io/) binary:

```bash
tectonic -X compile main.tex
```

For a standard TeX Live installation use the Makefile, which runs the usual
pdflatex/bibtex/pdflatex/pdflatex cycle:

```bash
make        # builds main.pdf
make pages  # reports the page count
make clean
```

## Layout

| file | contents |
|---|---|
| `main.tex` | the paper. Anonymous by default; `\usepackage[accepted]{tmlr}` produces the camera-ready and `[preprint]` a de-anonymised preprint |
| `appendix.tex` | proofs, the CSpD derivation, full tables, the metric that failed, falsifiers |
| `references.bib` | bibliography, cited with natbib author-year (natbib is loaded by `tmlr.sty`) |
| `tmlr.sty`, `tmlr.bst` | the official TMLR style files, unmodified, from `github.com/JmlrOrg/tmlr-style-file` |
| `figures/` | the four figure PDFs the paper includes |

`tmlr.sty` requires `fancyhdr`, which ships with any full TeX distribution; the upstream repository
also vendors a copy if the build machine lacks one.

## Figures

The paper includes four figures, all drawn by hand outside this repository and dropped into
`figures/` as PDFs:

| file | where it appears |
|---|---|
| `overview.pdf` | Figure 1, the setting and the obstruction |
| `feasibility.pdf` | Figure 2, the analytic collapse and its design reading |
| `conservation.pdf` | Figure 3, the Pythagorean split |
| `subspace.pdf` | Figure 4, five subspace rules on one frontier |

They are authored and compiled in Overleaf, so a local `make` needs those four PDFs present in
`figures/` first; nothing in this repository regenerates them.

## Reproducing the measurements

Every number in the tables comes from a scripted experiment that writes a JSON artefact recording
its own commit, versions, and arguments:

```bash
python scripts/run_e11.py                # E1.1 collapse sweep      -> Table 2
python scripts/run_e11b.py               # E1.1b convergence control
python scripts/run_e12a.py               # E1.2a random-projection null control
python scripts/run_e12_frontier.py       # E1.2 acceptance/fidelity frontier
python scripts/run_e12b.py               # E1.2b five subspace rules -> Table 3
python scripts/run_e14.py                # E1.4 distillation         -> Table 5
python scripts/run_e08.py                # E0.8 analytic reference curves
python scripts/plot_e05_spectrum.py \
    --results experiments/E0.5-kl16-jacobian/results.json \
    --samples experiments/E0.1-mar-base/samples.png \
    --out /tmp/e05_kl16_spectrum.png     # E0.5 KL16 spectrum        -> Table 4
```
