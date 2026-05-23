# quanFWI Manuscript — Comm. AI & Computing Submission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `qFWI/tex/main.tex` (currently a partially-drafted, partially-lipsum manuscript) into a submission-ready paper for *Communications AI & Computing* (Nature Portfolio).

**Architecture:** Structure-first approach. Phase 1 does mechanical cleanup on a stable shell (fixes refs, consolidates tables, restructures to Nature order, swaps citation style) so later phases can write prose against a finalized skeleton. Phases 2-4 draft prose in claim-first order (Results & Discussion before Abstract & Intro). Phase 5 polishes for submission. Human review checkpoints gate each phase transition.

**Tech Stack:** LaTeX (pdflatex + biber), biblatex (`numeric-comp`), tikz/quantikz for figures, latexmk for builds. Design spec: [`docs/superpowers/specs/2026-04-20-commsai-submission-design.md`](../specs/2026-04-20-commsai-submission-design.md).

**Working branch:** `main` (solo maintainer; per spec §6.1). All changes commit to `main` with descriptive messages.

**Build command:** `./tex/compile.sh` (runs `latexmk -pdf -interaction=nonstopmode main.tex` + cleanup).

---

## Phase 1 — Mechanical shell cleanup

**Goal:** Paper compiles with no undefined refs/cites and no duplicate labels. Structure matches spec §2. No prose changes yet.

### Task 1.1: Remove orphan figures

**Files:**
- Delete: `tex/figures/cat_momo_1.png`, `tex/figures/cat_momo_2.jpg`, `tex/figures/fig_a.png`, `tex/figures/fig_b.png`, `tex/figures/fig_c.png`, `tex/figures/velocity_field.png`, `tex/figures/training_loss_smoothed.png`, `tex/figures/metrics_l1_velocity.png`

- [ ] **Step 1: Verify no `\includegraphics` reference these files**

Run: `grep -nE 'includegraphics.*(cat_momo|fig_[abc]|velocity_field|training_loss_smoothed|metrics_l1_velocity)' tex/main.tex`
Expected: no matches

- [ ] **Step 2: Delete the files**

```bash
cd tex/figures
rm -f cat_momo_1.png cat_momo_2.jpg fig_a.png fig_b.png fig_c.png velocity_field.png training_loss_smoothed.png metrics_l1_velocity.png
```

- [ ] **Step 3: Build to confirm no impact**

Run: `./tex/compile.sh`
Expected: succeeds, `main.pdf` produced.

- [ ] **Step 4: Commit**

```bash
git add tex/figures/
git commit -m "Remove orphan figures from tex/figures/"
```

---

### Task 1.2: Remove Borehole and Teleseismic lipsum stubs

**Files:**
- Modify: `tex/main.tex` around lines 834-838

- [ ] **Step 1: Delete the two subsections**

Remove these lines from `main.tex`:

```tex
\subsection{Borehold}
\lipsum[5]

\subsection{Teleseismic}
\lipsum[6]
```

- [ ] **Step 2: Build**

Run: `./tex/compile.sh`
Expected: succeeds, PDF has no Borehole/Teleseismic subsections.

- [ ] **Step 3: Commit**

```bash
git add tex/main.tex
git commit -m "Remove Borehole and Teleseismic placeholder subsections"
```

---

### Task 1.3: Fix figure label `fig: pqc` → `fig:pqc_architecture`

**Files:**
- Modify: `tex/main.tex` line 538

- [ ] **Step 1: Rename label**

Change `\label{fig: pqc}` to `\label{fig:pqc_architecture}` in the quantikz figure (around line 538, inside the PQC multilayer figure).

- [ ] **Step 2: Verify the reference resolves**

Run: `grep -n 'fig:pqc_architecture' tex/main.tex`
Expected: at least two matches (one `\label{}`, one `\ref{}` in the Quantum computing background subsection around line 336).

- [ ] **Step 3: Build and check for undefined refs**

Run: `./tex/compile.sh 2>&1 | grep -i 'Reference.*pqc_architecture.*undefined'`
Expected: no matches.

- [ ] **Step 4: Commit**

```bash
git add tex/main.tex
git commit -m "Fix fig:pqc_architecture label to resolve reference"
```

---

### Task 1.4: Add section labels for intro cross-references

**Files:**
- Modify: `tex/main.tex`

- [ ] **Step 1: Add `\label` to four sections referenced from Introduction**

Find each `\section{...}` and add a `\label{}` immediately after. Required labels:

```tex
\section{Background}\label{sec:background}       % if we keep a Background section (currently absent)
\section{Methods}\label{sec:method}              % after reorder in Task 1.8; for now label current Methodology
\section{Results}\label{sec:results}
\section{Discussion}\label{sec:discussion}
```

Since `sec:background` is referenced but there's no such section, either:
- (a) Merge its content into Methods (no separate Background section) and update the intro overview paragraph to drop `sec:background`, OR
- (b) Keep the label on Methodology/Methods for now.

**Decision: (a)** — drop the `sec:background` reference from the intro overview paragraph (line 75). The relevant background content lives inside Methods §6.2-6.5 per spec §2.

So the changes are:
1. Edit line 75 section-overview paragraph: remove `Section~\ref{sec:background}` sentence.
2. Add labels: `\label{sec:method}` after `\section{Methodology}` (line 79), `\label{sec:results}` after `\section{Results}` (line 697), `\label{sec:discussion}` after `\section{Discussion}` (line 840).

- [ ] **Step 2: Build and check for undefined refs**

Run: `./tex/compile.sh 2>&1 | grep -i 'Reference.*undefined'`
Expected: no matches for `sec:background`, `sec:method`, `sec:results`, `sec:discussion`.

- [ ] **Step 3: Commit**

```bash
git add tex/main.tex
git commit -m "Add section labels and drop stale sec:background reference"
```

---

### Task 1.5: Add 6 missing bibtex entries

**Files:**
- Modify: `tex/references.bib` (append)
- Modify: `tex/main.tex` lines ~430-480 (remove commented bibtex block)

**Missing keys:** `benedetti2019parameterized`, `bergholm2018pennylane`, `mitarai2018quantum`, `schuld2019evaluating`, `schuld2021machine`, `sim2019expressibility`

- [ ] **Step 1: Append entries to `tex/references.bib`**

Append exactly these entries (they are the ones currently commented out inside `main.tex` lines ~432-479, cleaned up):

```bibtex
@article{benedetti2019parameterized,
  title={Parameterized quantum circuits as machine learning models},
  author={Benedetti, Marcello and Lloyd, Erika and Sack, Stefan and Fiorentini, Mattia},
  journal={Quantum Science and Technology},
  volume={4},
  number={4},
  pages={043001},
  year={2019},
  publisher={IOP Publishing},
  doi={10.1088/2058-9565/ab4eb5}
}

@article{mitarai2018quantum,
  title={Quantum circuit learning},
  author={Mitarai, Kosuke and Negoro, Makoto and Kitagawa, Masahiro and Fujii, Keisuke},
  journal={Physical Review A},
  volume={98},
  number={3},
  pages={032309},
  year={2018},
  publisher={American Physical Society},
  doi={10.1103/PhysRevA.98.032309}
}

@article{schuld2019evaluating,
  title={Evaluating analytic gradients on quantum hardware},
  author={Schuld, Maria and Bergholm, Ville and Gogolin, Christian and Izaac, Josh and Killoran, Nathan},
  journal={Physical Review A},
  volume={99},
  number={3},
  pages={032331},
  year={2019},
  publisher={American Physical Society},
  doi={10.1103/PhysRevA.99.032331}
}

@article{sim2019expressibility,
  title={Expressibility and entangling capability of parameterized quantum circuits for hybrid quantum-classical algorithms},
  author={Sim, Sukin and Johnson, Peter D. and Aspuru-Guzik, Al{\'a}n},
  journal={Advanced Quantum Technologies},
  volume={2},
  number={12},
  pages={1900070},
  year={2019},
  publisher={Wiley},
  doi={10.1002/qute.201900070}
}

@book{schuld2021machine,
  title={Machine Learning with Quantum Computers},
  author={Schuld, Maria and Petruccione, Francesco},
  year={2021},
  edition={2nd},
  publisher={Springer},
  series={Quantum Science and Technology}
}

@misc{bergholm2018pennylane,
  title={{PennyLane}: Automatic differentiation of hybrid quantum-classical computations},
  author={Bergholm, Ville and Izaac, Josh and Schuld, Maria and Gogolin, Christian and Ahmed, Shahnawaz and Ajith, Vishnu and Alam, M. Sohaib and Alonso-Linaje, Guillermo and AkashNarayanan, B. and Asadi, Ali and others},
  year={2018},
  eprint={1811.04968},
  archivePrefix={arXiv},
  primaryClass={quant-ph}
}
```

- [ ] **Step 2: Remove the commented-out bibtex block from main.tex**

Delete `main.tex` lines ~431-480 (the `% === BibTeX entries (add to your .bib file) ===` block through the closing `% }`).

- [ ] **Step 3: Build and check for undefined citations**

Run: `./tex/compile.sh 2>&1 | grep -i 'Citation.*undefined'`
Expected: no matches for the 6 keys above.

- [ ] **Step 4: Commit**

```bash
git add tex/references.bib tex/main.tex
git commit -m "Add bibtex entries for quantum-subsection citations"
```

---

### Task 1.6: Consolidate hyperparameter tables

**Files:**
- Modify: `tex/main.tex` lines 702-820 (three tables)
- This replaces the **two colliding `tab:hyperparams` tables** (lines 741-785 and 791-820) and the **forward-modeling table** (lines 702-735) with one main-text table plus two supplementary tables.

**Design from spec §3.2:**
- Main text **T1**: `tab:hyperparams_main` — Rasht FWI only, classical baseline + quantum only (2 rows).
- Supplementary **ST1**: `tab:hyperparams_full` — full 15+ FWI variant sweep (currently Table 3).
- Supplementary **ST2**: `tab:hyperparams_forward` — forward-modeling configurations (currently Table 1, already labeled).

- [ ] **Step 1: Create the main-text T1 (baseline + quantum only)**

Replace the entire block from `\begin{table}[h]` on line 741 through `\end{table}` on line 785 with a compact two-row table. Insert it where Table 1 currently is (since Results section is being rewritten later; precise position doesn't matter yet as long as it's in the Results area).

```tex
\begin{table}[h]
\centering
\caption{Network configurations for the Rasht FWI benchmark. $\mathcal{Q}(n_q^{\times n_L})$ denotes a parameterized quantum circuit with $n_q$ qubits and $n_L$ layers. Subdomain decomposition applies to the wavefield network $\phi$; the velocity network $c$ is not decomposed. Loss weights: $\lambda_\text{PDE}=0.1$, $\lambda_\text{IC}=1.0$, $\lambda_\text{seis}=1.0$, $\lambda_\text{BC}=0.1$. Learning rate $10^{-4}$. Subdomain overlap $0.35$. See Supplementary Table~\ref{tab:hyperparams_full} for the full variant sweep.}
\label{tab:hyperparams_main}
\begin{tabular}{l l l r}
\toprule
\textbf{Model} & \makecell[l]{$\boldsymbol{\phi}$ \textbf{network}\\$(x,z,t)\!\to\!\phi$} & \makecell[l]{$\boldsymbol{c}$ \textbf{network}\\$(x,z)\!\to\!c$} & \textbf{\# Params} \\
\midrule
\texttt{classical} & $\mathcal{N}\!: 32^{\times 2}\!\to\!16^{\times 2}$ & $\mathcal{N}\!: 20^{\times 5}$ & 151{,}836 \\
\texttt{quantum}   & $\mathcal{N}\!: 32^{\times 2}$ & $\mathcal{N}\!: 20^{\times 3}\!\to\!\mathcal{Q}\!: 4^{\times 2}$ & 101{,}812 \\
\bottomrule
\end{tabular}
\end{table}
```

- [ ] **Step 2: Move the full variant table to the appendix**

Find the second `tab:hyperparams` table (the larger one with all variants, currently around lines 741-785 before the change in Step 1 — but after Step 1 this IS the table we just replaced). So: the existing large table is being *replaced*, not moved.

Instead, the full variant table needs to be **re-created in the appendix** (lines 882-909 area) with:
- Label: `\label{tab:hyperparams_full}`
- All 15+ variant rows from the pre-replacement Table 2 / Table 3.

Add this to the appendix (before `\end{appendices}`):

```tex
\begin{table}[h]
\centering
\caption{Supplementary: full hyperparameter variant sweep for Rasht FWI. Values marked ``--'' are identical to the \texttt{classical} baseline (Table~\ref{tab:hyperparams_main}).}
\label{tab:hyperparams_full}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l l l c c c c c c c r}
\toprule
\textbf{Model} & \makecell{$\boldsymbol{c}$ \textbf{network}\\$(x,z)\!\to\!c$} & \makecell{$\boldsymbol{\phi}$ \textbf{network}\\$(x,z,t)\!\to\!\phi$} & \textbf{LR} & \textbf{Subdomains} & \textbf{Overlap} & $\boldsymbol{\lambda_\text{PDE}}$ & $\boldsymbol{\lambda_\text{IC}}$ & $\boldsymbol{\lambda_\text{seis}}$ & $\boldsymbol{\lambda_\text{BC}}$ & \textbf{\# Pars} \\
\midrule
\texttt{classical} & $\mathcal{N}\!: 20^{\times 5}$ & $\mathcal{N}\!: 32^{\times 2}$ & $10^{-4}$ & $5\!\times\!3\!\times\!5$ & 0.35 & 0.1 & 1.0 & 1.0 & 0.1 & 151{,}836 \\
\midrule
\texttt{c10x2} & $\mathcal{N}\!: 10^{\times 2}$ & -- & -- & -- & -- & -- & -- & -- & -- & 150{,}226 \\
\texttt{c10x3} & $\mathcal{N}\!: 10^{\times 3}$ & -- & -- & -- & -- & -- & -- & -- & -- & 150{,}336 \\
\texttt{c20x3} & $\mathcal{N}\!: 20^{\times 3}$ & -- & -- & -- & -- & -- & -- & -- & -- & 150{,}996 \\
\texttt{c40x3} & $\mathcal{N}\!: 40^{\times 3}$ & -- & -- & -- & -- & -- & -- & -- & -- & 153{,}516 \\
\texttt{c40x5} & $\mathcal{N}\!: 40^{\times 5}$ & -- & -- & -- & -- & -- & -- & -- & -- & 156{,}796 \\
\texttt{c60x3} & $\mathcal{N}\!: 60^{\times 3}$ & -- & -- & -- & -- & -- & -- & -- & -- & 157{,}636 \\
\midrule
\texttt{phi16x2} & -- & $\mathcal{N}\!: 16^{\times 2}$ & -- & -- & -- & -- & -- & -- & -- & 25{,}836 \\
\texttt{phi64x3} & -- & $\mathcal{N}\!: 64^{\times 3}$ & -- & -- & -- & -- & -- & -- & -- & 649{,}836 \\
\midrule
\texttt{sub3x2x3} & -- & -- & -- & $3\!\times\!2\!\times\!3$ & -- & -- & -- & -- & -- & 37{,}779 \\
\texttt{sub8x4x8} & -- & -- & -- & $8\!\times\!4\!\times\!8$ & -- & -- & -- & -- & -- & 514{,}017 \\
\midrule
\texttt{lr5e4} & -- & -- & $5\!\times\!10^{-4}$ & -- & -- & -- & -- & -- & -- & -- \\
\texttt{lr5e5} & -- & -- & $5\!\times\!10^{-5}$ & -- & -- & -- & -- & -- & -- & -- \\
\texttt{overlap50} & -- & -- & -- & -- & 0.50 & -- & -- & -- & -- & -- \\
\texttt{wpde01\_wbc01} & -- & -- & -- & -- & -- & 0.01 & -- & -- & 0.01 & -- \\
\texttt{wpde1\_wbc1} & -- & -- & -- & -- & -- & 1.0 & -- & -- & 1.0 & -- \\
\midrule
\texttt{quantum} & $\mathcal{N}\!: 20^{\times 3}\!\to\!\mathcal{Q}\!: 4^{\times 2}$ & $\mathcal{N}\!: 32^{\times 2}$ & -- & -- & -- & -- & -- & -- & -- & 101{,}812 \\
\bottomrule
\end{tabular}%
}
\end{table}
```

- [ ] **Step 3: Relabel forward-modeling table as supplementary**

In the existing forward-modeling table (lines 702-735), change:
- `\label{tab:hyperparams_forward}` stays as-is
- Move the whole table into the appendix (after `tab:hyperparams_full` from Step 2)
- Update its caption to prefix "Supplementary:"

- [ ] **Step 4: Build and check for duplicate labels**

Run: `./tex/compile.sh 2>&1 | grep -i 'multiply defined'`
Expected: no matches.

Run: `grep -c 'tab:hyperparams' tex/main.tex`
Expected: 3 (one `_main`, one `_full`, one `_forward`).

- [ ] **Step 5: Commit**

```bash
git add tex/main.tex
git commit -m "Consolidate hyperparameter tables: one main, two supplementary"
```

---

### Task 1.7: Switch biblatex citation style to `numeric-comp`

**Files:**
- Modify: `tex/preamble.tex` line 41

- [ ] **Step 1: Change biblatex style**

In `tex/preamble.tex` line 41, change:

```tex
\usepackage[style=apa, backend=biber]{biblatex} % APA 7th edition style citations using biblatex
```

to:

```tex
\usepackage[style=numeric-comp, backend=biber, sorting=none]{biblatex} % Nature-style numeric citations
```

- [ ] **Step 2: Remove the APA-specific preamble lines**

In `tex/preamble.tex`, remove lines 47-55 (the APA-7 language mapping, volume format redefinition, and the `\finalnamedelim` override — these are APA-specific and won't apply cleanly to numeric style).

- [ ] **Step 3: Build and inspect citation rendering**

Run: `./tex/compile.sh`
Expected: succeeds. Open `tex/main.pdf` and spot-check that citations render as numeric brackets `[1]`, `[2, 3]` rather than APA author-year.

- [ ] **Step 4: Commit**

```bash
git add tex/preamble.tex
git commit -m "Switch biblatex to numeric-comp style for Nature format"
```

---

### Task 1.8: Reorder document — Methodology → Methods at end

**Files:**
- Modify: `tex/main.tex`

**Target final order (spec §2):**
```
\section{Introduction}
\section{Results}\label{sec:results}
  \subsection{Architecture and training setup}
  \subsection{Rasht velocity anomaly benchmark}
  \subsection{Checkerboard benchmark}
\section{Discussion}\label{sec:discussion}
\section{Conclusions}
\section{Methods}\label{sec:method}
  \subsection{Acoustic wave equation}
  \subsection{Physics-informed neural networks}
  \subsection{Finite-basis physics-informed neural networks}
  \subsection{Parameterized quantum circuits}
  \subsection{Hybrid quantum-classical architecture}
  \subsection{JAX-based statevector simulator}
  \subsection{Training setup}
  \subsection{Data}
\section*{Data availability}
\section*{Code availability}
\section*{Acknowledgments}
\section*{Author contributions}
\section*{Conflicts of interest}
\printbibliography
\begin{appendices}...\end{appendices}
```

**Note:** Results and Conclusions subsection bodies are still largely lipsum/empty — that is intentional. This task *only* reorders. Content drafting happens in Phases 2-4.

- [ ] **Step 1: Cut the Methodology block**

Select lines from `\section{Methodology}` (line 79) through the end of the last methodology subsection (`\subsection{Hybrid quantum physics-informed neural network}` ending around line 694 — i.e. just before `\section{Results}`).

- [ ] **Step 2: Rename and paste after `\section{Conclusions}`**

Rename `\section{Methodology}` → `\section{Methods}\label{sec:method}` and paste the whole block after the current `\section{Conclusions}\lipsum[8]` block (around line 844), before `\section*{Conflicts of Interest}`.

- [ ] **Step 3: Reorder subsections inside Methods**

Within the Methods block, ensure subsection order matches the target above:
1. Acoustic wave equation
2. Physics-informed neural networks (currently empty — stays empty, Phase 4 drafts)
3. Finite-basis physics-informed neural networks (currently empty — stays empty)
4. Parameterized quantum circuits (merge: current "Quantum computing background" + "Quantum parameterized circuits" stub — consolidate into one subsection)
5. Hybrid quantum-classical architecture (current "Hybrid quantum physics-informed neural network")
6. JAX-based statevector simulator (current "JAX-based quantum simulators")
7. Training setup (new — stays empty, Phase 4 drafts)
8. Data (new — stays empty, Phase 4 drafts)

Merge-step specifics:
- Delete the `\subsection{Quantum parameterized circuits}` stub (one-line subsection heading with nothing in it — lines ~482).
- The long `\subsection{Quantum computing background}` with its `\subsubsection`s becomes `\subsection{Parameterized quantum circuits}`.
- The `\subsection{Hybrid quantum physics-informed neural network}` becomes `\subsection{Hybrid quantum-classical architecture}`.
- Add empty `\subsection{Training setup}` and `\subsection{Data}` headers at the end of Methods for Phase 4 to fill.

- [ ] **Step 4: Add the `\section*{Data availability}` header**

After Methods and before `\section*{Code Availability}`, add:

```tex
\section*{Data availability}

The synthetic seismic data used in this study were generated with SPECFEM2D~\cite{komatitsch1998spectral} and are available in the project repository at \url{https://github.com/x-repos/quFWI} under \texttt{data/rasht/} and \texttt{data/checkerboard/}.
```

(SPECFEM2D citation placeholder — add to references.bib in Phase 5 polish if not already present. If absent, mark with a FIXME comment for now.)

- [ ] **Step 5: Add missing section labels**

Ensure `\label{sec:results}`, `\label{sec:discussion}`, `\label{sec:method}` are all set after their `\section{...}` commands (Task 1.4 may have already done this — verify).

- [ ] **Step 6: Build and verify**

Run: `./tex/compile.sh`
Expected: succeeds. PDF order: Abstract (empty) → Intro → Results (lipsum subsections) → Discussion (lipsum) → Conclusions (lipsum) → Methods (drafted) → Data availability → Code availability → Acknowledgments → Author contributions → Conflicts → References → Appendix.

- [ ] **Step 7: Check ref resolution**

Run: `./tex/compile.sh 2>&1 | grep -iE '(undefined|multiply defined)'`
Expected: only possibly `komatitsch1998spectral` undefined (will be added in polish); no other issues.

- [ ] **Step 8: Commit**

```bash
git add tex/main.tex
git commit -m "Reorder: Methodology to Methods at end (Nature format)"
```

---

### **CHECKPOINT 1 — user review**

**Assistant reports to user:** "Phase 1 done. Building main.pdf. Check: (1) compile is clean (no undefined refs/cites except SPECFEM2D which is flagged); (2) section order is Intro → Results → Discussion → Conclusions → Methods → Code/Data/Ack → Refs → Appendix; (3) tables: one main (classical+quantum), two supplementary; (4) citations render as numeric `[1]` in the PDF. Let me know if anything looks off before I start drafting prose."

**User reviews** by opening `tex/main.pdf` and confirming structure. Any issues → fix and re-commit before Phase 2 starts.

---

## Phase 2 — Results and Discussion

**Goal:** Draft Results + Discussion prose so the user can verify the scientific claim framing before Abstract and Intro are written on top of it.

**Framing rules (spec §5, all approved):**
- Claim: "faster convergence in training iterations" / "improved optimization dynamics" — never "quantum speedup" or "quantum advantage."
- FBPINN is core; emphasize domain decomposition + quantum.
- Simulation framed as deliberate design choice (in Methods) + limitation (here in Discussion).
- Cross-domain applications: one-sentence mention at most.

**Quantitative claims available (from figures already in the repo):**
- Quantum reaches L1 ≈ 2.4×10⁻² at 65k training steps.
- Classical baseline reaches L1 ≈ 2.9×10⁻² at 500k steps; all 15+ classical variants stay above the quantum curve through 500k.
- Parameter count: quantum = 101,812; classical baseline = 151,836 (33% fewer).
- So: quantum reaches better final L1 with ~8× fewer iterations and ~33% fewer parameters.

### Task 2.1: Draft §4.1 Architecture and training setup

**Files:**
- Modify: `tex/main.tex` (Results section, insert as first subsection)

**What goes in this subsection (≤ 200 words):**
- One paragraph summarizing the FBPINN-FWI setup: scalar potential formulation, two networks (φ wavefield + c velocity), classical baseline vs hybrid quantum-classical variant.
- Points the reader to Methods §6 for details.
- Introduces Figures F1 (architecture schematic) and F2 (PQC circuit) with pointer references.
- Introduces Table T1 (`tab:hyperparams_main`).

- [ ] **Step 1: Draft the subsection**

Write a paragraph with:
- Opening: we compare a classical FBPINN baseline against a hybrid quantum-classical variant on two 2D acoustic FWI benchmarks.
- Description: both use a wavefield network φ(x,z,t) and a velocity network c(x,z). The classical baseline uses fully-connected nets for both; the hybrid variant replaces the c-network with a classical-to-quantum pipeline (classical embedding → PQC → scalar output).
- Architecture summary: reference `\autoref{fig:hybrid_arch}` for the overall layout, `\autoref{fig:pqc_architecture}` for the PQC.
- Hyperparameters: reference `\autoref{tab:hyperparams_main}`, defer the 15+ variant sweep to Supplementary Table `\ref{tab:hyperparams_full}`.
- One-sentence pointer: "Full methodological details are in Methods §\ref{sec:method}."

- [ ] **Step 2: Replace the first `\lipsum[1]` in Results with this subsection**

Insert under `\section{Results}` / replace the `\lipsum[1]` placeholder. Delete the three `\subsubsection` lipsum stubs (`\subsubsection{Physics-informed neural network}\lipsum[2]` etc.) — content is in Methods, not Results.

- [ ] **Step 3: Build**

Run: `./tex/compile.sh`
Expected: succeeds. Verify `\autoref{fig:hybrid_arch}` and `\autoref{fig:pqc_architecture}` resolve.

- [ ] **Step 4: Commit**

```bash
git add tex/main.tex
git commit -m "Draft Results §4.1 Architecture and training setup"
```

---

### Task 2.2: Draft §4.2 Rasht — inversion quality

**Files:**
- Modify: `tex/main.tex`

**What goes here (≤ 250 words):**
- Paragraph 1: Rasht benchmark setup (reference to Methods §Data). Source/receivers, true velocity (laterally varying Gaussian anomaly), observation window.
- Paragraph 2: Both models recover the anomaly (figures (a) true, (b) initial, (c) inverted). Qualitative agreement. Reference `\autoref{fig:rasht_inversion}`.

Figure F3 is the 4-panel `l1_variants.pdf` — panel (a)(b)(c) is the velocity models, panel (d) is the convergence curve. So we point to F3 for both velocity and convergence.

- [ ] **Step 1: Write two paragraphs**

Paragraph 1 (~100 words): Benchmark description. "The Rasht benchmark is a 1.3 km × 0.45 km laterally varying velocity model featuring a localized high-velocity anomaly (α ≈ 3.1 km/s) embedded in a 2.7 km/s background. A single seismic source at (0.35 km, 0.19 km) emits a Gaussian wavelet, recorded by a linear array of 20 receivers along the right boundary. Synthetic seismograms were generated with SPECFEM2D. The inversion window is 10 s of wave propagation." (Adjust numbers to match actual SPECFEM setup — cross-check against `data/rasht/specfem/event1/` and scripts if uncertain.)

Paragraph 2 (~100 words): Qualitative inversion result. "Both classical and quantum hybrid models recover the velocity anomaly from an initial homogeneous model (`\autoref{fig:rasht_inversion}` a–c). The inverted anomaly location and amplitude closely match the true model, with residual smoothing consistent with the limited source-receiver illumination." No quantitative claim here; that goes in Task 2.3 (convergence).

- [ ] **Step 2: Add figure reference**

Introduce the figure block for F3 (Rasht inversion + convergence) here or in §4.2.2 — decide based on flow. Use the existing `tex/figures/l1_variants.pdf` file. Caption the figure with all four panels described.

```tex
\begin{figure}[h]
\centering
\includegraphics[width=\linewidth]{figures/l1_variants.pdf}
\caption{Rasht benchmark inversion and convergence. (a) True velocity model with source (red star) and receiver array (green triangles). (b) Initial homogeneous velocity. (c) Inverted velocity from the quantum hybrid model at 65k training steps. (d) L1 velocity error as a function of training step for the classical FBPINN baseline (blue), the quantum hybrid model (black diamonds), and 15 classical hyperparameter variants (thin colored lines). The quantum hybrid reaches a lower L1 error in $\sim$8$\times$ fewer iterations than any classical variant.}
\label{fig:rasht_inversion}
\end{figure}
```

- [ ] **Step 3: Build and verify**

Run: `./tex/compile.sh`
Expected: figure renders, reference resolves.

- [ ] **Step 4: Commit**

```bash
git add tex/main.tex
git commit -m "Draft Results §4.2 Rasht inversion quality"
```

---

### Task 2.3: Draft §4.2 Rasht — convergence comparison (THE key paragraph)

**Files:**
- Modify: `tex/main.tex`

**What goes here (~200 words):**
This is the paper's money claim. Be precise.

- [ ] **Step 1: Write the convergence paragraph**

Content to cover:
- At 65k training steps, the quantum hybrid model reaches L1 velocity error ≈ 2.4×10⁻².
- The classical FBPINN baseline, run out to 500k steps, reaches L1 ≈ 2.9×10⁻². The quantum model at 65k is already below the classical baseline at 500k.
- Framing: "The quantum hybrid velocity network achieves improved optimization dynamics, reaching comparable or better inversion accuracy in approximately 8× fewer training iterations." **Do not write "quantum speedup" or "quantum advantage."**
- Parameter count note: quantum model has 101,812 parameters vs classical baseline's 151,836 (~33% fewer).
- Reference `\autoref{fig:rasht_inversion}d`.

Skeleton (fill in):

```tex
\autoref{fig:rasht_inversion}(d) shows the L1 velocity error as a function of training iteration for the classical FBPINN baseline and the quantum hybrid model. The quantum hybrid reaches an L1 error of $\sim$$2.4\times 10^{-2}$ after 65{,}000 iterations, already below the classical baseline's L1 error of $\sim$$2.9\times 10^{-2}$ at 500{,}000 iterations---an $\sim$8$\times$ reduction in the number of training steps required to reach comparable inversion accuracy. The quantum hybrid also uses fewer trainable parameters overall (101{,}812 vs.\ 151{,}836 for the classical baseline). We interpret this as improved optimization dynamics from the quantum-hybrid parameterization of the velocity field, rather than a claim of quantum computational advantage; the PQC itself is simulated classically (see Methods).
```

- [ ] **Step 2: Build**

Run: `./tex/compile.sh`
Expected: succeeds.

- [ ] **Step 3: Commit**

```bash
git add tex/main.tex
git commit -m "Draft Results §4.2 Rasht convergence comparison"
```

---

### Task 2.4: Draft §4.2 Rasht — hyperparameter variant robustness

**Files:**
- Modify: `tex/main.tex`

**What goes here (~150 words):**
- Paragraph showing the quantum model outperforms *all* 15+ classical variants (velocity network size, φ network size, subdomain decomposition, learning rate, loss weights, overlap).
- This addresses the "are you just unlucky with classical hyperparameters?" reviewer concern.
- Reference Supplementary Figure SF3 (variant L1 sweep) and Supplementary Table ST1.

- [ ] **Step 1: Write paragraph**

```tex
To verify that the convergence advantage is robust to classical hyperparameter choices, we ran 15 additional classical FBPINN variants spanning velocity-network capacity ($c$-network hidden sizes from $10\!\times\!2$ to $60\!\times\!3$), wavefield-network size ($\phi$-network hidden sizes $16\!\times\!2$ and $64\!\times\!3$), subdomain decomposition ($3\!\times\!2\!\times\!3$ and $8\!\times\!4\!\times\!8$), learning rate ($5\!\times\!10^{-5}$ to $5\!\times\!10^{-4}$), subdomain overlap ($0.35$ and $0.50$), and loss weights. All variants are shown in \autoref{fig:rasht_inversion}(d) as thin colored curves and listed in Supplementary Table~\ref{tab:hyperparams_full}. None of the 15 classical variants reach the quantum hybrid's L1 error at any training step up to 500{,}000 iterations. The quantum hybrid's convergence advantage is therefore not an artifact of a specific classical baseline choice.
```

- [ ] **Step 2: Build**

Run: `./tex/compile.sh`
Expected: succeeds.

- [ ] **Step 3: Commit**

```bash
git add tex/main.tex
git commit -m "Draft Results §4.2 Rasht hyperparameter robustness"
```

---

### Task 2.5: Draft §4.3 Checkerboard

**Files:**
- Modify: `tex/main.tex`
- Copy: `results/checkerboard/plots/true_and_inverted.png` → `tex/figures/checkerboard_inversion.png`

**What goes here (~150 words):**
- Checkerboard is a classical resolution stress test. 3×3 checkerboard velocity pattern.
- Classical FBPINN recovers the pattern structure.
- **Explicit note:** quantum hybrid results are pending and will be added in revision. Be honest.
- Reference new Figure F5.

- [ ] **Step 1: Copy the checkerboard figure**

```bash
cp /home/x/Workspace/1-qFWI-bundle/qFWI/results/checkerboard/plots/true_and_inverted.png \
   /home/x/Workspace/1-qFWI-bundle/qFWI/tex/figures/checkerboard_inversion.png
```

- [ ] **Step 2: Add figure block**

```tex
\begin{figure}[h]
\centering
\includegraphics[width=\linewidth]{figures/checkerboard_inversion.png}
\caption{Checkerboard benchmark inversion. (a) True velocity model: a 3$\times$3 checkerboard of alternating $\pm$$0.4$ km/s perturbations overlaid on a 3.0 km/s background, with source (red star) and receiver array (green triangles). (b) Inverted velocity from the classical FBPINN at 2M training steps. The checkerboard pattern is recovered with resolution-limited smoothing, particularly away from the receiver array. Quantum hybrid results for this benchmark are pending and will be reported in a follow-up study.}
\label{fig:checkerboard_inversion}
\end{figure}
```

- [ ] **Step 3: Write subsection body (~100 words)**

```tex
\subsection{Checkerboard benchmark}
\label{sec:results_checkerboard}

To assess resolution characteristics, we also invert a checkerboard velocity model: a 3$\times$3 grid of alternating $\pm$$0.4$ km/s perturbations embedded in a 3.0 km/s background (\autoref{fig:checkerboard_inversion}a). This is a standard FWI resolution test. The classical FBPINN recovers the checkerboard structure after 2M training iterations (\autoref{fig:checkerboard_inversion}b), with resolution progressively degrading for blocks located further from the receiver array---consistent with the limited angular coverage provided by a single source and a one-sided receiver line. A quantum hybrid experiment on this benchmark is in progress and will be included in a revised version of this work; the current comparison therefore rests on the Rasht benchmark alone.
```

- [ ] **Step 4: Build**

Run: `./tex/compile.sh`
Expected: succeeds. Figure renders.

- [ ] **Step 5: Commit**

```bash
git add tex/main.tex tex/figures/checkerboard_inversion.png
git commit -m "Draft Results §4.3 Checkerboard (classical only)"
```

---

### Task 2.6: Draft Discussion

**Files:**
- Modify: `tex/main.tex`

**What goes here (~600 words, 4 short paragraphs):**

Paragraph D1 — Interpretation of the quantum convergence advantage (~150 words):
- Reiterate: quantum hybrid converges ~8× faster in iteration count to comparable/better L1.
- Plausible mechanism: the PQC provides an inductive bias — bounded expressibility in a structured functional basis — that matches the smooth, spatially correlated nature of velocity fields better than a fully-connected tanh network at the same parameter count.
- Caveat: optimization-dynamics improvements in iteration count are not quantum computational speedup — the circuit runs on a classical statevector simulator.

Paragraph D2 — Relation to prior work (~150 words):
- LEONG2025106782 (HQPINN for compressible flow): similar conclusion that hybrid-quantum beats pure classical for PDE-constrained learning.
- Berger2025 (TE-QPINN): similar observation at matched parameter count.
- Rahst2022 (pure PINN for FWI): established the base framework this extends.
- Moseley2023 (FBPINN): domain decomposition foundation.
- **Our contribution relative to these:** first integration of PQCs into *domain-decomposed* PINNs for waveform inversion; first demonstration on a geophysical inverse problem with multi-source-term, multi-receiver data.

Paragraph D3 — Limitations (~150 words):
- 2D only; 3D FWI is the real target and statevector simulation cost scales $2^n$.
- Single source in current experiments; multi-source inversion is standard in production FWI.
- Classical simulation: no real quantum hardware experiments. Hardware runs would require parameter-shift gradients (2p evaluations per gradient) and current-device noise, which degrade the clean statevector picture.
- Acoustic only (no elasticity / density). Anisotropy and attenuation are important for real data.

Paragraph D4 — Outlook (~150 words):
- Hardware experiments as NISQ devices mature (small qubit counts, which is the regime used here).
- Scaling: more qubits → deeper/wider velocity representations; compare with classical baselines at matched parameter count to verify the trend.
- Extension to elastic FWI, multi-source, 3D.
- One sentence on cross-domain applicability (medical ultrasound tomography, photoacoustic imaging, NDE) — signal generality but do not over-claim.

- [ ] **Step 1: Replace `\lipsum[7]` with the four-paragraph draft**

Draft paragraphs D1-D4 as above, citing relevant prior work with the biblatex keys already in references.bib (`LEONG2025106782`, `Berger2025`, `rahst`, `Moseley2023`, etc.).

- [ ] **Step 2: Build**

Run: `./tex/compile.sh`
Expected: succeeds, no undefined cites.

- [ ] **Step 3: Commit**

```bash
git add tex/main.tex
git commit -m "Draft Discussion (interpretation, prior work, limitations, outlook)"
```

---

### **CHECKPOINT 2 — user review**

**Assistant reports to user:** "Phase 2 done — Results + Discussion drafted. Please read specifically:
1. §4.2 convergence paragraph (Task 2.3) — is the 8× / L1 / parameter claim correct?
2. §4.3 Checkerboard — does the 'quantum pending' framing match your plan?
3. Discussion D3 Limitations — did I miss anything that should be disclosed up front?

If any claim is wrong, off, or overclaimed — tell me now, before Abstract and Intro are built on top."

**User reviews** Results + Discussion. Corrections → inline revisions → re-commit.

---

## Phase 3 — Framing

**Goal:** Draft abstract + finalize intro + draft conclusions, now that the central story is locked.

### Task 3.1: Draft Abstract

**Files:**
- Modify: `tex/main.tex` (abstract block around line 35-37)

**Abstract spec (spec §5, decision 4.1):**
- 150-200 words
- Novelty framing + quantitative win
- No jargon overload in first sentence
- Closes with generality / outlook sentence

- [ ] **Step 1: Write abstract**

Skeleton (~180 words):

```tex
\begin{abstract}
Full waveform inversion (FWI) reconstructs subsurface material properties from seismic data but remains computationally demanding. Physics-informed neural networks (PINNs) and their domain-decomposed variants (FBPINNs) offer a mesh-free alternative but face challenges in convergence speed and representational efficiency for velocity fields. We present a hybrid quantum-classical FBPINN for acoustic FWI, in which the velocity network is implemented as a classical-to-quantum pipeline terminating in a parameterized quantum circuit (PQC). The PQC is realized as a differentiable JAX statevector simulator, enabling end-to-end automatic differentiation through the classical PINN, the quantum circuit, and the physics-informed loss. On a laterally varying velocity benchmark, the quantum hybrid reaches lower L1 velocity error than a classical FBPINN baseline in approximately $8\times$ fewer training iterations, using $\sim$33\% fewer trainable parameters, and outperforms all 15 classical hyperparameter variants we tested. A second benchmark (checkerboard) confirms the classical inversion pipeline; quantum hybrid results there are forthcoming. Our framework is broadly applicable to wave-based inverse problems beyond geophysics, including medical ultrasound and non-destructive evaluation.
\end{abstract}
```

- [ ] **Step 2: Build and count words**

Run: `./tex/compile.sh` (succeeds)
Run: `grep -A 30 'begin{abstract}' tex/main.tex | sed -n '/begin{abstract}/,/end{abstract}/p' | wc -w`
Expected: 150-200 (allow slack for LaTeX markup).

- [ ] **Step 3: Commit**

```bash
git add tex/main.tex
git commit -m "Draft Abstract"
```

---

### Task 3.2: Rewrite Introduction overview + contributions

**Files:**
- Modify: `tex/main.tex` lines 58-76 approximately (from blue "Need to review" marker through the "remainder of this paper" paragraph)

- [ ] **Step 1: Remove the blue "Need to review from here" marker**

Delete line 58: `\textcolor{blue}{Need to review from here}`.

- [ ] **Step 2: Rewrite the "In this work, we present..." paragraph**

Replace lines 60-66 with a cleaner version that:
- Drops "QFBPINNs" (our name changed to "quanFWI" / "hybrid quantum-classical FBPINN")
- Aligns with the final claim (8× convergence + fewer parameters)
- Is shorter (~100 words)

Replacement:

```tex
In this work we present a hybrid quantum-classical FBPINN for acoustic full waveform inversion. The framework retains classical fully-connected networks for the wavefield representation across decomposed subdomains and introduces a parameterized quantum circuit into the velocity network via a classical-to-quantum pipeline. The quantum circuit is implemented as a JAX-native statevector simulator, allowing end-to-end automatic differentiation through the hybrid pipeline without the overhead of parameter-shift rules. This design combines the scalability advantages of FBPINN domain decomposition with the structured, bounded expressivity of a PQC-based velocity parameterization, yielding substantially faster convergence than classical FBPINNs at comparable or smaller parameter counts.
```

- [ ] **Step 3: Rewrite the contributions list**

Replace lines 68-73 with three tight bullets:

```tex
The main contributions of this paper are:
\begin{enumerate}
    \item A hybrid quantum-classical FBPINN for acoustic FWI, integrating a parameterized quantum circuit into the velocity network within a domain-decomposed physics-informed framework.
    \item An end-to-end differentiable JAX statevector implementation that embeds the PQC into the FBPINN training loop without parameter-shift overhead, verified numerically against PennyLane's \texttt{default.qubit} backend to \texttt{float32} precision.
    \item A demonstration on the Rasht laterally-varying velocity benchmark showing $\sim$$8\times$ faster convergence in training iterations and lower final L1 velocity error than a classical FBPINN baseline and 15 classical hyperparameter variants, with $\sim$33\% fewer trainable parameters.
\end{enumerate}
```

- [ ] **Step 4: Rewrite the "The remainder of this paper is organized as follows" paragraph**

Replace line 75 (one paragraph) with Nature-style text that points to the new section order:

```tex
The remainder of this paper is organized as follows. Section~\ref{sec:results} presents the inversion results on two benchmark problems and the comparison between classical and quantum hybrid models. Section~\ref{sec:discussion} discusses mechanism, limitations, and outlook. Section~\ref{sec:method} provides full methodological details of the FBPINN framework, the parameterized quantum circuit, the hybrid architecture, and the training setup.
```

- [ ] **Step 5: Build**

Run: `./tex/compile.sh`
Expected: succeeds, refs resolve.

- [ ] **Step 6: Commit**

```bash
git add tex/main.tex
git commit -m "Rewrite Intro overview, contributions, and section-overview paragraphs"
```

---

### Task 3.3: Draft Conclusions

**Files:**
- Modify: `tex/main.tex` (currently `\section{Conclusions}\lipsum[8]`)

**What goes here (~150 words):**
- Restate contribution in one sentence.
- Restate main result in one sentence.
- Point to limitations and outlook (already in Discussion; brief recap here).
- Close with broader relevance.

- [ ] **Step 1: Replace `\lipsum[8]` with real conclusions**

```tex
\section{Conclusions}

We introduced a hybrid quantum-classical FBPINN for acoustic full waveform inversion, integrating a parameterized quantum circuit into the velocity network within a domain-decomposed physics-informed framework. The approach is implemented as an end-to-end differentiable JAX program, avoiding parameter-shift-rule overhead and enabling efficient joint optimization of classical PINN parameters and quantum circuit angles. On the Rasht velocity benchmark, the hybrid model converges in $\sim$$8\times$ fewer training iterations than a classical FBPINN baseline---and than 15 classical hyperparameter variants---while using $\sim$33\% fewer trainable parameters. Key limitations include reliance on classical statevector simulation, a 2D acoustic setup, and a single-source configuration; extension to elastic, 3D, and multi-source inversion, as well as validation on near-term quantum hardware, are the natural next steps. The framework generalizes beyond geophysics to any wave-based inverse problem amenable to physics-informed learning.
```

- [ ] **Step 2: Build**

Run: `./tex/compile.sh`

- [ ] **Step 3: Commit**

```bash
git add tex/main.tex
git commit -m "Draft Conclusions"
```

---

### **CHECKPOINT 3 — user review**

**Assistant reports:** "Phase 3 done — Abstract + rewritten Intro + Conclusions. Please read the paper end-to-end now. Things to scrutinize: (1) does the Abstract match the Results claims exactly? (2) does the Intro overview paragraph (lines 60-ish in the draft) reflect your preferred emphasis? (3) does the Conclusions section overclaim? Tell me if any word-level framing needs to change."

**User reads end-to-end, requests corrections.** Fix and re-commit before Phase 4.

---

## Phase 4 — Methods

**Goal:** Fill the currently-empty Methods subsections, polish existing drafted Methods content.

### Task 4.1: Draft §6.2 Physics-informed neural networks

**Files:**
- Modify: `tex/main.tex` (Methods subsection `\subsection{Physics-informed neural networks}`)

**What goes here (~250 words):**
- General PINN formulation: neural net approximates PDE solution, loss = residual + data terms.
- The PDE residual loss equation (already in the draft as lines 199-207 — reuse).
- The composite loss equation (already in the draft lines 209-215 — reuse).
- Short note on spectral bias and why it matters for FWI.

- [ ] **Step 1: Draft the subsection**

The existing equations on lines 199-215 stay. Add surrounding prose:

```tex
\subsection{Physics-informed neural networks}
\label{sec:methods_pinn}

Physics-informed neural networks (PINNs)~\cite{RAISSI2019686} approximate solutions of partial differential equations by a neural network $u_\theta(\mathbf{x})$ and train $\theta$ to minimize the PDE residual at a set of collocation points $\{x_i\}_{i=1}^{N}$ in the domain. Writing the governing PDE in residual form $\mathcal{N}[u](\mathbf{x}) = 0$, the PDE loss is

\begin{equation}
    \mathcal{L}_\text{PDE}
    = \frac{1}{N} \sum_{i=1}^{N}
    \left| \mathcal{N}[u_\theta(\mathbf{x}_i)] \right|^{2}.
\end{equation}

Observations and boundary/initial conditions are added as additional weighted terms,

\begin{equation}
    \mathcal{L} = \lambda_\text{PDE}\,\mathcal{L}_\text{PDE}
    + \lambda_\text{BC}\,\mathcal{L}_\text{BC}
    + \lambda_\text{IC}\,\mathcal{L}_\text{IC}
    + \lambda_\text{data}\,\mathcal{L}_\text{data},
\end{equation}

where the derivatives in $\mathcal{L}_\text{PDE}$ are computed by automatic differentiation of the network with respect to its inputs.

Classical PINNs are known to suffer from spectral bias~\cite{rahaman2019spectral}---a tendency to learn low-frequency features before high-frequency ones---which severely limits their effectiveness in multi-scale wave problems such as FWI, where both the velocity field and the wavefield exhibit sharp transitions and oscillatory structure.
```

- [ ] **Step 2: Add the `rahaman2019spectral` citation to references.bib**

Append:

```bibtex
@inproceedings{rahaman2019spectral,
  title={On the spectral bias of neural networks},
  author={Rahaman, Nasim and Baratin, Aristide and Arpit, Devansh and Draxler, Felix and Lin, Min and Hamprecht, Fred A and Bengio, Yoshua and Courville, Aaron},
  booktitle={International Conference on Machine Learning (ICML)},
  pages={5301--5310},
  year={2019}
}
```

- [ ] **Step 3: Build**

Run: `./tex/compile.sh`
Expected: succeeds, `rahaman2019spectral` resolves.

- [ ] **Step 4: Commit**

```bash
git add tex/main.tex tex/references.bib
git commit -m "Draft Methods §6.2 PINN subsection"
```

---

### Task 4.2: Draft §6.3 Finite-basis physics-informed neural networks

**Files:**
- Modify: `tex/main.tex`

**What goes here (~300 words):**
- FBPINN idea: global domain partitioned into overlapping subdomains, each with a local network + smooth partition-of-unity weighting.
- Formula: $u(\mathbf{x}) = \sum_{k=1}^{K} w_k(\mathbf{x}) \, u_{\theta_k}(\mathbf{x})$ where $w_k$ are cosine window functions.
- Benefit 1: mitigates spectral bias by localizing each network.
- Benefit 2: embarrassingly parallel over subdomains.
- Reference Moseley2023.

- [ ] **Step 1: Draft the subsection**

```tex
\subsection{Finite-basis physics-informed neural networks}
\label{sec:methods_fbpinn}

To overcome the spectral bias limitation of standard PINNs, we adopt the Finite-Basis Physics-Informed Neural Network (FBPINN) framework of Moseley et al.~\cite{Moseley2023}. The global domain $\Omega$ is partitioned into $K$ overlapping rectangular subdomains $\{\Omega_k\}$, each equipped with an independent neural network $u_{\theta_k}(\mathbf{x})$. The global solution is then reconstructed as a partition-of-unity sum

\begin{equation}
    u(\mathbf{x}) = \sum_{k=1}^{K} w_k(\mathbf{x})\, u_{\theta_k}(\mathbf{x}),
\end{equation}

where the window functions $w_k$ are smooth cosine bumps supported on $\Omega_k$ with
$\sum_k w_k(\mathbf{x}) = 1$ for all $\mathbf{x} \in \Omega$. Each local network only needs to represent the solution's behavior on its own subdomain, which has two consequences: (i) the local frequency content each network must learn is bounded by the subdomain size, alleviating spectral bias; and (ii) training is embarrassingly parallel across subdomains, enabling batched evaluation via JAX's \texttt{vmap}. The physics-informed loss is evaluated on the reconstructed global solution, so gradients flow through the window-weighted sum back to the individual subdomain networks.

For the wavefield $\phi(x,z,t)$ we use a three-dimensional subdomain decomposition $(n_x, n_z, n_t)$, e.g. $5\!\times\!3\!\times\!5$ in the Rasht baseline. The velocity network $c(x,z)$ is not decomposed---it is a single global network---since the velocity field typically varies more smoothly than the wavefield and does not require local representation.
```

- [ ] **Step 2: Build**

Run: `./tex/compile.sh`

- [ ] **Step 3: Commit**

```bash
git add tex/main.tex
git commit -m "Draft Methods §6.3 FBPINN subsection"
```

---

### Task 4.3: Rewrite §6.5 Hybrid quantum-classical architecture

**Files:**
- Modify: `tex/main.tex` existing subsection around current lines 490-695 (after reorder)

**What goes here (~400 words):**
Replace the current mix of figure + equations + bolded placeholder note with a clean description.

Components:
- Two parallel networks (wavefield φ, velocity c).
- Wavefield network: classical FC (no quantum), domain-decomposed.
- Velocity network (the hybrid): classical embedding layer → angle-encoded input to n-qubit PQC → Pauli-Z measurement → classical output scaling.
- Composite loss equation (already in draft).
- Reference Figure F1 (`fig:hybrid_arch`).

- [ ] **Step 1: Find and remove the bolded placeholder note**

Delete the line: `\textbf{the velocity isn't stored as a grid — it's stored as network weights that produce the velocity when evaluated. A raw grid of values can't be directly assigned to network weights.}`

- [ ] **Step 2: Draft the architecture prose**

Replace the current lead-in prose (the two paragraphs starting "Figure~\ref{fig:hybrid_arch} presents...") with:

```tex
\subsection{Hybrid quantum-classical architecture}
\label{sec:hybrid_network}

The hybrid architecture (\autoref{fig:hybrid_arch}) couples two neural networks with physics-informed constraints. The first network $\mathcal{N}_\phi(x,z,t)$ represents the wavefield and is a classical fully-connected network decomposed over subdomains as described in \ref{sec:methods_fbpinn}; it does not involve a quantum circuit. The second network represents the velocity field and is the component where the quantum circuit enters.

The velocity network is a classical-to-quantum pipeline. A classical fully-connected sub-network $\mathcal{N}_c: \mathbb{R}^2 \to \mathbb{R}^{n}$ maps each spatial coordinate $(x,z)$ to an $n$-dimensional latent vector, where $n$ is the number of qubits in the PQC. This latent vector is angle-encoded into the initial quantum state via $R_y(\beta_i x_i)$ rotations (Methods~\ref{sec:methods_pqc}). The PQC $\mathcal{Q}_c$ then evolves the state through $L$ variational layers and returns a scalar via Pauli-$Z$ measurement averaged over qubits. A final classical scaling and shifting layer maps this $[-1,1]$ output to physical velocity units:

\begin{equation}
    c(x,z) = c_\text{bg} + A \cdot \mathcal{Q}_c(\mathcal{N}_c(x,z); \boldsymbol{\beta}, \boldsymbol{\theta}),
\end{equation}

where $c_\text{bg}$ is a background velocity and $A$ is an amplitude scale learned jointly with the network and circuit parameters. This gives the velocity network three sets of trainable parameters: the classical weights $\theta_c$ of $\mathcal{N}_c$, the embedding scale $\boldsymbol{\beta}$, and the variational angles $\boldsymbol{\theta}$.

The full training objective is

\begin{equation}
\mathcal{L}_\text{total}
= \lambda_\text{data}\,\mathcal{R}_\text{data}(\phi)
+ \lambda_\text{PDE}\,\mathcal{R}_\text{PDE}(\phi, c)
+ \lambda_\text{BC}\,\mathcal{R}_\text{BC}(\phi)
+ \lambda_\text{IC}\,\mathcal{R}_\text{IC}(\phi),
\end{equation}

where $\mathcal{R}_\text{data}$ is the seismogram misfit between predicted and observed wavefields at the receiver locations, $\mathcal{R}_\text{PDE}$ enforces the source-free acoustic wave equation (Methods~\ref{sec:methods_wave}) via automatic differentiation, and $\mathcal{R}_\text{BC}, \mathcal{R}_\text{IC}$ impose the free-surface and initial-condition constraints. All parameters---classical PINN weights, embedding scales, and quantum gate angles---are optimized jointly by Adam gradient descent through the full differentiable pipeline.
```

- [ ] **Step 3: Verify the existing hybrid_arch figure remains and the caption is sensible**

The existing `fig:hybrid_arch` tikz block (current lines 614-694) stays. Trim its caption if it's verbose.

- [ ] **Step 4: Build**

Run: `./tex/compile.sh`
Expected: succeeds, all refs resolve.

- [ ] **Step 5: Commit**

```bash
git add tex/main.tex
git commit -m "Rewrite Methods §6.5 hybrid architecture subsection"
```

---

### Task 4.4: Draft §6.7 Training setup

**Files:**
- Modify: `tex/main.tex`

**What goes here (~200 words):**
- Optimizer, batch sizes, training length for each benchmark.
- Hardware.
- Loss weights (point to Table T1).
- Note on JAX + JIT.

- [ ] **Step 1: Draft**

```tex
\subsection{Training setup}
\label{sec:methods_training}

All training was performed with the Adam optimizer at learning rate $10^{-4}$ (baseline; see Supplementary Table~\ref{tab:hyperparams_full} for variants). Each training step evaluates the composite loss on a batch of collocation points drawn from the subdomains, combined with the full set of observation points (seismograms, initial conditions, boundary conditions). Collocation batch size is $(n_x, n_z, n_t) = (40,40,40)$ per subdomain; test-set evaluation uses $(50,50,50)$. The classical baseline is trained for 500{,}000 iterations; the quantum hybrid is trained for 65{,}000 iterations on the Rasht benchmark. Loss weights are fixed throughout training at $\lambda_\text{PDE}=0.1$, $\lambda_\text{IC}=1.0$ (applied equally to both initial-condition terms), $\lambda_\text{seis}=1.0$, $\lambda_\text{BC}=0.1$.

Training uses JAX's \texttt{jit} + \texttt{vmap} transformations to compile the update step into a single XLA program and batch evaluation over subdomains. The full update step---classical network forward/backward, PQC statevector evolution, and loss computation---executes as a single GPU kernel. Training was performed on a single NVIDIA GeForce RTX 5090 GPU. For the quantum hybrid, memory footprint is dominated by the classical wavefield network; the 4-qubit PQC statevector contributes negligibly ($2^4 = 16$ complex entries per forward pass).
```

- [ ] **Step 2: Build**

Run: `./tex/compile.sh`

- [ ] **Step 3: Commit**

```bash
git add tex/main.tex
git commit -m "Draft Methods §6.7 Training setup"
```

---

### Task 4.5: Draft §6.8 Data

**Files:**
- Modify: `tex/main.tex`

**What goes here (~200 words):**
- SPECFEM2D synthetic data generation.
- Rasht and Checkerboard specifics: domain size, source, receivers, time window, sampling.
- Reference to where data + scripts live in the repo.

- [ ] **Step 1: Draft**

```tex
\subsection{Data}
\label{sec:methods_data}

Synthetic seismic data for both benchmarks were generated with SPECFEM2D~\cite{komatitsch1998spectral}, a spectral-element forward solver, using the configurations distributed with the project repository (\texttt{data/rasht/specfem/} and \texttt{data/checkerboard/specfem/}).

\paragraph{Rasht benchmark.}
A 1.3 km $\times$ 0.45 km domain with a localized velocity anomaly (amplitude $\sim$$0.4$ km/s above a 2.7 km/s background, spatial extent $\sim$0.4 km laterally and $\sim$0.15 km vertically) centered near $(0.95, 0.2)$ km. One source at $(0.35, 0.19)$ km emits a Gaussian wavelet; 20 receivers are deployed along $x = 1.15$ km at 22.5 m spacing. We record 10 s of wave propagation, subsampled every 100 SPECFEM timesteps for the seismogram loss, and we use wavefield snapshots at $t = 0.10$ s and $t = 0.115$ s as initial conditions for the physics-informed loss.

\paragraph{Checkerboard benchmark.}
Same domain geometry and source/receiver layout as Rasht, but with a $3\!\times\!3$ checkerboard velocity pattern of alternating $\pm$$0.4$ km/s perturbations on a $3.0$ km/s background, centered in the region $[0.7, 1.05]\ \text{km} \times [0.1, 0.3]\ \text{km}$.

Coordinates are scaled by $L_x = L_z = 3$ km in all training computations.
```

- [ ] **Step 2: Add SPECFEM citation if missing**

Check: `grep komatitsch1998spectral tex/references.bib`
If missing, append:

```bibtex
@article{komatitsch1998spectral,
  title={The spectral element method: An efficient tool to simulate the seismic response of 2D and 3D geological structures},
  author={Komatitsch, Dimitri and Vilotte, Jean-Pierre},
  journal={Bulletin of the Seismological Society of America},
  volume={88},
  number={2},
  pages={368--392},
  year={1998},
  publisher={Seismological Society of America}
}
```

- [ ] **Step 3: Build**

Run: `./tex/compile.sh`

- [ ] **Step 4: Commit**

```bash
git add tex/main.tex tex/references.bib
git commit -m "Draft Methods §6.8 Data subsection"
```

---

### Task 4.6: Light polish on existing Methods subsections

**Files:**
- Modify: `tex/main.tex`

**Existing subsections to polish:**
- §6.1 Acoustic wave equation (currently lines 81-188 — mostly solid, light polish only)
- §6.4 Parameterized quantum circuits (currently lines 220-428 from the old "Quantum computing background" — already thorough, just polish)
- §6.6 JAX-based statevector simulator (currently ~lines 482-488 — already concise, verify)

- [ ] **Step 1: §6.1 Acoustic wave equation polish**

Read the existing subsection. Confirm it flows and that equation numbering is still correct after the document reorder. Light touch only. No content change unless something is wrong.

- [ ] **Step 2: §6.4 PQC polish**

Read. Ensure the `fig:pqc_architecture` reference on line ~336 still resolves (it should, after Task 1.3). Check that citations `benedetti2019parameterized` etc. now render as numeric. Trim any verbose paragraphs.

- [ ] **Step 3: §6.6 JAX simulator polish**

Read. Ensure it references `fig:pqc_gradient_agreement` (the supplementary figure) correctly. The figure is in the appendix.

- [ ] **Step 4: Build**

Run: `./tex/compile.sh 2>&1 | grep -iE '(undefined|multiply defined)'`
Expected: no matches.

- [ ] **Step 5: Commit**

```bash
git add tex/main.tex
git commit -m "Light polish on existing Methods subsections"
```

---

### **CHECKPOINT 4 — user review**

**Assistant reports:** "Phase 4 done — Methods complete. The full paper now compiles with no placeholders. Please read Methods end-to-end. Particular concerns: (1) Did I get your Adam/LR/batch sizes right? (2) The PQC output-scaling formula in §6.5 is my best guess from your network architecture — verify it matches `fbpinns/networks.py::HybridQuantumFCN`. (3) Is the Data subsection's SPECFEM setup accurate (source position, receiver count, recording window)?"

**User reviews Methods.** Corrections → revisions → re-commit.

---

## Phase 5 — Polish & submission prep

**Goal:** Word count, figure captions, data availability, cover letter, final build.

### Task 5.1: Word count check

**Files:**
- None modified (just check)

- [ ] **Step 1: Count main body words (excluding Methods, refs, captions)**

```bash
cd tex
# Extract main-body text (Intro through Conclusions, excluding Methods)
awk '/\\section\{Introduction\}/,/\\section\{Methods\}/' main.tex \
  | sed -n '/\\section\{Introduction\}/,/\\section\{Methods\}/p' \
  | detex -l -n \
  | wc -w
```

Alternative: `texcount main.tex -sum -inc` (install texcount if needed via `apt install texlive-extra-utils`)

Expected: ≤ 5000 words in main body (abstract + intro + results + discussion + conclusions).

- [ ] **Step 2: If over, trim where**

If count exceeds 5000:
- Start with Discussion D3/D4 (limitations/outlook) — tightest place to cut 100-200 words.
- Then intro lit-review paragraphs (lines 52-56 have compact references; unlikely to shrink further).

- [ ] **Step 3: Report count to user**

---

### Task 5.2: Figure caption audit

**Files:**
- Modify: `tex/main.tex` (figure captions throughout)

**Check each figure caption is:**
- Self-contained (reader understands without reading body text)
- ≤ 100 words
- Identifies all panels (a, b, c, d)
- States what the reader should take away

- [ ] **Step 1: Audit F1 `fig:hybrid_arch` caption**

Currently ends at line 685 or similar. Confirm it names: networks, circuits, residuals.

- [ ] **Step 2: Audit F2 `fig:pqc_architecture` caption**

Currently "Multilayer PQC $\mathcal{P}$" — expand to include what the encoding layer, variational layers, and measurement do.

Replacement:

```tex
\caption{Multilayer parameterized quantum circuit $\mathcal{P}$. Each input feature $x_i$ is encoded via an $R_y(\theta_{i,1})$ rotation (encoding layer). Variational layers of single-qubit rotations ($R_x$, $R_y$, $R_z$) followed by a CNOT ladder are repeated $N_Q$ times. Each qubit is measured in the Pauli-$Z$ basis, and the outputs are averaged to yield a scalar in $[-1, 1]$.}
```

- [ ] **Step 3: Audit F3 `fig:rasht_inversion` caption** (already drafted in Task 2.2). Re-read and trim if needed.

- [ ] **Step 4: Audit F5 `fig:checkerboard_inversion` caption** (already drafted in Task 2.5). Re-read.

- [ ] **Step 5: Audit supplementary SF1 `fig:pqc_gradient_agreement` caption** (already drafted). Re-read.

- [ ] **Step 6: Build and visually inspect captions in PDF**

Run: `./tex/compile.sh` and open `tex/main.pdf`. Check each figure's caption renders as expected.

- [ ] **Step 7: Commit**

```bash
git add tex/main.tex
git commit -m "Polish figure captions for self-containment"
```

---

### Task 5.3: Verify Data availability and Code availability statements

**Files:**
- Modify: `tex/main.tex`

- [ ] **Step 1: Verify Data availability statement** (added in Task 1.8 Step 4)

Content should state where the synthetic data live and how to reproduce. Confirm it mentions `data/rasht/` and `data/checkerboard/`.

- [ ] **Step 2: Verify Code availability statement** (existing, around line 858)

Existing text mentions `https://github.com/x-repos/quFWI`. Confirm:
- Repo URL correct (flagged assumption in spec §6.4)
- Mentions FBPINNs reference repo
- Mentions the PINN-FWI reference repo

Add a sentence about how to reproduce (`compile.sh` for tex, `pip install -e .` for code):

```tex
\section*{Code availability}

The JAX-based quantum simulator and quanFWI framework presented here are available at \url{https://github.com/x-repos/quFWI}. Our implementation builds upon the FBPINNs framework~\cite{Moseley2023} (\url{https://github.com/benmoseley/FBPINNs}) with the PINN-based FWI formulation of Rasht-Behesht et al.~\cite{rahst} (\url{https://github.com/maziarash/PINN-FWI}). Running \texttt{pip install -e .} from the repository root installs the project and its JAX dependencies; training scripts are provided under \texttt{scripts/rasht/} and \texttt{scripts/checkerboard/}.
```

- [ ] **Step 3: Build**

Run: `./tex/compile.sh`

- [ ] **Step 4: Commit**

```bash
git add tex/main.tex
git commit -m "Polish Data and Code availability statements"
```

---

### Task 5.4: Author Contributions + Conflicts of Interest placeholders

**Files:**
- Modify: `tex/main.tex`

**User fills these (flagged in spec §6.4). Draft placeholder content with clear `% FILL` markers.**

- [ ] **Step 1: Expand `\section*{Author Contributions}`**

```tex
\section*{Author contributions}

% FILL: author contribution statement. Typical format:
% H.A.N. conceived the project, designed the hybrid architecture, implemented the JAX codebase, ran the experiments, analyzed the results, and wrote the manuscript.
% D.V. contributed to [...].
% A.T. contributed to [...].
% All authors reviewed and approved the final manuscript.
```

- [ ] **Step 2: Expand `\section*{Conflicts of Interest}`**

```tex
\section*{Conflicts of interest}

% FILL: declare any financial or non-financial competing interests, or write:
The authors declare no competing interests.
```

- [ ] **Step 3: Commit**

```bash
git add tex/main.tex
git commit -m "Add placeholders for Author Contributions and Conflicts of Interest"
```

---

### Task 5.5: Cover letter draft

**Files:**
- Create: `tex/cover_letter.tex`

- [ ] **Step 1: Create cover letter**

```tex
\documentclass[11pt,a4paper]{letter}
\usepackage[a4paper,margin=1in]{geometry}
\usepackage{hyperref}

\signature{Hoang Anh Nguyen}
\address{Department of Geophysics \\ Colorado School of Mines \\ Golden, CO 80401, USA \\ hoanganh\_nguyen@mines.edu}

\begin{document}

\begin{letter}{Editorial Office \\ Communications AI \& Computing \\ Nature Portfolio}

\opening{Dear Editor,}

We are pleased to submit our manuscript, ``quanFWI: Hybrid Quantum-Classical Finite Basis Physics-Informed Neural Networks for Wave Propagation and Full Waveform Inversion,'' for consideration in \emph{Communications AI \& Computing}.

The manuscript presents the first integration of parameterized quantum circuits into domain-decomposed physics-informed neural networks for seismic inverse problems. Our main result is that a hybrid quantum-classical velocity network achieves lower L1 velocity error than a classical FBPINN baseline---and than 15 classical hyperparameter variants---in approximately $8\times$ fewer training iterations, while using $\sim$33\% fewer trainable parameters, on a laterally varying velocity benchmark. The quantum circuit is implemented as a JAX-native differentiable statevector simulator, enabling end-to-end automatic differentiation through the classical PINN, the quantum circuit, and the physics-informed loss.

We believe this work falls squarely within the scope of \emph{Communications AI \& Computing}, particularly the \emph{Hybrid quantum-classical systems}, \emph{AI for scientific discovery}, and \emph{Computational modeling in physics} areas. The manuscript represents a concrete demonstration of how near-term quantum parameterization can improve an important scientific inverse problem---full waveform inversion---with broader applicability to medical ultrasound, photoacoustic imaging, and non-destructive evaluation.

All data, code, and reproduction scripts are openly available at \url{https://github.com/x-repos/quFWI}.

We confirm that this manuscript has not been published elsewhere and is not under consideration by another journal. We suggest the following reviewers with expertise in this area: % FILL: suggest 3-4 reviewers.

Thank you for considering our submission.

\closing{Sincerely,}

\end{letter}

\end{document}
```

- [ ] **Step 2: Build the cover letter**

```bash
cd tex && latexmk -pdf cover_letter.tex && latexmk -c cover_letter.tex
```

Expected: `tex/cover_letter.pdf` exists.

- [ ] **Step 3: Commit**

```bash
git add tex/cover_letter.tex
git commit -m "Draft cover letter for Comm. AI & Computing submission"
```

---

### Task 5.6: Full build + proofread pass

**Files:**
- None modified unless issues found

- [ ] **Step 1: Clean build from scratch**

```bash
cd tex
latexmk -C main.tex  # full clean
./compile.sh         # rebuild
```

Expected: succeeds, clean `main.pdf`.

- [ ] **Step 2: Check for stale warnings**

```bash
cd tex && grep -iE '(undefined|multiply defined|warning.*overfull|warning.*underfull)' main.log | head -20
```

Expected: ideally no undefined, no multiply-defined. Some overfull/underfull warnings are normal; fix the worst ones (those > 5pt).

- [ ] **Step 3: Open main.pdf and proofread**

Visual check:
- Title page renders
- Abstract is 150-200 words
- Section order: Intro → Results → Discussion → Conclusions → Methods → Data/Code Availability → Ack → Author Contributions → Conflicts → References
- All 5 main figures render (F1 schematic, F2 PQC circuit, F3 Rasht, F4 [if exists], F5 checkerboard)
- All 6 figures (F4 TBD — note status) — wait, F4 (seismogram fit) was an open question in spec §3.4. If not generated, fine; omit from main text but note in Phase 5 report.
- Appendix has supplementary figures and tables
- Page count: aim for 20-30 pages

- [ ] **Step 4: Commit any fixes**

```bash
git add -A
git commit -m "Fix build warnings and proofreading issues"
```

---

### Task 5.7: Final deliverables check

- [ ] **Step 1: Confirm deliverables exist**

```bash
ls -la tex/main.pdf tex/cover_letter.pdf
```

- [ ] **Step 2: Confirm `references.bib` is clean**

```bash
cd tex
./compile.sh 2>&1 | grep -iE '(unused|not used)' | head -10
```

Remove unused entries from `references.bib` if biber reports any. Biblatex doesn't usually warn on unused entries by default — this is belt-and-braces.

- [ ] **Step 3: Confirm `tex/figures/` has no orphans**

```bash
cd tex
for f in figures/*; do
  basename=$(basename "$f")
  if ! grep -q "$basename" main.tex; then
    echo "ORPHAN: $f"
  fi
done
```

Expected: no output (all figures in the directory are referenced).

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit --allow-empty -m "Submission-ready: quanFWI manuscript for Comm. AI & Computing"
```

---

### **CHECKPOINT 5 — final user review**

**Assistant reports:** "Submission-ready. Final deliverables:

- `tex/main.pdf` — manuscript
- `tex/cover_letter.pdf` — cover letter (needs reviewer suggestions filled in)
- `tex/references.bib` — bibliography
- Git commits on `main`, linear history

User-owned items still to fill before submitting to the portal:
1. Author Contributions section body (marked `% FILL` in main.tex)
2. Reviewer suggestions in cover letter (marked `% FILL`)
3. Conflicts of Interest: confirm 'no competing interests' applies, else edit
4. Verify the github repo `https://github.com/x-repos/quFWI` is public and reproducible by the time of submission
5. Open questions from earlier (seismogram-fit figure F4 status — include or leave for revision?)

Anything wrong? Any final edits before you hand this off to the submission portal?"

---

## Self-review notes (for the plan-writing assistant)

- **Spec coverage:** Each spec section is covered — §1 scope (Phase 1 Task 1.2 removes Borehole/Teleseismic), §2 structure (Phase 1 Task 1.8), §3 figures/tables (Phase 1 Tasks 1.1, 1.6; Phase 2 Task 2.2, 2.5; Phase 5 Task 5.2), §4 drafting order (Phases 2-4 follow spec order), §5 framing decisions (Tasks 2.3, 3.1, 3.2, 3.3 all reference them), §6 workflow (per-task commits throughout).
- **Placeholder scan:** All prose drafts have actual text. Data availability (Task 1.8 Step 4) has full text. Author Contributions / Conflicts of Interest intentionally have `% FILL` markers — these are user-owned per spec §6.4 assumptions; flagged as such in the Phase 5 report.
- **Type consistency:** References used: `\ref{fig:hybrid_arch}`, `\ref{fig:pqc_architecture}`, `\ref{fig:rasht_inversion}`, `\ref{fig:checkerboard_inversion}`, `\ref{tab:hyperparams_main}`, `\ref{tab:hyperparams_full}`, `\ref{tab:hyperparams_forward}`, `\ref{sec:results}`, `\ref{sec:discussion}`, `\ref{sec:method}`, `\ref{sec:methods_pinn}`, `\ref{sec:methods_fbpinn}`, `\ref{sec:hybrid_network}`, `\ref{sec:methods_training}`, `\ref{sec:methods_data}`, `\ref{sec:methods_wave}`. These are consistent across tasks.
- **Gaps:** F4 (seismogram fit) remains TBD — spec §3.4 flagged it as an open question; Phase 5 Task 5.6 reports status rather than requiring resolution. If the user wants F4 included, that becomes a new task; otherwise the manuscript can submit without it and reviewers may request it in revision.
