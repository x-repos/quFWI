# Design: Commun. AI & Computing submission — quanFWI paper

**Date:** 2026-04-20
**Manuscript:** `qFWI/tex/main.tex`
**Target journal:** Communications AI & Computing (Nature Portfolio)
**Status:** Design approved; implementation not started.

---

## 1. Scope and claim

### 1.1 Target journal

Communications AI & Computing — Nature Portfolio, open-access. Scope covers "Hybrid quantum-classical systems," "AI for scientific discovery," "Computational modeling in physics." Publication criteria: novelty, strong evidence, FAIR data/code standards, sub-field importance.

Implies:
- Main text ≤ ~5000 words (excluding Methods, refs, captions)
- Up to ~10 display items
- Unstructured abstract ~150-200 words
- Nature section order: Introduction → Results → Discussion → Methods (at end)
- Numeric citation style
- Required Data Availability + Code Availability statements

### 1.2 Benchmarks

- **Rasht** (laterally varying velocity anomaly) — full quantum vs classical comparison. Main story.
- **Checkerboard** — classical-only inversion for now. Quantum results to be added in revision. Explicit caveat in text.

Dropped from scope: Borehole, Teleseismic (currently lipsum placeholders in draft).

### 1.3 Main claim

Hybrid quantum-classical velocity networks (PQC-based) converge in **~8× fewer training steps** than classical FBPINN velocity networks on FWI (Rasht benchmark), reaching lower final L1 velocity error with ~33% fewer parameters (~101k vs ~152k), and outperforming all 15+ classical hyperparameter variants.

Framing constraints (Section 4 decisions below):
- "Faster convergence in training iterations" / "improved optimization dynamics." **Not** "quantum speedup" or "quantum advantage."
- FBPINN is core to the contribution, not incidental.
- Cross-discipline applications (medical, NDE) get one sentence — no over-claiming.
- Statevector simulation is a deliberate design choice (Methods) *and* a limitation to acknowledge (Discussion).

### 1.4 Division of labor

Assistant drafts full prose. User edits and pushes back. Assistant never fabricates numbers — every quantitative claim in prose comes from existing figure/table/log. If a number is needed that doesn't exist, assistant asks.

### 1.5 Timeline

No hard deadline. Pace for quality.

---

## 2. Final paper structure

```
1. Title + authors + affiliations
2. Abstract (~150-200 words, unstructured)
3. Introduction
4. Results
   4.1  Architecture and training setup
   4.2  Benchmark 1 — Rasht velocity anomaly
        - Inversion quality (true/initial/inverted)
        - Convergence comparison (quantum vs classical)    [KEY FIGURE]
        - Hyperparameter robustness (15-variant sweep)
   4.3  Benchmark 2 — Checkerboard
        - Inversion quality (classical only)
        - Note: quantum on checkerboard pending
5. Discussion
6. Methods
   6.1  Acoustic wave equation (scalar potential formulation)
   6.2  PINN and FBPINN foundations
   6.3  Parameterized quantum circuits
   6.4  Hybrid quantum-classical architecture
   6.5  JAX statevector simulator
   6.6  Training setup (loss, optimizer, hardware)
   6.7  Data (SPECFEM2D synthetic, Rasht + Checkerboard)
7. Data availability
8. Code availability
9. References
10. Acknowledgments + Author contributions + Conflicts of interest
11. Supplementary information
    SF1  PQC gradient agreement with PennyLane
    SF2  Training loss curves by component (Rasht)
    SF3  Full hyperparameter variant L1 sweep
    SF4  Checkerboard loss decomposition
    ST1  Full hyperparameter variant table
    ST2  Forward-modeling (known-velocity) configurations
```

### 2.1 Restructures vs current draft

- Methodology (currently §3) moves to end as "Methods" (§6).
- Results section (currently all `\lipsum[1..6]`) expanded with two subsections.
- Three overlapping hyperparameter tables consolidated: one main-text FWI table (baseline + quantum), two supplementary tables (full variants, forward-modeling).
- Borehole / Teleseismic subsections removed.

---

## 3. Figure and table plan

### 3.1 Main text (target 6-8 display items)

| #  | Type   | Content                                                                                              | Source                                               | Status                     |
|----|--------|------------------------------------------------------------------------------------------------------|------------------------------------------------------|----------------------------|
| F1 | Figure | Hybrid QFBPINN architecture schematic                                                                 | tikz block in `main.tex`                              | drafted; polish caption    |
| F2 | Figure | PQC circuit diagram (quantikz)                                                                        | quantikz in `main.tex`                                | drafted; fix label         |
| F3 | Figure | Rasht (a) true / (b) initial / (c) inverted velocity + (d) L1 convergence (quantum vs classical)     | `tex/figures/l1_variants.pdf`                         | exists, keep as single     |
| F4 | Figure | Rasht seismogram fit (predicted vs observed)                                                          | **to confirm / generate**                             | TBD                        |
| F5 | Figure | Checkerboard (a) true / (b) inverted classical                                                        | `results/checkerboard/plots/true_and_inverted.png`    | exists, move to `tex/figures/` |
| T1 | Table  | Consolidated hyperparameters: Rasht FWI classical baseline + quantum                                  | merge current Tables 2+3                              | needs consolidation        |

### 3.2 Supplementary

| #   | Content                                                       | Source                                                |
|-----|---------------------------------------------------------------|-------------------------------------------------------|
| SF1 | PQC gradient agreement with PennyLane                         | `tex/figures/pqc_gradient_agreement.pdf`              |
| SF2 | Rasht: training loss curves by component                      | `results/rasht/plots/metrics_losses.png`              |
| SF3 | Rasht: hyperparameter variant L1 sweep (expanded)             | same as F3 but full variants                          |
| SF4 | Checkerboard: full loss decomposition                         | `results/checkerboard/plots/metrics_losses.png`       |
| ST1 | Full hyperparameter variant table (all 15+)                   | current Table 3                                       |
| ST2 | Forward-modeling (known-velocity) configurations              | current Table 1                                       |

### 3.3 Removed

Orphan figures not referenced in final layout: `figures/cat_momo_1.png`, `cat_momo_2.jpg`, `fig_a.png`, `fig_b.png`, `fig_c.png`, `velocity_field.png`, `training_loss_smoothed.png`, `metrics_l1_velocity.png` (the earlier-iteration versions).

### 3.4 Open questions

- **Seismogram-fit figure (F4):** verify one exists or generate. Reviewers expect this for FWI.
- Architecture schematic F1 stays as current hand-drawn tikz.
- F3 stays as single 4-panel figure.

---

## 4. Drafting order and review checkpoints

Order is chosen to front-load the parts where the claim framing matters most (Results + Discussion), so the user catches problems before large amounts of prose are written on a wrong premise.

### Phase 1 — Mechanical shell cleanup (no prose changes)

1. Remove unused figures from `tex/figures/`.
2. Add the 6 missing bibtex entries (quantum-subsection citations: `benedetti2019parameterized`, `bergholm2018pennylane`, `mitarai2018quantum`, `schuld2019evaluating`, `schuld2021machine`, `sim2019expressibility`).
3. Fix `fig: pqc` → `fig:pqc_architecture`; add `\label{}`s to every `\section` referenced from intro (resolves `sec:background`, `sec:method`, `sec:results`, `sec:discussion`).
4. Consolidate three hyperparameter tables → 1 main-text + 2 supplementary; resolve the multiply-defined `tab:hyperparams` label.
5. Switch biblatex from `apa` to `numeric-comp`.
6. Reorder document: Methodology → Methods at end.
7. Remove `\subsection{Borehold}` and `\subsection{Teleseismic}`.

(`\lipsum{...}` placeholders and the blue "Need to review from here" marker stay in place until their section is drafted in Phases 2-4; they get removed as prose lands.)

**Checkpoint 1:** Paper compiles, no undefined refs/cites, no duplicate labels. User signs off before prose drafting begins.

### Phase 2 — Results + Discussion

10. Draft Results §4.1 Architecture and training setup paragraph.
11. Draft Results §4.2 Rasht: inversion quality, convergence comparison, variant robustness.
12. Draft Results §4.3 Checkerboard: classical inversion, "quantum pending" note.
13. Draft Discussion.

**Checkpoint 2:** User reviews Results + Discussion. Wrong claims / inflated language / missing caveats flagged here, revised before abstract and intro are drafted on top.

### Phase 3 — Framing

14. Draft Abstract.
15. Rewrite Introduction overview paragraphs to match final story arc. (Existing lit-review paragraphs mostly survive; contribution list and section-overview paragraph get rewritten.)
16. Draft Conclusions.

**Checkpoint 3:** User reads full paper end-to-end.

### Phase 4 — Methods

17. Draft §6.2 PINN and FBPINN foundations (currently empty).
18. Draft / rewrite §6.4 Hybrid architecture (currently has a bolded placeholder note).
19. Light editing on §6.1, §6.3, §6.5 (existing drafted content).
20. Draft §6.6 Training setup, §6.7 Data.

**Checkpoint 4:** Methods review.

### Phase 5 — Polish & submission prep

21. Word-count check (main body ≤ 5000).
22. Figure captions complete and self-contained.
23. Data availability statement (new).
24. Author Contributions + Conflicts of Interest (user fills).
25. Full build, proofread.
26. Cover letter draft in `tex/cover_letter.tex` (separate build).

**Checkpoint 5:** Submission-ready.

---

## 5. Content framing decisions (approved)

| ID   | Decision                                                                                                                                                                                                  |
|------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 4.1  | One-line pitch: novelty framing + quantitative win ("first integration of VQCs into domain-decomposed PINNs for seismic inverse problems; ~8× faster convergence, ~33% fewer parameters"). |
| 4.2  | FBPINN is core to the contribution, not incidental scaffolding. Paper emphasizes combination of domain decomposition with quantum velocity network.                                                         |
| 4.3  | Cross-discipline applications (medical, photoacoustic, NDE) get one sentence — signal generality without over-claiming.                                                                                      |
| 4.4  | Statevector simulation framed both ways: deliberate design choice in Methods (enables end-to-end autodiff, avoids parameter-shift overhead); limitation in Discussion (hardware experiments are future work). |
| 4.5  | Convergence claim framed as "faster convergence in training iterations" / "improved optimization dynamics." **Not** "quantum speedup" or "quantum advantage."                                               |

---

## 6. Workflow, review mechanics, assumptions

### 6.1 Git workflow

Work committed to `main` branch (solo maintainer; checkpoints gate review). Each phase = one or a few commits. Only touches `tex/` and `docs/superpowers/specs/`. No changes to `fbpinns/`, `pqcs/`, `scripts/`, `results/`.

### 6.2 Review mechanics

At each checkpoint: build `main.pdf`, point user to section titles / page numbers, paste prose inline for fast pushback. Mechanical changes reported as brief summary.

### 6.3 Deliverables

- `tex/main.pdf` submission-ready
- All figures in `tex/figures/`, no orphans
- Clean `references.bib` — all cites resolve, no unused entries
- Data + Code availability statements complete
- Cover letter draft in `tex/cover_letter.tex`

### 6.4 Assumptions (flag if any are wrong)

- User owns all scientific claims. No fabricated numbers — assistant asks when a number is needed that doesn't exist.
- Author order stays Nguyen, Vashisth, Tura. User fills Author Contributions + Conflicts of Interest.
- ORCID line in title block is correct.
- `https://github.com/x-repos/quFWI` exists / will exist by submission.
- User handles actual journal submission (PDF upload, form, reviewer selection, OA fee).
- Communications AI & Computing author guide wins over this design if they conflict; assistant flags such changes at polish stage.

### 6.5 Risks

| Risk                                                                                         | Mitigation                                                                                                                                                     |
|----------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Checkerboard-quantum data may not arrive before submission.                                   | Submit with classical-only checkerboard + honest caveat sentence. Revision round is fine.                                                                      |
| Reviewer demands a real-hardware quantum experiment.                                          | Pre-empted in Discussion (decision 4.4). Frame simulation as deliberate.                                                                                       |
| Figure quality not print-ready (low-res, unclear legends).                                    | Flag per figure at polish stage; regenerate with plotting scripts in `results/*/pyplots/`.                                                                     |
| Current intro "contributions" list and section-overview paragraph promise a structure that no longer matches. | Phase 3 step 15 rewrites those two paragraphs.                                                                                                                  |
| Restructuring tables risks breaking downstream references.                                     | Build after each atomic change (Phase 1 item 4). Fix refs before moving on.                                                                                    |
