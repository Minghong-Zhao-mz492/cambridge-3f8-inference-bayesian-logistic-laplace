# Bayesian Logistic Regression with Laplace Approximation (Cambridge 3F8 Inference FTR)

This repository contains a complete, reproducible implementation of **Bayesian binary logistic regression** with an **RBF feature expansion**, using the **Laplace approximation** for approximate posterior inference and predictive probabilities.

The code follows the Full Technical Report (FTR) pipeline (Exercises **A–F**):
- obtain the **MAP** estimate by optimising the log-posterior,
- compute the **Laplace approximation** (Gaussian posterior around the MAP),
- perform **evidence-based hyper-parameter tuning** for the RBF basis via a **10×10 grid search** visualised with **heat maps**.

---

## Highlights
- Bayesian logistic regression with an isotropic Gaussian prior (L2 regularisation / weight decay)
- MAP solvers:
  - iterative gradient ascent (baseline / sanity check)
  - L-BFGS-B (SciPy) for fast, stable optimisation
- Laplace approximation:
  - Hessian of the negative log-posterior and posterior covariance via matrix inverse
  - predictive probability using the standard logistic–Gaussian approximation
- Evidence-based tuning:
  - Laplace-approximated log evidence computed over a 10×10 grid of \((\sigma_0^2, l)\)
  - heat map visualisation (coarse grid + optional refinement grid)
- Outputs:
  - log-likelihood traces (train/test)
  - contour plots of predictive probabilities / decision boundary
  - confusion matrices (as fractions)

---

## Repository Structure
- `data/`  
  Input dataset files (e.g. `X.txt`, `y.txt`).
- `report/`  
  LaTeX source for the Full Technical Report (FTR) and the Short Lab Report.
- `figures/`  
  Generated plots and figures used in the report (contours, heat maps, etc.).
- Top-level Python scripts (main runnable code for exercises):
  - `3f8_ftr_exercises_a_to_f.py` — end-to-end run for Exercises A–F  
  - `3F8_code_ftr_exercise_a_d.py` — earlier checkpoint for Exercises A–D  
  - `3F8_code_ftr_exercise_a_d_refined.py` — refined A–D version / experiments

---

## Environment / Setup

### Conda (recommended)
```bash
conda env create -f environment.yml
conda activate 3f8-ftr
```

### pip (alternative)
If you prefer pip, install the following packages (minimum):
- numpy
- matplotlib
- scipy
