# Bayesian Logistic Regression with Laplace Approximation (3F8 Inference FTR)

This repository implements a Bayesian binary logistic regression model and applies the Laplace approximation for predictive inference.  
MAP estimation is performed using (1) an iterative gradient-ascent routine and (2) L-BFGS-B via SciPy, enabling a direct comparison of convergence behaviour and predictive performance.

## Features
- Bayesian logistic regression with Gaussian prior (L2 regularisation)
- MAP estimation:
  - Iterative gradient ascent
  - L-BFGS-B (quasi-Newton)
- Laplace approximation:
  - Hessian of the negative log-posterior
  - Predictive probability using the logistic-Gaussian approximation
- Metrics & visualisations:
  - Average log-likelihood traces (train/test)
  - Predictive contour plots / decision boundary visualisation

## Repository Structure
- `data/raw/` input data (`x.txt`, `y.txt`)
- `report/` LaTeX full technical report (FTR)
- `notebooks/` exploratory notebook(s) (optional)

## Setup
### Option A: Conda (recommended)
```bash
conda env create -f environment.yml
conda activate 3f8-ftr
