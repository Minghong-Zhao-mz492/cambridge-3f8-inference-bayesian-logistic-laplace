import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b
from pathlib import Path

# ============================================================
# 0) Load data + shuffle
# ============================================================
# NOTE: X is expected to be (N,2), y is expected to be (N,) with labels {0,1}
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
X = np.loadtxt(DATA_DIR / "X.txt")
y = np.loadtxt(DATA_DIR / "y.txt")

# Make sure y is 1D to avoid shape/broadcasting issues later
y = np.asarray(y).ravel()

# Shuffle data (keeps X and y aligned)
permutation = np.random.permutation(X.shape[0])
X = X[permutation, :]
y = y[permutation]


# ============================================================
# 1) Plotting utilities (data + log-likelihood curves)
# ============================================================
def plot_data_internal(X, y):
    """
    Create a scatter plot of the 2D dataset and return a mesh grid (xx, yy)
    used for contour plotting later.
    """
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    plt.figure()
    plt.xlim(xx.min(None), xx.max(None))
    plt.ylim(yy.min(None), yy.max(None))
    ax = plt.gca()

    # NOTE: labels here follow your original script (y==0 as "Class 1", y==1 as "Class 2")
    ax.plot(X[y == 0, 0], X[y == 0, 1], 'ro', label='Class 1')
    ax.plot(X[y == 1, 0], X[y == 1, 1], 'bo', label='Class 2')

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Plot data')
    plt.legend(loc='upper left', scatterpoints=1, numpoints=1)
    return xx, yy


def plot_data(X, y):
    """
    Show the scatter plot of the dataset.
    """
    _ = plot_data_internal(X, y)
    plt.show()


def plot_ll(ll):
    """
    Plot the average log-likelihood trace over optimization steps.
    """
    plt.figure()
    ax = plt.gca()
    plt.xlim(0, len(ll) + 2)
    plt.ylim(min(ll) - 0.1, max(ll) + 0.1)
    ax.plot(np.arange(1, len(ll) + 1), ll, 'r-')
    plt.xlabel('Steps')
    plt.ylabel('Average log-likelihood')
    plt.title('Plot Average Log-likelihood Curve')
    plt.show()


# Quick sanity plot of the raw data
plot_data(X, y)


# ============================================================
# 2) Train/test split
# ============================================================
n_train = 800
X_train = X[0:n_train, :]
X_test  = X[n_train:, :]
y_train = y[0:n_train].ravel()
y_test  = y[n_train:].ravel()


# ============================================================
# 3) Core logistic regression primitives
# ============================================================
def logistic(x):
    """
    Numerically-stable logistic sigmoid.
    We clip the input to avoid exp overflow.
    """
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def predict(X_tilde, w):
    """
    Predict p(y=1|x,w) for a design matrix X_tilde.
    """
    return logistic(np.dot(X_tilde, w))


def compute_average_ll(X_tilde, y, w):
    """
    Plugin average log-likelihood:
        mean_n [ y_n log p_n + (1-y_n) log (1-p_n) ]
    with clipping to avoid log(0).
    """
    y = np.asarray(y).ravel()
    output_prob = predict(X_tilde, w)

    eps = 1e-12
    output_prob = np.clip(output_prob, eps, 1 - eps)

    return np.mean(y * np.log(output_prob) + (1 - y) * np.log(1.0 - output_prob))


def get_x_tilde(X):
    """
    Add a bias column of ones:
        X_tilde = [1, X]
    """
    return np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)


def fit_w(X_tilde_train, y_train, X_tilde_test, y_test, n_steps, alpha):
    """
    (Coursework-style) gradient ascent for *likelihood only* (no prior term).
    Kept as-is from your earlier template; not used in the MAP+prior part below.
    """
    y_train = np.asarray(y_train).ravel()
    y_test  = np.asarray(y_test).ravel()

    w = np.random.randn(X_tilde_train.shape[1])
    ll_train = np.zeros(n_steps)
    ll_test = np.zeros(n_steps)

    for i in range(n_steps):
        sigmoid_value = predict(X_tilde_train, w)
        gradient = np.dot(X_tilde_train.T, y_train - sigmoid_value)
        w = w + alpha * gradient

        ll_train[i] = compute_average_ll(X_tilde_train, y_train, w)
        ll_test[i]  = compute_average_ll(X_tilde_test,  y_test,  w)

    return w, ll_train, ll_test


def plot_predictive_distribution(X, y, w, map_inputs=lambda x: x):
    """
    Plot contour of predictive probability under standard MAP/plugin prediction:
        p(y=1|x) = sigma(w^T x_tilde)
    NOTE: This function uses `predict(...)`, i.e., NOT Laplace.
    """
    xx, yy = plot_data_internal(X, y)
    ax = plt.gca()

    X_grid = np.concatenate((xx.ravel().reshape((-1, 1)),
                             yy.ravel().reshape((-1, 1))), axis=1)
    X_tilde = get_x_tilde(map_inputs(X_grid))

    Z = predict(X_tilde, w).reshape(xx.shape)

    cs2 = ax.contour(xx, yy, Z, cmap='RdBu', linewidths=2)
    plt.clabel(cs2, fmt='%2.1f', colors='k', fontsize=14)
    plt.show()


def evaluate_basis_functions(l, X, Z):
    """
    Gaussian/RBF basis expansion.
    If Z are the centres (often the training inputs), then
        Phi[n,m] = exp( - ||x_n - z_m||^2 / (2 l^2) )

    Returns: Phi of shape (N, M) where N = X.shape[0], M = Z.shape[0]
    """
    X2 = np.sum(X**2, axis=1)
    Z2 = np.sum(Z**2, axis=1)

    ones_Z = np.ones(Z.shape[0])
    ones_X = np.ones(X.shape[0])

    r2 = np.outer(X2, ones_Z) - 2 * np.dot(X, Z.T) + np.outer(ones_X, Z2)
    return np.exp(-0.5 / l**2 * r2)


# ============================================================
# 4) MAP with Gaussian prior + L-BFGS-B (MINIMIZATION)
# ============================================================
def compute_log_posterior(w, X_tilde, y, alpha_prior):
    """
    Log-posterior up to constants:
        log p(w|D) = log p(y|X,w) + log p(w)
    with Gaussian prior: p(w) ~ N(0, alpha_prior^{-1} I)

    Here:
        log p(y|X,w) is computed as a SUM over data points (not average).
        log p(w) = -0.5 * alpha_prior * ||w||^2  (ignoring constants).
    """
    y = np.asarray(y).ravel()
    p = predict(X_tilde, w)

    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)

    ll = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    lp = -0.5 * alpha_prior * np.dot(w, w)
    return ll + lp


def compute_grad_log_posterior(w, X_tilde, y, alpha_prior):
    """
    Gradient of the log-posterior:
        ∇ log p(w|D) = X^T (y - p) - alpha_prior * w
    """
    y = np.asarray(y).ravel()
    p = predict(X_tilde, w)
    return X_tilde.T @ (y - p) - alpha_prior * w


def fit_w_map(X_tilde_train, y_train, X_tilde_test, y_test, n_steps, alpha_lr, alpha_prior):
    """
    Iterative gradient ascent for MAP (log-posterior).
    This is a valid alternative to L-BFGS-B; kept here for optional comparison.
    """
    y_train = np.asarray(y_train).ravel()
    y_test  = np.asarray(y_test).ravel()

    w = np.zeros(X_tilde_train.shape[1])
    ll_train = np.zeros(n_steps)
    ll_test = np.zeros(n_steps)

    for i in range(n_steps):
        grad = compute_grad_log_posterior(w, X_tilde_train, y_train, alpha_prior)
        w = w + alpha_lr * grad

        # Plugin average log-likelihood (NOT posterior) for monitoring convergence
        ll_train[i] = compute_average_ll(X_tilde_train, y_train, w)
        ll_test[i]  = compute_average_ll(X_tilde_test,  y_test,  w)

    return w, ll_train, ll_test


def fit_w_map_lbfgs(X_tilde_train, y_train, X_tilde_test, y_test,
                    alpha_prior, maxiter=200, w0=None):
    """
    Use L-BFGS-B to obtain w_map by MINIMIZING negative log-posterior.
    Also records plugin avg log-likelihood (train/test) for plotting.
    """
    y_train = np.asarray(y_train).ravel()
    y_test  = np.asarray(y_test).ravel()

    D = X_tilde_train.shape[1]
    if w0 is None:
        w0 = np.zeros(D)

    ll_train_hist = []
    ll_test_hist = []

    # Objective to MINIMIZE: negative log-posterior
    def f_obj(w):
        return -compute_log_posterior(w, X_tilde_train, y_train, alpha_prior)

    # Gradient of objective: negative gradient of log-posterior
    def f_grad(w):
        return -compute_grad_log_posterior(w, X_tilde_train, y_train, alpha_prior)

    def callback(wk):
        # Plugin average log-likelihood (NOT posterior) for monitoring/plotting
        ll_train_hist.append(compute_average_ll(X_tilde_train, y_train, wk))
        ll_test_hist.append(compute_average_ll(X_tilde_test,  y_test,  wk))

    w_map, f_min, info = fmin_l_bfgs_b(
        func=f_obj,
        x0=w0,
        fprime=f_grad,
        maxiter=maxiter,
        callback=callback
    )

    return w_map, np.array(ll_train_hist), np.array(ll_test_hist), info


# ============================================================
# 5) Laplace approximation (Hessian + predictive)
# ============================================================
def hessian_neg_log_posterior(w_map, X_tilde, alpha_prior):
    """
    Hessian of negative log-posterior at w_map:
        H = X^T R X + alpha_prior * I
    where R = diag(p_n (1 - p_n)), p_n = sigmoid(x_n^T w_map)
    """
    a = X_tilde @ w_map
    p = logistic(a)
    r = p * (1 - p)  # shape (N,)

    XR = X_tilde * r[:, None]                # multiply each row by r_n
    H = X_tilde.T @ XR + alpha_prior * np.eye(X_tilde.shape[1])
    return H


def kappa_from_var(var_a):
    """
    MacKay logistic-Gaussian approximation factor:
        kappa(v) = 1 / sqrt(1 + (pi/8) v)
    """
    return 1.0 / np.sqrt(1.0 + (np.pi / 8.0) * var_a)


def predict_laplace(X_tilde_star, w_map, S_N):
    """
    Laplace predictive probability approximation:
        p(y=1|x,D) ≈ sigma( kappa(var_a) * mu_a )
    with:
        mu_a  = x_tilde^T w_map
        var_a = x_tilde^T S_N x_tilde
    """
    mu_a = X_tilde_star @ w_map
    var_a = np.sum((X_tilde_star @ S_N) * X_tilde_star, axis=1)
    kappa = kappa_from_var(var_a)
    return logistic(kappa * mu_a)


def compute_average_ll_laplace(X_tilde, y, w, S_N):
    """
    Average log-likelihood computed using Laplace predictive probabilities.
    """
    y = np.asarray(y).ravel()
    output_prob = predict_laplace(X_tilde, w, S_N)

    eps = 1e-12
    output_prob = np.clip(output_prob, eps, 1 - eps)

    return np.mean(y * np.log(output_prob) + (1 - y) * np.log(1.0 - output_prob))


def plot_predictive_distribution_laplace(X, y, w_map, S_N, map_inputs=lambda x: x):
    """
    Contour plot for Laplace predictive probability over the 2D input space.
    """
    xx, yy = plot_data_internal(X, y)
    ax = plt.gca()

    X_grid = np.concatenate((xx.ravel().reshape((-1, 1)),
                             yy.ravel().reshape((-1, 1))), axis=1)
    X_tilde_grid = get_x_tilde(map_inputs(X_grid))

    Z = predict_laplace(X_tilde_grid, w_map, S_N).reshape(xx.shape)

    cs2 = ax.contour(xx, yy, Z, cmap='RdBu', linewidths=2)
    plt.clabel(cs2, fmt='%2.1f', colors='k', fontsize=14)
    plt.show()


# ============================================================
# 6) MAP predictive (non-Laplace) contour plot
# ============================================================
def predict_map(X_tilde_star, w_map):
    """
    MAP/plugin predictive probability:
        p(y=1|x) = sigma(x_tilde^T w_map)
    """
    a = X_tilde_star @ w_map
    return logistic(a)


def plot_predictive_distribution_map(X, y, w_map, map_inputs=lambda x: x):
    """
    Contour plot for MAP/plugin predictive probability over the 2D input space.
    """
    xx, yy = plot_data_internal(X, y)
    ax = plt.gca()

    X_grid = np.concatenate((xx.ravel().reshape((-1, 1)),
                             yy.ravel().reshape((-1, 1))), axis=1)
    X_tilde_grid = get_x_tilde(map_inputs(X_grid))

    Z = predict_map(X_tilde_grid, w_map).reshape(xx.shape)

    cs = ax.contour(xx, yy, Z, cmap='RdBu', linewidths=2)
    plt.clabel(cs, fmt='%2.1f', colors='k', fontsize=14)
    plt.show()


# ============================================================
# 7) Helper: confusion matrix (fractions) with more decimals
# ============================================================
def print_confusion_matrix_fractions(y_true, y_pred, decimals=6, title="Confusion matrix (fractions)"):
    """
    Print confusion matrix entries as FRACTIONS (means over the test set):
      TN = mean(ŷ=0 & y=0)
      FP = mean(ŷ=1 & y=0)
      FN = mean(ŷ=0 & y=1)
      TP = mean(ŷ=1 & y=1)
    """
    y_true = np.asarray(y_true).ravel().astype(int)
    y_pred = np.asarray(y_pred).ravel().astype(int)

    TN = np.mean((y_pred == 0) & (y_true == 0))
    FP = np.mean((y_pred == 1) & (y_true == 0))
    FN = np.mean((y_pred == 0) & (y_true == 1))
    TP = np.mean((y_pred == 1) & (y_true == 1))

    fmt = f"{{:.{decimals}f}}"
    print(title + ":")
    print(f"TN = mean(ŷ=0 & y=0) = {fmt.format(TN)}")
    print(f"FP = mean(ŷ=1 & y=0) = {fmt.format(FP)}")
    print(f"FN = mean(ŷ=0 & y=1) = {fmt.format(FN)}")
    print(f"TP = mean(ŷ=1 & y=1) = {fmt.format(TP)}")


# ============================================================
# 8) Main experiment flow (your original flow, cleaned)
# ============================================================
# Basis expansion (centres = X_train), then add bias via get_x_tilde
l = .1  # lengthscale (tunable)
X_tilde_train = get_x_tilde(evaluate_basis_functions(l, X_train, X_train))
X_tilde_test  = get_x_tilde(evaluate_basis_functions(l, X_test,  X_train))

# Hyperparameters
alpha_lr   = 0.001   # learning rate for gradient ascent (if you switch to fit_w_map)
n_steps    = 3000    # steps for gradient ascent (if used)
alpha_prior = 1.0    # prior precision (regularization strength)

# ------------------------------------------------------------
# Choose ONE of the two MAP solvers:
#   (A) Iterative gradient ascent (commented)
#   (B) L-BFGS-B quasi-Newton (active)
# ------------------------------------------------------------

# (A) Gradient ascent MAP (optional)
# w_map, ll_train_hist, ll_test_hist = fit_w_map(
#     X_tilde_train, y_train,
#     X_tilde_test,  y_test,
#     n_steps=n_steps,
#     alpha_lr=alpha_lr,
#     alpha_prior=alpha_prior
# )

# (B) L-BFGS-B MAP (recommended)
w_map, ll_train_hist, ll_test_hist, info = fit_w_map_lbfgs(
    X_tilde_train, y_train,
    X_tilde_test,  y_test,
    alpha_prior=alpha_prior
)

# Plot convergence traces (plugin avg log-likelihood)
plot_ll(ll_train_hist)
plot_ll(ll_test_hist)

# Print last few values to show convergence + final value (clearer than only last-5)
print("ll_train (last 5) =", ll_train_hist[-5:])
print("ll_test  (last 5) =", ll_test_hist[-5:])
print("ll_train (final)  =", ll_train_hist[-1])
print("ll_test  (final)  =", ll_test_hist[-1])

# ------------------------------------------------------------
# Laplace approximation around w_map
# ------------------------------------------------------------
H = hessian_neg_log_posterior(w_map, X_tilde_train, alpha_prior)
S_N = np.linalg.inv(H)

# Laplace predictive on test set
p_test = predict_laplace(X_tilde_test, w_map, S_N)

# Plot Laplace contour (note the basis mapping)
plot_predictive_distribution_laplace(
    X, y, w_map, S_N,
    map_inputs=lambda x: evaluate_basis_functions(l, x, X_train)
)

# Hard predictions + confusion matrix (Laplace)
y_hat = (p_test > 0.5).astype(int)
print_confusion_matrix_fractions(y_test, y_hat, decimals=6,
                                 title="Confusion matrix (Laplace predictive)")

# Laplace train/test average log-likelihood
laplace_train_ll = compute_average_ll_laplace(X_tilde_train, y_train, w_map, S_N)
laplace_test_ll  = compute_average_ll_laplace(X_tilde_test,  y_test,  w_map, S_N)

print("laplace_train_ll =", laplace_train_ll)
print("laplace_test_ll  =", laplace_test_ll)

# ------------------------------------------------------------
# MAP/plugin predictive (no Laplace)
# ------------------------------------------------------------
p_test = predict_map(X_tilde_test, w_map)

# MAP contour plot
plot_predictive_distribution_map(
    X, y, w_map,
    map_inputs=lambda z: evaluate_basis_functions(l, z, X_train)
)

# Hard predictions + confusion matrix (MAP)
y_hat = (p_test > 0.5).astype(int)
print_confusion_matrix_fractions(y_test, y_hat, decimals=6,
                                 title="Confusion matrix (MAP/plugin predictive)")

# MAP train/test average log-likelihood (plugin)
map_train_ll = compute_average_ll(X_tilde_train, y_train, w_map)
map_test_ll  = compute_average_ll(X_tilde_test,  y_test,  w_map)

print("map_train_ll =", map_train_ll)
print("map_test_ll  =", map_test_ll)

# Summary of the four key numbers (easy to compare)
print("\nSummary (avg log-likelihood):")
print(f"  MAP     : train = {map_train_ll:.6f}, test = {map_test_ll:.6f}")
print(f"  Laplace : train = {laplace_train_ll:.6f}, test = {laplace_test_ll:.6f}")

print("l =", l)

# ============================================================
# 9) Exercise (e) + (f): grid search over (sigma0_sq, l) by Laplace evidence
# ============================================================

def log_evidence_laplace_unnorm(X_tilde_train, y_train, w_map, alpha_prior):
    """
    Laplace log-evidence (marginal likelihood) approximation, up to constants
    independent of (sigma0^2, l) when alpha_prior is fixed:

      log p(D) ≈ log p(y|w_map) + log p(w_map) + (M/2)log(2π) - 0.5 log|H|

    where H is the Hessian of the NEGATIVE log-posterior at w_map.
    """
    y_train = np.asarray(y_train).ravel()

    # Log-likelihood at w_map (SUM, not average)
    p = predict(X_tilde_train, w_map)
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    log_lik = np.sum(y_train * np.log(p) + (1 - y_train) * np.log(1 - p))

    # Log-prior at w_map (Gaussian prior, up to additive constant)
    log_prior = -0.5 * alpha_prior * np.dot(w_map, w_map)

    # Hessian of negative log-posterior + stable logdet
    H = hessian_neg_log_posterior(w_map, X_tilde_train, alpha_prior)
    sign, logdetH = np.linalg.slogdet(H)
    if sign <= 0:
        # Numerical guard: add small jitter to encourage positive definiteness
        jitter = 1e-6
        Hj = H + jitter * np.eye(H.shape[0])
        sign, logdetH = np.linalg.slogdet(Hj)
        if sign <= 0:
            return -np.inf

    M = X_tilde_train.shape[1]
    return log_lik + log_prior + 0.5 * M * np.log(2.0 * np.pi) - 0.5 * logdetH


def plot_heatmap(evidence_grid, sigma0_sq_grid, l_grid, save_path="heatmap_evidence.png"):
    """
    Heatmap of the Laplace log-evidence over (sigma0^2, l).
    evidence_grid shape: (len(sigma0_sq_grid), len(l_grid))
    """
    plt.figure()
    plt.imshow(evidence_grid, origin="lower", aspect="auto")
    plt.colorbar(label="Laplace log-evidence (approx.)")

    plt.xticks(np.arange(len(l_grid)), [f"{v:.3g}" for v in l_grid], rotation=45)
    plt.yticks(np.arange(len(sigma0_sq_grid)), [f"{v:.3g}" for v in sigma0_sq_grid])

    plt.xlabel("l")
    plt.ylabel(r"$\sigma_0^2$")
    plt.title("Heatmap: Laplace evidence over (sigma0^2, l)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()
    print(f"[saved] heatmap -> {save_path}")

# -----------------------------
# (e) Choose 10 grid points (log-spaced grids for scale hyperparameters)
# -----------------------------
sigma0_sq_grid = np.logspace(-1, 1, 10)             # 0.1 ... 10
l_grid         = np.logspace(-2, np.log10(0.5), 10) # 0.01 ... 0.5

print("\nGrid points:")
print("sigma0_sq_grid =", sigma0_sq_grid)
print("l_grid         =", l_grid)

# -----------------------------
# (e) Grid search loop
# -----------------------------
evidence_grid = np.zeros((len(sigma0_sq_grid), len(l_grid)))

best_logZ = -np.inf
best_sigma0_sq = None
best_l = None
best_w_map = None

for i, sigma0_sq in enumerate(sigma0_sq_grid):
    for j, l_val in enumerate(l_grid):

        # Phi = sigma0^2 * exp(-||x-z||^2 / (2 l^2)), centres = X_train
        Phi_train = sigma0_sq * evaluate_basis_functions(l_val, X_train, X_train)
        Phi_test  = sigma0_sq * evaluate_basis_functions(l_val, X_test,  X_train)

        X_tilde_train_g = get_x_tilde(Phi_train)
        X_tilde_test_g  = get_x_tilde(Phi_test)

        # MAP fit (reusing your existing solver)
        w_map_g, _, _, info_g = fit_w_map_lbfgs(
            X_tilde_train_g, y_train,
            X_tilde_test_g,  y_test,
            alpha_prior=alpha_prior,
            maxiter=200
        )

        # Laplace log-evidence (approx.)
        logZ = log_evidence_laplace_unnorm(X_tilde_train_g, y_train, w_map_g, alpha_prior)
        evidence_grid[i, j] = logZ

        if logZ > best_logZ:
            best_logZ = logZ
            best_sigma0_sq = sigma0_sq
            best_l = l_val
            best_w_map = w_map_g

        print(f"[grid] sigma0_sq={sigma0_sq:.3g}, l={l_val:.3g} -> logZ≈{logZ:.6f}")

print("\n[best] sigma0_sq =", best_sigma0_sq)
print("[best] l         =", best_l)
print("[best] logZ≈     =", best_logZ)

# Heatmap for Figure in Exercise (e)
plot_heatmap(evidence_grid, sigma0_sq_grid, l_grid, save_path="heatmap_evidence.png")

# -----------------------------
# (f) Re-run Laplace predictive at tuned hyperparameters
# -----------------------------
Phi_train_best = best_sigma0_sq * evaluate_basis_functions(best_l, X_train, X_train)
Phi_test_best  = best_sigma0_sq * evaluate_basis_functions(best_l, X_test,  X_train)

X_tilde_train_best = get_x_tilde(Phi_train_best)
X_tilde_test_best  = get_x_tilde(Phi_test_best)

# Refit MAP at the best hyperparameters (keeps the flow simple and reproducible)
w_map_best, _, _, info_best = fit_w_map_lbfgs(
    X_tilde_train_best, y_train,
    X_tilde_test_best,  y_test,
    alpha_prior=alpha_prior,
    maxiter=200
)

H_best = hessian_neg_log_posterior(w_map_best, X_tilde_train_best, alpha_prior)
S_N_best = np.linalg.inv(H_best)

p_test_best = predict_laplace(X_tilde_test_best, w_map_best, S_N_best)

plot_predictive_distribution_laplace(
    X, y, w_map_best, S_N_best,
    map_inputs=lambda x: (best_sigma0_sq * evaluate_basis_functions(best_l, x, X_train))
)
# Optional save:
plt.savefig("prediction_after_tuning.png", dpi=200)

y_hat_best = (p_test_best > 0.5).astype(int)
print_confusion_matrix_fractions(
    y_test, y_hat_best, decimals=6,
    title="Confusion matrix (Laplace after tuning)"
)

laplace_train_ll_tuned = compute_average_ll_laplace(X_tilde_train_best, y_train, w_map_best, S_N_best)
laplace_test_ll_tuned  = compute_average_ll_laplace(X_tilde_test_best,  y_test,  w_map_best, S_N_best)

print("laplace_train_ll_after_tuning =", laplace_train_ll_tuned)
print("laplace_test_ll_after_tuning  =", laplace_test_ll_tuned)

print("\nTuned hyperparameters:")
print(f"  tuned sigma0^2 = {best_sigma0_sq:.6g}")
print(f"  tuned l        = {best_l:.6g}")
print(f"  Laplace after tuning: train = {laplace_train_ll_tuned:.6f}, test = {laplace_test_ll_tuned:.6f}")


# -----------------------------
# (finer) Choose 10 grid points (log-spaced grids for scale hyperparameters)
# -----------------------------
sigma0_sq_grid = np.logspace(np.log10(0.3), np.log10(3.0), 10)   # 0.3 ... 3
l_grid         = np.logspace(np.log10(0.1), np.log10(2.0), 10)   # 0.1 ... 2.0

print("\nGrid points:")
print("sigma0_sq_grid =", sigma0_sq_grid)
print("l_grid         =", l_grid)

# -----------------------------
# (finer) Grid search loop
# -----------------------------
evidence_grid = np.zeros((len(sigma0_sq_grid), len(l_grid)))

best_logZ = -np.inf
best_sigma0_sq = None
best_l = None
best_w_map = None

for i, sigma0_sq in enumerate(sigma0_sq_grid):
    for j, l_val in enumerate(l_grid):

        # Phi = sigma0^2 * exp(-||x-z||^2 / (2 l^2)), centres = X_train
        Phi_train = sigma0_sq * evaluate_basis_functions(l_val, X_train, X_train)
        Phi_test  = sigma0_sq * evaluate_basis_functions(l_val, X_test,  X_train)

        X_tilde_train_g = get_x_tilde(Phi_train)
        X_tilde_test_g  = get_x_tilde(Phi_test)

        # MAP fit (reusing your existing solver)
        w_map_g, _, _, info_g = fit_w_map_lbfgs(
            X_tilde_train_g, y_train,
            X_tilde_test_g,  y_test,
            alpha_prior=alpha_prior,
            maxiter=200
        )

        # Laplace log-evidence (approx.)
        logZ = log_evidence_laplace_unnorm(X_tilde_train_g, y_train, w_map_g, alpha_prior)
        evidence_grid[i, j] = logZ

        if logZ > best_logZ:
            best_logZ = logZ
            best_sigma0_sq = sigma0_sq
            best_l = l_val
            best_w_map = w_map_g

        print(f"[grid] sigma0_sq={sigma0_sq:.3g}, l={l_val:.3g} -> logZ≈{logZ:.6f}")

print("\n[best] sigma0_sq =", best_sigma0_sq)
print("[best] l         =", best_l)
print("[best] logZ≈     =", best_logZ)

# Heatmap for Figure in Exercise (e)
plot_heatmap(evidence_grid, sigma0_sq_grid, l_grid, save_path="heatmap_evidence.png")


# -----------------------------
# (finer) Re-run Laplace predictive at tuned hyperparameters
# -----------------------------
Phi_train_best = best_sigma0_sq * evaluate_basis_functions(best_l, X_train, X_train)
Phi_test_best  = best_sigma0_sq * evaluate_basis_functions(best_l, X_test,  X_train)

X_tilde_train_best = get_x_tilde(Phi_train_best)
X_tilde_test_best  = get_x_tilde(Phi_test_best)

# Refit MAP at the best hyperparameters (keeps the flow simple and reproducible)
w_map_best, _, _, info_best = fit_w_map_lbfgs(
    X_tilde_train_best, y_train,
    X_tilde_test_best,  y_test,
    alpha_prior=alpha_prior,
    maxiter=200
)

H_best = hessian_neg_log_posterior(w_map_best, X_tilde_train_best, alpha_prior)
S_N_best = np.linalg.inv(H_best)

p_test_best = predict_laplace(X_tilde_test_best, w_map_best, S_N_best)

plot_predictive_distribution_laplace(
    X, y, w_map_best, S_N_best,
    map_inputs=lambda x: (best_sigma0_sq * evaluate_basis_functions(best_l, x, X_train))
)
# Optional save:
plt.savefig("prediction_after_tuning.png", dpi=200)

y_hat_best = (p_test_best > 0.5).astype(int)
print_confusion_matrix_fractions(
    y_test, y_hat_best, decimals=6,
    title="Confusion matrix (Laplace after tuning)"
)

laplace_train_ll_tuned = compute_average_ll_laplace(X_tilde_train_best, y_train, w_map_best, S_N_best)
laplace_test_ll_tuned  = compute_average_ll_laplace(X_tilde_test_best,  y_test,  w_map_best, S_N_best)

print("laplace_train_ll_after_tuning =", laplace_train_ll_tuned)
print("laplace_test_ll_after_tuning  =", laplace_test_ll_tuned)

print("\nTuned hyperparameters:")
print(f"  tuned sigma0^2 = {best_sigma0_sq:.6g}")
print(f"  tuned l        = {best_l:.6g}")
print(f"  Laplace after tuning: train = {laplace_train_ll_tuned:.6f}, test = {laplace_test_ll_tuned:.6f}")

# Optional baseline comparison (remove if you prefer to compare only in the report)
print("\nBaseline (sigma0^2=1, l=0.1) from Exercise d:")
print("  Laplace : train = -0.258494, test = -0.338271")
print("  MAP     : train = -0.219165, test = -0.315505")
