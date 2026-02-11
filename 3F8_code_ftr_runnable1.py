import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
X = np.loadtxt(DATA_DIR / "x.txt")
y = np.loadtxt(DATA_DIR / "y.txt")

permutation = np.random.permutation(X.shape[ 0 ])
X = X[ permutation, : ]
y = y[ permutation ]

def plot_data_internal(X, y):
    x_min, x_max = X[ :, 0 ].min() - .5, X[ :, 0 ].max() + .5
    y_min, y_max = X[ :, 1 ].min() - .5, X[ :, 1 ].max() + .5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    plt.figure()
    plt.xlim(xx.min(None), xx.max(None))
    plt.ylim(yy.min(None), yy.max(None))
    ax = plt.gca()
    ax.plot(X[y == 0, 0], X[y == 0, 1], 'ro', label = 'Class 1')
    ax.plot(X[y == 1, 0], X[y == 1, 1], 'bo', label = 'Class 2')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Plot data')
    plt.legend(loc = 'upper left', scatterpoints = 1, numpoints = 1)
    return xx, yy

def plot_data(X, y):
    xx, yy = plot_data_internal(X, y)
    plt.show()

plot_data(X, y)


n_train = 800
X_train = X[ 0 : n_train, : ]
X_test = X[ n_train :, : ]
y_train = y[ 0 : n_train ]
y_test = y[ n_train : ]

def logistic(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))

def predict(X_tilde, w): return logistic(np.dot(X_tilde, w))

def compute_average_ll(X_tilde, y, w):
    output_prob = predict(X_tilde, w)
    eps = 1e-12
    output_prob = np.clip(output_prob, eps, 1 - eps)
    return np.mean(y * np.log(output_prob) + (1 - y) * np.log(1.0 - output_prob))

def get_x_tilde(X): return np.concatenate((np.ones((X.shape[ 0 ], 1 )), X), 1)

def fit_w(X_tilde_train, y_train, X_tilde_test, y_test, n_steps, alpha):
    w = np.random.randn(X_tilde_train.shape[ 1 ])
    ll_train = np.zeros(n_steps)
    ll_test = np.zeros(n_steps)
    for i in range(n_steps):
        sigmoid_value = predict(X_tilde_train, w)
        gradient = np.dot(X_tilde_train.T, y_train - sigmoid_value)
        w = w + alpha * gradient # XXX Gradient-based update rule for w. To be completed by the student
        ll_train[ i ] = compute_average_ll(X_tilde_train, y_train, w)
        ll_test[ i ] = compute_average_ll(X_tilde_test, y_test, w)
        # print(ll_train[ i ], ll_test[ i ])

    return w, ll_train, ll_test

def plot_ll(ll):
    plt.figure()
    ax = plt.gca()
    plt.xlim(0, len(ll) + 2)
    plt.ylim(min(ll) - 0.1, max(ll) + 0.1)
    ax.plot(np.arange(1, len(ll) + 1), ll, 'r-')
    plt.xlabel('Steps')
    plt.ylabel('Average log-likelihood')
    plt.title('Plot Average Log-likelihood Curve')
    plt.show()

def plot_predictive_distribution(X, y, w, map_inputs = lambda x : x):
    xx, yy = plot_data_internal(X, y)
    ax = plt.gca()
    X_tilde = get_x_tilde(map_inputs(np.concatenate(
        (xx.ravel().reshape((-1, 1)), yy.ravel().reshape((-1, 1))), 1)))
    Z = predict(X_tilde, w)
    Z = Z.reshape(xx.shape)
    cs2 = ax.contour(xx, yy, Z, cmap = 'RdBu', linewidths = 2)
    plt.clabel(cs2, fmt = '%2.1f', colors = 'k', fontsize = 14)
    plt.show()

def evaluate_basis_functions(l, X, Z):
    X2 = np.sum(X**2, 1)
    Z2 = np.sum(Z**2, 1)
    ones_Z = np.ones(Z.shape[ 0 ])
    ones_X = np.ones(X.shape[ 0 ])
    r2 = np.outer(X2, ones_Z) - 2 * np.dot(X, Z.T) + np.outer(ones_X, Z2)
    return np.exp(-0.5 / l**2 * r2)


import numpy as np
from scipy.optimize import fmin_l_bfgs_b

def compute_log_posterior(w, X_tilde, y, alpha_prior):
    p = predict(X_tilde, w)
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    ll = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))          # log-likelihood (sum)
    lp = -0.5 * alpha_prior * np.dot(w, w)                        # log-prior up to const
    return ll + lp


def compute_grad_log_posterior(w, X_tilde, y, alpha_prior):
    p = predict(X_tilde, w)
    return X_tilde.T @ (y - p) - alpha_prior * w


def fit_w_map(X_tilde_train, y_train, X_tilde_test, y_test, n_steps, alpha_lr, alpha_prior):
    w = np.zeros(X_tilde_train.shape[1])
    ll_train = np.zeros(n_steps)
    ll_test = np.zeros(n_steps)

    for i in range(n_steps):
        grad = compute_grad_log_posterior(w, X_tilde_train, y_train, alpha_prior)
        w = w + alpha_lr * grad

        # plugin average log-likelihood (as in coursework)
        ll_train[i] = compute_average_ll(X_tilde_train, y_train, w)
        ll_test[i]  = compute_average_ll(X_tilde_test,  y_test,  w)

    return w, ll_train, ll_test

def fit_w_map_lbfgs(X_tilde_train, y_train, X_tilde_test, y_test,
                    alpha_prior, maxiter=200, w0=None):
    """
    Use L-BFGS-B to obtain w_map by MINIMIZING negative log posterior.
    Also records plugin avg log-likelihood (train/test) for plotting.
    """
    D = X_tilde_train.shape[1]
    if w0 is None:
        w0 = np.zeros(D)

    ll_train_hist = []
    ll_test_hist = []

    # Objective to MINIMIZE: negative log posterior
    def f_obj(w):
        return -compute_log_posterior(w, X_tilde_train, y_train, alpha_prior)

    # Gradient of objective: negative gradient of log posterior
    def f_grad(w):
        return -compute_grad_log_posterior(w, X_tilde_train, y_train, alpha_prior)

    def callback(wk):
        # plugin average log-likelihood (NOT posterior) for plotting/debug
        ll_train_hist.append(compute_average_ll(X_tilde_train, y_train, wk))
        ll_test_hist.append(compute_average_ll(X_tilde_test, y_test, wk))

    w_map, f_min, info = fmin_l_bfgs_b(
        func=f_obj,
        x0=w0,
        fprime=f_grad,
        maxiter=maxiter,
        callback=callback
    )

    return w_map, np.array(ll_train_hist), np.array(ll_test_hist), info

def hessian_neg_log_posterior(w_map, X_tilde, alpha_prior):
    """
    Hessian of negative log posterior at w_map:
        H = X^T R X + alpha I
    where R = diag(p_n (1 - p_n)), p_n = sigmoid(x_n^T w_map)
    """
    a = X_tilde @ w_map
    p = logistic(a)
    r = p * (1 - p)  # shape (N,)

    # Efficient: X^T R X = (X^T * r) @ X, where r multiplies rows of X
    XR = X_tilde * r[:, None]
    H = X_tilde.T @ XR + alpha_prior * np.eye(X_tilde.shape[1])
    return H

def kappa_from_var(var_a):
    # MacKay approximation factor
    return 1.0 / np.sqrt(1.0 + (np.pi / 8.0) * var_a)


def predict_laplace(X_tilde_star, w_map, S_N):
    mu_a = X_tilde_star @ w_map  # shape (N*,)
    var_a = np.sum((X_tilde_star @ S_N) * X_tilde_star, axis=1)
    kappa = kappa_from_var(var_a)
    return logistic(kappa * mu_a)

def compute_average_ll_laplace(X_tilde, y, w, S_N):
    output_prob = predict_laplace(X_tilde, w, S_N)
    eps = 1e-12
    output_prob = np.clip(output_prob, eps, 1 - eps)
    return np.mean(y * np.log(output_prob) + (1 - y) * np.log(1.0 - output_prob))

# def log_evidence_laplace(X_tilde, y, w_map, H, alpha_prior):
#     """
#     Laplace approximation to log model evidence:
#         log p(D) ≈ log p(y|w_map) + log p(w_map) + (M/2)log(2π) - 0.5 log|H|
#     where H is Hessian of negative log posterior at w_map (i.e. -∇² log p(w|D))
#     """
#     a = X_tilde @ w_map
#     p = logistic(a)
#     eps = 1e-12
#     p = np.clip(p, eps, 1 - eps)

#     log_lik = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
#     log_prior = 0.5 * X_tilde.shape[1] * np.log(alpha_prior / (2.0 * np.pi)) - 0.5 * alpha_prior * np.dot(w_map, w_map)

#     sign, logdetH = np.linalg.slogdet(H)
#     if sign <= 0:
#         # should not happen if H is PD; if it does, numeric issue
#         raise ValueError("Hessian not positive definite (slogdet sign <= 0).")

#     M = X_tilde.shape[1]
#     logZ = log_lik + log_prior + 0.5 * M * np.log(2.0 * np.pi) - 0.5 * logdetH
#     return logZ


def plot_predictive_distribution_laplace(X, y, w_map, S_N, map_inputs=lambda x: x):
    xx, yy = plot_data_internal(X, y)
    ax = plt.gca()
    X_grid = np.concatenate((xx.ravel().reshape((-1, 1)), yy.ravel().reshape((-1, 1))), 1)
    X_tilde_grid = get_x_tilde(map_inputs(X_grid))
    Z = predict_laplace(X_tilde_grid, w_map, S_N).reshape(xx.shape)
    cs2 = ax.contour(xx, yy, Z, cmap='RdBu', linewidths=2)
    plt.clabel(cs2, fmt='%2.1f', colors='k', fontsize=14)
    plt.show()

def predict_map(X_tilde_star, w_map):
    a = X_tilde_star @ w_map
    return logistic(a)  # 你已有 logistic()

def plot_predictive_distribution_map(X, y, w_map, map_inputs=lambda x: x):
    xx, yy = plot_data_internal(X, y)
    ax = plt.gca()
    X_grid = np.concatenate(
        (xx.ravel().reshape((-1, 1)), yy.ravel().reshape((-1, 1))),
        axis=1
    )
    X_tilde_grid = get_x_tilde(map_inputs(X_grid))
    Z = predict_map(X_tilde_grid, w_map).reshape(xx.shape)

    cs = ax.contour(xx, yy, Z, cmap='RdBu', linewidths=2)
    plt.clabel(cs, fmt='%2.1f', colors='k', fontsize=14)
    plt.show()


# Gaussian basis expansion
l = .1  # XXX
X_tilde_train = get_x_tilde(evaluate_basis_functions(l, X_train, X_train))
X_tilde_test  = get_x_tilde(evaluate_basis_functions(l, X_test,  X_train))

# Hyperparameters
alpha_lr = 0.001     # XXX learning rate for gradient ascent
n_steps  = 3000     # XXX steps
alpha_prior = 1.0    # prior precision (regularization strength)

# w_map, ll_train, ll_test = fit_w_map(
#     X_tilde_train, y_train,
#     X_tilde_test,  y_test,
#     n_steps=n_steps,
#     alpha_lr=alpha_lr,
#     alpha_prior=alpha_prior
# )

w_map, ll_train_hist, ll_test_hist, info = fit_w_map_lbfgs(
    X_tilde_train, y_train,
    X_tilde_test,  y_test,
    alpha_prior=alpha_prior
)

plot_ll(ll_train_hist)
plot_ll(ll_test_hist)

# Laplace covariance
H = hessian_neg_log_posterior(w_map, X_tilde_train, alpha_prior)
S_N = np.linalg.inv(H)

# Predictive probabilities (Laplace predictive)
p_test = predict_laplace(X_tilde_test, w_map, S_N)

# Plot predictive distribution must use basis mapping!
plot_predictive_distribution_laplace(
    X, y, w_map, S_N,
    map_inputs=lambda x: evaluate_basis_functions(l, x, X_train)
)


# Apply threshold tau = 0.5 to obtain hard predictions
y_hat = (p_test > 0.5).astype(int)

# Confusion matrix entries (fractions)
TN = np.mean((y_hat == 0) & (y_test == 0))  # True Negatives
FP = np.mean((y_hat == 1) & (y_test == 0))  # False Positives
FN = np.mean((y_hat == 0) & (y_test == 1))  # False Negatives
TP = np.mean((y_hat == 1) & (y_test == 1))  # True Positives

print("Confusion matrix (fractions):")
print(f"TN = P(ŷ=0 | y=0) = {TN:.3f}")
print(f"FP = P(ŷ=1 | y=0) = {FP:.3f}")
print(f"FN = P(ŷ=0 | y=1) = {FN:.3f}")
print(f"TP = P(ŷ=1 | y=1) = {TP:.3f}")

print('ll_train =',ll_train_hist[-5:],'ll_test =', ll_test_hist[-5:])

print('laplace_train_ll =', compute_average_ll_laplace(X_tilde_train, y_train, w_map, S_N))
print('laplace_test_ll =', compute_average_ll_laplace(X_tilde_test, y_test, w_map, S_N))

p_test = predict_map(X_tilde_test, w_map)

print('l =',l)


plot_predictive_distribution_map(
    X, y, w_map,
    map_inputs=lambda z: evaluate_basis_functions(l, z, X_train)
)

y_hat = (p_test > 0.5).astype(int)

# Confusion matrix entries (fractions)
TN = np.mean((y_hat == 0) & (y_test == 0))  # True Negatives
FP = np.mean((y_hat == 1) & (y_test == 0))  # False Positives
FN = np.mean((y_hat == 0) & (y_test == 1))  # False Negatives
TP = np.mean((y_hat == 1) & (y_test == 1))  # True Positives

print("Confusion matrix (fractions):")
print(f"TN = P(ŷ=0 | y=0) = {TN:.3f}")
print(f"FP = P(ŷ=1 | y=0) = {FP:.3f}")
print(f"FN = P(ŷ=0 | y=1) = {FN:.3f}")
print(f"TP = P(ŷ=1 | y=1) = {TP:.3f}")
