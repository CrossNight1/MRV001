import numpy as np
from numba import njit
from numpy.typing import NDArray
from typing import Tuple
from .i_1 import _kalman_filter_beta

@njit
def pca_basket_weights(returns: NDArray[np.float64]) -> NDArray[np.float64]:
    T, N = returns.shape
    mean_ret = np.zeros(N)
    for i in range(N):
        s = 0.0
        for t in range(T):
            s += returns[t, i]
        mean_ret[i] = s / T

    cov = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            s = 0.0
            for t in range(T):
                s += (returns[t, i] - mean_ret[i]) * (returns[t, j] - mean_ret[j])
            cov[i, j] = s / (T - 1)

    eigvals, eigvecs = np.linalg.eigh(cov)
    idx_max = np.argmax(eigvals)
    weights = eigvecs[:, idx_max]

    sgn = 0.0
    for i in range(N):
        sgn += weights[i]
    if sgn < 0:
        weights = -weights

    abs_sum = 0.0
    for i in range(N):
        abs_sum += abs(weights[i])
    if abs_sum > 0:
        weights /= abs_sum

    return weights

@njit
def basket_returns(returns: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
    T = returns.shape[0]
    basket_ret = np.zeros(T)
    for t in range(T):
        for i in range(weights.shape[0]):
            basket_ret[t] += returns[t, i] * weights[i]
    return basket_ret

@njit
def residual_zscore(asset: NDArray[np.float64],
                    basket: NDArray[np.float64],
                    beta: NDArray[np.float64],
                    window: int = 50) -> NDArray[np.float64]:
    T = asset.shape[0]
    residuals = np.empty(T, dtype=np.float64)
    zscores = np.zeros(T, dtype=np.float64)

    # Calculate residuals
    for t in range(T):
        residuals[t] = asset[t] - beta[t] * basket[t]

    # Precompute rolling sums
    sum_res = 0.0
    sum_sq = 0.0

    for t in range(T):
        val = residuals[t]
        sum_res += val
        sum_sq += val * val
        if t >= window:
            # Remove old value
            old = residuals[t - window]
            sum_res -= old
            sum_sq -= old * old

            mean = sum_res / window
            var = (sum_sq / window) - mean * mean
            denom = np.sqrt(var) if var > 1e-12 else 1e-6
            zscores[t] = (val - mean) / denom

    return zscores

@njit
def compute_all_signals(returns: NDArray[np.float64],
                        basket_ret: NDArray[np.float64],
                        window: int = 50,
                        threshold: float = 2.0,
                        adj_factors: float = 0.1):
    T, N = returns.shape
    beta_matrix = np.zeros((T, N))
    z_matrix = np.zeros((T, N))
    signal_matrix = np.zeros((T, N), dtype=np.int64)

    for i in range(N):
        beta = _kalman_filter_beta(returns[:, i], basket_ret, adjust_factor=adj_factors)
        z = residual_zscore(returns[:, i], basket_ret, beta, window)

        beta_matrix[:, i] = beta
        z_matrix[:, i] = z

        for t in range(T):
            if z[t] > threshold:
                signal_matrix[t, i] = -1
            elif z[t] < -threshold:
                signal_matrix[t, i] = 1

    return beta_matrix, z_matrix, signal_matrix

@njit
def basket_prices(prices: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
    T = prices.shape[0]
    basket_price = np.zeros(T)
    for t in range(T):
        for i in range(weights.shape[0]):
            basket_price[t] += prices[t, i] * weights[i]
    return basket_price

class Indicator_2:
    pca_basket_weights = staticmethod(pca_basket_weights)
    basket_returns = staticmethod(basket_returns)
    basket_prices = staticmethod(basket_prices)
    residual_zscore = staticmethod(residual_zscore)
    compute_all_signals = staticmethod(compute_all_signals)
