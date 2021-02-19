"""Data generating functions"""
import numpy as np
from .prophet_util import get_changepoints, get_A, nth_order_fourier_terms


def generate_noise(N: int, sigma: float) -> np.ndarray:
    """Generate noise"""
    return np.random.normal(0, sigma, (1, N))


def generate_dataset_v2(
    deltas: np.ndarray = np.array([0]),
    k: float = 1.0,
    N=100,
    sigma=0.5,
    fourier_coefficients: np.ndarray = np.array([0, 0]),
    period: float = 10,
):
    """

    Parameters
    ----------
    deltas : np.ndarray
        shape (num_changepoints, 1)
    N : int, optional
        number of samples to generate
    fourier_coefficients: np.ndarray [a1, a2, ..., aN, b1, b2, ..., bN]
        shape (1, 2*fourier_order)
    period: float
        period of fourier series
    Returns
    -------
    [type]
        [description]
    """
    x = np.linspace(0, N - 1, N).reshape(1, N)
    num_changepoints = len(deltas)
    deltas = deltas.reshape(1, num_changepoints)  # (1, num_changepoints)
    changepoints = get_changepoints(x, num_changepoints)  # (num_changepoints, 1)

    assert deltas.shape == (1, num_changepoints), f"deltas' shape: {deltas.shape}"
    assert changepoints.shape == (
        1,
        num_changepoints,
    ), f"changepoints' shape: {changepoints.shape}"

    gamma = -1 * deltas * changepoints
    assert gamma.shape == (1, num_changepoints), f"gamma's shape: {gamma.shape}"

    A = get_A(x, changepoints)  # ()
    assert A.shape == (num_changepoints, N), f"A's shape: {A.shape}"

    fourier_coefficients = fourier_coefficients.reshape(1, -1)  # (1, 2*fourier_order)
    assert (
        fourier_coefficients.shape[1] % 2 == 0
    ), "Fourier coefficients must be of length 2*fourier_order"
    fourier_order = int(fourier_coefficients.shape[1] / 2)

    fourier_terms = nth_order_fourier_terms(x, period, fourier_order)
    assert fourier_terms.shape == (
        2 * fourier_order,
        N,
    ), f"Fourier terms shape: {fourier_terms.shape}"

    seasonality = np.matmul(fourier_coefficients, fourier_terms)
    assert seasonality.shape == (1, N)

    y = (
        (np.matmul(deltas, A) + k) * x
        + np.matmul(gamma, A)
        + generate_noise(N, sigma)
        + seasonality
    )
    assert y.shape == (1, N)

    x = x.squeeze()
    y = y.squeeze()
    return x, y
