"""Utility functions for prophet"""
from typing import Dict
from functools import partial

import numpy as np
import pandas as pd


def get_changepoints(x: np.ndarray, num_changepoints: int):
    """
    Given x, we split x into num_changepoints+1 equal parts.
    The changepoints are defined as the ends of the first num_changepoints parts.

    Parameters
    ----------
    x: np.ndarray of shape (num_samples, )
    num_changepoints: int

    Returns
    -------
    changepoints
        np.ndarray of shape (1, num_changepoints)
    """
    x = x.squeeze()
    split_x = np.array_split(x, num_changepoints + 1)
    changepoints = [split_x[i][-1] for i in range(num_changepoints)]
    return np.array(changepoints).reshape(1, -1)


def get_A(X: np.ndarray, changepoints: int) -> np.ndarray:
    """
    Given changepoints S = (s_1, s_2, ..., s_num_changepoints), we want to create a
    matrix A such that

        A_{ij} = 1 if X_j > s_i and A_{ij} = 0 other wise.

    A matrix multiplication of a vector delta of shape (1, num_changepoints), which
    contains deltas the different changepoints, with A:

        matmul(delta, A) = result

    leads to a result where the (1, i)'th element of result is basically the
    'cumulative' sum of delta at time i.

    Parameters
    ----------
    X : np.ndarray
        of shape (num_samples, )
    changepoints: np.ndarray
        shape (1, num_changepoints)

    Returns
    -------
    np.ndarray
        of shape (num_changepoints, num_samples)
    """
    if X.ndim > 1:
        X = X.squeeze()

    print(X.shape)
    A = X > np.array(changepoints).reshape(-1, 1)
    return A


def nth_order_fourier_terms(x: np.ndarray, period: int, fourier_order: int):
    """


    Parameters
    ----------
    x : np.ndarray
        shape (num_samples, )
    period : int
        period of fourier series
    fourier_order : int
        max order of fourier series
    Returns
    -------
    result: np.ndarray
        shape (2*fourier_order, num_samples)
    """
    x = x.squeeze()
    result = np.zeros((2 * fourier_order, len(x)))

    for i in range(fourier_order):
        result[i, :] = np.sin(2 * np.pi * x * (i + 1) / period)
        result[i + fourier_order, :] = np.cos(2 * np.pi * x * (i + 1) / period)
    return result


def sort_arrays_using_array(array: np.ndarray, *arrays: np.ndarray):
    """Sort arrays using array"""
    sorted_index = array.argsort()
    sorted_arrays = [x[sorted_index] for x in arrays]
    array = array[sorted_index]
    return array, sorted_arrays


def summary(samples: Dict, num_changepoints, fourier_order):
    """
    Provides a summary of the samples of the weights from the posterior
    """
    site_names = [k for k, v in samples.items()].remove("delta")

    deltas_site_names = [f"delta_{i}" for i in range(num_changepoints)]
    deltas = samples["delta"].cpu().numpy()
    delta_df = pd.DataFrame(deltas, columns=deltas_site_names)

    fourier_a_site_names = [f"a_{i}" for i in range(fourier_order)]
    fourier_b_site_names = [f"b_{i}" for i in range(fourier_order)]
    fourier_coefficients = samples["beta"].cpu().numpy()
    beta_df = pd.DataFrame(
        fourier_coefficients, columns=fourier_a_site_names + fourier_b_site_names
    )

    describe = partial(pd.Series.describe, percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    df = pd.concat(
        [pd.DataFrame(samples, columns=site_names), delta_df, beta_df], axis=1
    ).drop(columns=["delta", "beta"])

    cols = ["mean", "std", "5%", "25%", "50%", "75%", "95%"]
    summary_df = df.apply(describe, axis=0).transpose()[cols]
    return summary_df
