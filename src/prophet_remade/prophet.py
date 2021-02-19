# pylint: disable = not-callable
# pylint: disable = no-member
# pylint: disable = missing-docstring
from typing import Dict

import numpy as np
import torch
import pyro
from pyro.infer.mcmc.api import MCMC, NUTS

# from pyro.ops.stats import hpdi, pi  # , pi, quantile

from .prophet_util import (
    nth_order_fourier_terms,
    sort_arrays_using_array,
    get_changepoints,
    get_A,
    summary,
)


class Prophet:
    """
    Prophet remake
    """

    def __init__(
        self,
        changepoint_scale: float = 0.5,
        num_changepoints: int = 5,
        fourier_order: int = 5,
        period: float = 10,
        fourier_scale: float = 0.5,
    ):
        self.changepoint_scale = changepoint_scale
        self.num_changepoints = num_changepoints
        self.fourier_order = fourier_order
        self.period = period
        self.fourier_scale = fourier_scale
        if num_changepoints < 2:
            # TODO: allow 1 changepoint, but must fix dimensions (the squeeze)
            # in that case
            raise ValueError("num_changepoints should be >= 2.")

        self._samples: Dict = None
        self._changepoints: np.ndarray = None

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        num_samples: int = 100,
    ):
        """

        Parameters
        ----------
        X : np.ndarray
            of shape (num_samples, )
        Y : np.ndarray
            of shape (num_samples, )
        """
        model = self.model
        (
            X_tensor,
            Y_tensor,
            changepoints_tensor,
            A_tensor,
            fourier_terms_tensor,
        ) = self._prep_data_fit(X, Y)

        nuts_kernel = NUTS(
            model,
            adapt_step_size=True,
            jit_compile=True,
            adapt_mass_matrix=True,
        )

        hmc_posterior = MCMC(
            nuts_kernel,
            num_samples=num_samples,
            warmup_steps=100,
            num_chains=1,
            disable_progbar=False,
        )

        hmc_posterior.run(
            X_tensor, Y_tensor, changepoints_tensor, A_tensor, fourier_terms_tensor
        )
        self._samples = squeeze_samples(hmc_posterior.get_samples())
        return self._samples

    def model(
        self,
        X: torch.tensor,
        Y: torch.tensor,
        changepoints: torch.tensor,
        A: torch.tensor,
        fourier_terms: torch.tensor,
    ):
        """


        Parameters
        ----------
        X : torch.tensor
            (1, num_samples)
        Y : torch.tensor
            (1, num_samples)
        changepoints : torch.tensor
            (1, num_changepoints)
        A : torch.tensor
            (num_changepoints, num_samples)
        fourier_terms: torch.tensor
            (2*fourier_order, num_samples)
        """
        # pylint: disable = too-many-locals
        k_prior = pyro.distributions.Normal(0, 5)
        # m_prior = pyro.distributions.Normal(0, 0.5)
        delta_prior = pyro.distributions.Laplace(
            torch.zeros(1, self.num_changepoints),
            torch.ones(1, self.num_changepoints) * self.changepoint_scale,
        )  # shape (1, num_changepoints)
        beta_prior = pyro.distributions.Normal(
            torch.zeros(1, 2 * self.fourier_order),
            torch.ones(1, 2 * self.fourier_order),
        )  # shape (1, 2*fourier_order)

        sigma_prior = pyro.distributions.Exponential(1)

        k = pyro.sample("k", k_prior)
        delta = pyro.sample("delta", delta_prior)
        beta = pyro.sample("beta", beta_prior)
        sigma = pyro.sample("sigma", sigma_prior)
        gamma = -1 * torch.mul(delta, changepoints)

        seasonality = torch.mm(beta, fourier_terms)
        print(f"Deltas shape: {delta.shape}")
        print(f"Changepoints shape: {changepoints.shape}")
        print(f"Gamma shape: {gamma.shape}")
        print(f"X shape: {X.shape}")
        print(f"A shape: {A.shape}")
        print(f"Fourier terms shape: {fourier_terms.shape}")
        print(f"seasonality shape: {seasonality.shape}")

        mu = (torch.mm(delta, A) + k) * X + torch.mm(gamma, A) + seasonality

        likelihood = pyro.distributions.Normal(loc=mu, scale=sigma)
        return pyro.sample("likelihood", likelihood, obs=Y)

    def predict(self, X: np.ndarray):
        """

        Parameters
        ----------
        X : np.ndarray
            shape (num_samples, )

        Returns
        --------
        mu: torch.tensor
            shape (num_samples, num_generated_samples)
        """
        # TODO: this method assumes X is the same X used during training
        # TODO: Deal with this by calculating A etc differently
        X, changepoints, A, fourier_terms = self._prep_data_predict(X)
        delta = self._samples["delta"]  # (num_generated_samples, num_changepoints)
        k = self._samples["k"].reshape(-1, 1)  # (num_generated_samples, 1)
        beta = self._samples["beta"]  # (num_generated_samples, 2*fourier_order)

        gamma = -1 * delta * changepoints  # (num_generated_samples, num_changepoints)
        print(f"k: {k.shape}\ndelta: {delta.shape}\nA: {A.shape}\ngamma: {gamma.shape}")
        seasonality = torch.mm(
            beta, fourier_terms
        )  # (num_generated_samples, num_samples)
        mu = (
            torch.add(torch.mm(delta, A), k) * X + torch.mm(gamma, A) + seasonality
        )  # (num_generated_samples, num_samples)

        return mu.T  # (num_samples, num_generated_samples)

    def _prep_data_predict(self, X: np.ndarray):
        """

        Parameters
        ----------
        X : np.ndarray
            of shape (num_samples, )

        Returns
        -------
        X_tensor
            shape (1, num_samples)
        changepoints_tensor
            shape (1, num_changepoints)
        A_tensor
            shape (num_changepoints, num_samples)
        fourier_terms_tensor
            shape (2*fourier_order, num_samples)
        """

        changepoints = self._changepoints
        A = get_A(X, changepoints)
        fourier_terms = nth_order_fourier_terms(X, self.period, self.fourier_order)

        X_tensor = torch.tensor(X, dtype=torch.float).reshape(1, -1)
        changepoints_tensor = torch.tensor(changepoints, dtype=torch.float)
        A_tensor = torch.tensor(A, dtype=torch.float)
        fourier_terms_tensor = torch.tensor(fourier_terms, dtype=torch.float)

        return X_tensor, changepoints_tensor, A_tensor, fourier_terms_tensor

    def _prep_data_fit(self, X: np.ndarray, Y: np.ndarray):
        """
        Parameters
        ----------
        X : np.ndarray
            (num_samples, )
        Y : np.ndarray, optional
            (num_samples, )

        Returns
        -------
        X_tensor
            shape (1, num_samples)
        Y_tensor
            shape (1, num_samples)
        changepoints_tensor
            shape (1, num_changepoints)
        A_tensor
            shape (num_changepoints, num_samples)
        """
        X, Y = sort_arrays_using_array(X, Y)
        Y_tensor = torch.tensor(Y, dtype=torch.float).reshape(1, -1)
        changepoints = get_changepoints(X, self.num_changepoints)
        self._changepoints = changepoints

        (
            X_tensor,
            changepoints_tensor,
            A_tensor,
            fourier_terms_tensor,
        ) = self._prep_data_predict(
            X,
        )
        return X_tensor, Y_tensor, changepoints_tensor, A_tensor, fourier_terms_tensor

    def summary(self):
        """Return summary dataframe"""
        return summary(self._samples, self.num_changepoints, self.fourier_order)


def squeeze_samples(samples: Dict):
    """
    Squeeze the dimensions of samples
    """
    for key, value in samples.items():
        print(f"BEFORE {key}: dim {value.shape}")
        # Squeeze latest dimension if we have more than 2 dimensions
        ndim = value.ndim
        if ndim > 2:
            # samples[key] = value.squeeze(axis=ndim-1)
            samples[key] = value.squeeze()
            print(f"AFTER {key}: dim {samples[key].shape}")

    return samples
