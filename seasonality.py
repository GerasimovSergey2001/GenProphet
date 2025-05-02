import pymc as pm
import numpy as np


class Seasonality:
    """Seasonality model."""

    def __init__(self, period='yearly', seasonality_prior_scale=10):
        """Create a seasonality model."""
        self.seasonality_prior_scale = seasonality_prior_scale
        self.period = period

    def create_fourier_matrix(self, t, P=7, N=10):
        """Create the Fourier matrix.

        Parameters
        ----------
        t: np.array
                1D array of time.
        P: int
                Number of Fourier components.

        Returns
        -------
        X :np.array
                An array of Fourier components.
        """
        X = 2*np.pi*np.arange(1, N+1)/P
        X = X*t[:, None]
        X = np.concatenate((np.cos(X), np.sin(X)), axis=1)

        return X

    def buid_seasonality(self, model, ds_max, ds_min, t):
        """Create the seasonality model.

        Parameters
        ----------
        model: PyMC model
        ds_max: datetime
                DataFrame with the date maximum value.
        ds_min: datetime
                DataFrame with the date minimum value.
        t: np.array
                1D array of scaled time.
        period: str
                Period of the seasonality.
        seasonality_prior_scale: float
                Controls the flexibility of the seasonality.

        Returns
        -------
        S :np.array
                An array of deterministic values representing the seasonality.
        """
        scale = (ds_max - ds_min).days
        if self.period == 'yearly':
            self.P = 365.25/scale
            self.N = 10
        elif self.period == 'monthly':
            self.P = 28/scale
            self.N = 5
        elif self.period == 'weekly':
            self.P = 7/scale
            self.N = 3
        elif self.period == '2weekly':
            self.P = 14/scale
            self.N = 3
        else:  # P = 1 / 24  # Daily periodicity (scaled for hourly data)???
            raise ValueError('Period not supported')

        X = self.create_fourier_matrix(t, self.P, self.N)

        with model:
            beta = pm.Normal('beta'+'_'+self.period, mu=0,
                             sigma=self.seasonality_prior_scale, shape=X.shape[1])

        return beta, X

    def restore_seasonality(self, beta, X, scale, t=None):
        """Restore the seasonality from the model.

        Parameters
        ----------
        beta: ndarray
            An array of seasonality parameters.
        X: ndarray
            An array of Fourier components.
        scale: float
                Scale of the seasonality.
        """
        if X is None:
            X = self.create_fourier_matrix(t, self.P, self.N)
        return beta.dot(X.T).T * scale
