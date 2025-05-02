import pymc as pm
import numpy as np


class Lineartrend:
    """Linear trend model."""

    def __init__(self, n_changepoints=25, changepoints_prior_scale=0.05, growth_prior_scale=5, changepoint_range=0.8):
        """Create a linear trend model with changepoints."""

        self.n_changepoints = n_changepoints
        self.changepoints_prior_scale = changepoints_prior_scale
        self.growth_prior_scale = growth_prior_scale
        self.changepoint_range = changepoint_range

    def buid_trend(self, model, t):
        """Create the full trend model with changepoints.

        Parameters
        ----------
        model: PyMC model
        t: np.array
            1D array of time.
        n_changepoints: int
            Number of changepoints to use for the forecast.
        changepoints_prior_scale: float
            Controls the flexibility of the changepoints.
        growth_prior_scale: float	
            Controls the flexibility of the trend.		
        changepoint_range: float
            Proportion of history in which trend changepoints will be estimated.

        Returns
        -------
        g :pt vector
            An array of deterministic values representing the trend.
        s :ndarray
            An array of changepoints.
        A :ndarray
            Indicator matrix for changepoints.
        """
        s = np.linspace(0, self.changepoint_range *
                        t.max(), self.n_changepoints+1)[1:]
        A = (t[:, None] > s)*1

        with model:
            m = pm.Normal('m', mu=0, sigma=5)
            k = pm.Normal('k', mu=0, sigma=self.growth_prior_scale)
            self.changepoints_prior_scale = pm.Exponential(
                'tau', self.changepoints_prior_scale)
            delta = pm.Laplace(
                'delta', 0, self.changepoints_prior_scale, shape=self.n_changepoints)
            gamma = -s*delta
            g = pm.Deterministic(
                'trend', (k+pm.math.dot(A, delta))*t + (m+pm.math.dot(A, gamma)))

        return g, s, A

    def restore_trend(self, t, s, A, delta, k, m, scale):
        """Restore the trend from the model.

        Parameters
        ----------
        t: ndarray
            1D array of time.
        s :ndarray
            An array of changepoints.
        A :ndarray
            Indicator matrix for changepoints.
        delta :ndarray
            An array of changepoint deltas (growth rate adjustments).
        k: ndarray
            Growth rate.
        m: ndarray
            Intercept.
        scale: float
            Scale of the data.
        Returns
        -------
        trend :np.array
            An array of deterministic values representing the trend.
        """
        gamma = -s*delta
        slope = (k + np.dot(A, delta.T)) * \
            t if k.size == 1 else (k + np.dot(A, delta.T)) * t[:, None]
        trend = (slope + m + np.dot(A, gamma.T)) * scale
        return trend
