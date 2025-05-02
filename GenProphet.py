import pymc as pm
import numpy as np
import pandas as pd
import scipy.stats as sps
import matplotlib.pyplot as plt

# import plotly.graph_objects as go

from .trend import Lineartrend
from .seasonality import Seasonality


class GenProphet:
    """
    Generalized Prophet model using Bayesian inference with PyMC.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with 'ds' (datetime) and 'y' (target) columns.
    trend_parameters : dict, optional
        Dictionary with parameters for the trend component.
    seasonality_parameters : list of dicts, optional
        List of dictionaries defining seasonal components.
    default_likelihood : bool, optional
        Whether to use a default likelihood distribution.
    likelihood : str, optional
        Type of likelihood distribution: 'laplace' or 'normal'.
    p : float, optional
        Significance level for confidence intervals.

    Attributes
    ----------
    model : pm.Model
        PyMC model object.
    trend : Lineartrend
        Trend component of the model.
    seasonality : dict
        Seasonality components by period.
    posterior : InferenceData or dict
        Posterior samples or MAP estimates after fitting.
    df_results : pd.DataFrame
        Dataframe containing fitted results and intervals.
    """

    def __init__(self, df,
                 trend_parameters={'n_changepoints': 25, 'changepoints_prior_scale': 0.05,
                                   'growth_prior_scale': 5, 'changepoint_range': 0.8},
                 seasonality_parameters=[
                     {'period': 'yearly', 'seasonality_prior_scale': 10},
                     {'period': 'weekly', 'seasonality_prior_scale': 10}
                 ],
                 default_likelihood=True,
                 likelihood='laplace', p=0.05
                 ):
        """
        Initialize the generalized Prophet model.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with columns 'ds' (datetime) and 'y' (target variable).
        trend_parameters : dict, optional
            Parameters for the trend component. Expected keys:
                - 'n_changepoints': int
                - 'changepoints_prior_scale': float
                - 'growth_prior_scale': float
                - 'changepoint_range': float
        seasonality_parameters : list of dicts, optional
            List of dictionaries with seasonality configuration. Each dictionary should contain:
                - 'period': str, e.g., 'weekly', 'yearly'
                - 'seasonality_prior_scale': float
        default_likelihood : bool, optional
            If True, adds a default likelihood (Laplace or Normal) to the model.
        likelihood : str, optional
            Type of likelihood to use: 'laplace' or 'normal'.
        p : float, optional
            Significance level for uncertainty intervals, default is 0.05 (i.e., 95% interval).
        """
        self.df = df
        self.trend_parameters = trend_parameters
        self.trend_parameters = seasonality_parameters
        self.p = p
        self.trend = Lineartrend(**trend_parameters)

        if len(seasonality_parameters) == 0:
            self.seasonality = None
        else:
            self.seasonality = dict()
            for seasonality_period in seasonality_parameters:
                self.seasonality[seasonality_period['period']
                                 ] = Seasonality(**seasonality_period)
        self.preprocess()

        self.model = pm.Model()
        self.model.add_coords({"t": self.df['t'].values})

        with self.model:

            y_hat, self.s, self._A = self.trend.buid_trend(
                self.model, self.df['t'].values)

            self._fourier_matrices = dict()
            if self.seasonality is not None:
                for period in self.seasonality:
                    beta, X = self.seasonality[period].buid_seasonality(
                        self.model, self.df['ds'].max(
                        ), self.df['ds'].min(), self.df['t'].values
                    )
                    self._fourier_matrices[period] = X
                    y_hat += pm.Deterministic(period +
                                              '_seasonality', pm.math.dot(X, beta))
            y_hat = pm.Deterministic('y_hat', y_hat)
        eps = 10e-6
        self.likelihood = likelihood
        if default_likelihood:
            with self.model:
                sigma = pm.HalfNormal('sigma', 0.5)
                observed = np.asarray(self.df['y_scaled'], dtype=np.float64)
                if self.likelihood == 'laplace':
                    y = pm.Laplace('y', mu=y_hat, b=sigma,
                                   observed=observed)
                else:
                    y = pm.Normal('y', mu=y_hat, sigma=sigma,
                                  observed=observed)

        else:
            print("Please specify a likelihood function using 'with model: ...'")

    def preprocess(self):
        """
        Preprocesses the input dataframe.

        Converts 'ds' to datetime, scales 'y', and creates a normalized time index 't'.
        """
        self.df['ds'] = pd.to_datetime(self.df['ds'])
        self.df['y_scaled'] = self.df['y'] / self.df['y'].max()
        self.df['t'] = (self.df['ds'] - self.df['ds'].min()) / \
            (self.df['ds'].max() - self.df['ds'].min())

    def fit(self, mcmc_samples=1000, cores=4, progressbar=False, nuts_sampler='blackjax'):
        """
        Fit the model using MCMC or MAP estimation.

        Parameters
        ----------
        mcmc_samples : int
            Number of MCMC samples to draw.
        cores : int
            Number of CPU cores to use.
        progressbar : bool
            Whether to display progress bar.
        nuts_sampler : str
            Sampler backend to use (e.g. 'blackjax').

        Returns
        -------
        GenProphet
            Fitted model instance.
        """
        self.mcmc_samples = mcmc_samples
        self.cores = cores
        self.fitted = True

        with self.model:

            if self.mcmc_samples > 0:
                self.posterior = pm.sample(
                    mcmc_samples, cores=cores, progressbar=progressbar, nuts_sampler=nuts_sampler).posterior

            else:
                self.posterior = pm.find_MAP(progressbar=progressbar)

            self.make_dataset(self._A, self._fourier_matrices)
        return self

    def make_dataset(self, A, fourier_matrices=None):
        """
        Constructs posterior predictive dataset.

        Parameters
        ----------
        A : ndarray
            Changepoint indicator matrix.
        fourier_matrices : dict, optional
            Fourier design matrices for seasonality.

        Returns
        -------
        None
            Results are stored in `self.df_results`.
        """

        self.scale = self.df['y'].max()
        if self.fitted:
            info_data = dict()
            if self.mcmc_samples > 0:

                delta = self.posterior.delta.values.reshape(
                    self.mcmc_samples*self.cores, -1)
                k = self.posterior.k.values.reshape(-1)
                m = self.posterior.m.values.reshape(-1)

                trend_posterior = self.trend.restore_trend(self.df['t'].values, self.s, A,
                                                           delta, k, m, self.scale)

                quant = np.quantile(
                    trend_posterior, [self.p, 1 - self.p], axis=1)
                info_data['trend_mean'] = trend_posterior.mean(axis=1)
                info_data['trend_lower'] = quant[0]
                info_data['trend_upper'] = quant[1]

                y_hat = trend_posterior.copy()

                if self.seasonality is not None:
                    for period in self.seasonality:
                        beta = self.posterior['beta'+'_' +
                                              period].values.reshape(-1, fourier_matrices[period].shape[1])
                        seasonality_posterior = self.seasonality[period].restore_seasonality(beta,
                                                                                             fourier_matrices[period],
                                                                                             self.scale
                                                                                             )

                        quant = np.quantile(seasonality_posterior, [
                                            self.p/2, 1 - self.p/2], axis=1)
                        info_data[period+'_' +
                                  'seasonality_mean'] = seasonality_posterior.mean(axis=1)
                        info_data[period+'_'+'seasonality_lower'] = quant[0]
                        info_data[period+'_'+'seasonality_upper'] = quant[1]
                        y_hat += seasonality_posterior

                info_data['y_hat'] = y_hat.mean(axis=1)
                sigma = self.posterior['sigma'].mean().to_numpy()*self.scale

                # сделать несколько вариантов на выбор !!!
                print(y_hat.mean(axis=1))
                print(y_hat)
                dist = sps.laplace(info_data['y_hat'].values, sigma) if self.likelihood == 'laplace' else sps.norm(
                    info_data['y_hat'].values, sigma)

                info_data['y_hat_lower'] = dist.ppf(self.p/2)
                info_data['y_hat_upper'] = dist.ppf(1-self.p/2)

                info_data['y'] = self.df['y']

                info_data['ds'] = self.df['ds']

                self.df_results = pd.DataFrame(info_data)[
                    ['ds', 'y', 'y_hat', 'y_hat_lower', 'y_hat_upper', 'trend_mean', 'trend_lower', 'trend_upper'] +
                    sorted([period+'_'+'seasonality_mean' for period in self.seasonality] +
                           [period+'_'+'seasonality_lower' for period in self.seasonality] +
                           [period+'_'+'seasonality_upper' for period in self.seasonality])
                ]

            else:
                trend_posterior = self.trend.restore_trend(self.df['t'].values, self.s, A,
                                                           self.posterior['delta'], self.posterior['k'],
                                                           self.posterior['m'], self.scale
                                                           )

                info_data['trend_mean'] = trend_posterior
                y_hat = trend_posterior.copy()
                if self.seasonality is not None:
                    for period in self.seasonality:
                        beta = self.posterior['beta'+'_'+period]

                        seasonality_posterior = self.seasonality[period].restore_seasonality(beta,
                                                                                             fourier_matrices[period],
                                                                                             self.scale
                                                                                             )

                        info_data[period+'_' +
                                  'seasonality_mean'] = seasonality_posterior
                        y_hat += seasonality_posterior

                info_data['y_hat'] = y_hat
                sigma = self.posterior['sigma']*self.df['y'].max()
                dist = sps.laplace(info_data['y_hat'], sigma) if self.likelihood == 'laplace' else sps.norm(
                    info_data['y_hat'], sigma)

                info_data['y_hat_lower'] = dist.ppf(self.p/2)
                info_data['y_hat_upper'] = dist.ppf(1-self.p/2)

                info_data['y'] = self.df['y']
                info_data['ds'] = self.df['ds']
                self.df_results = pd.DataFrame(info_data)[
                    ['ds', 'y', 'y_hat', 'y_hat_lower', 'y_hat_upper', 'trend_mean'] + sorted([period+'_'+'seasonality_mean' for period in self.seasonality])]

    def predict(self, n_periods=5, n_samples=100):
        """
        Forecast future values.

        Parameters
        ----------
        n_periods : int
            Number of future time steps to forecast.
        n_samples : int
            Number of posterior predictive samples.

        Returns
        -------
        pd.DataFrame
            Forecast dataframe with predictions and uncertainty intervals.
        """

        history_points = self.df.shape[0]
        ds = self.df['ds'].to_list()
        ds.extend([self.df['ds'].max() + pd.Timedelta(t, 'D')
                  for t in range(1, n_periods+1)])

        future = pd.DataFrame({'ds': ds})

        future['t'] = np.concatenate(
            (self.df.t.values, 1+np.arange(1, n_periods+1)*np.diff(self.df.t.values)[0]))
        trend_forecast = []

        delta = self.posterior['delta'].mean(axis=(0, 1)).to_numpy(
        ) if self.mcmc_samples > 0 else self.posterior['delta']

        k = self.posterior['k'].mean(axis=(0, 1)).to_numpy(
        ) if self.mcmc_samples > 0 else self.posterior['k']
        m = self.posterior['m'].mean(axis=(0, 1)).to_numpy(
        ) if self.mcmc_samples > 0 else self.posterior['m']
        trend_forecast = []
        lambda_ = self.posterior['tau'].mean(axis=(0, 1)).to_numpy(
        ) if self.mcmc_samples > 0 else self.posterior['tau']
        probability_changepoint = (
            np.abs(delta) > 0.0001).sum() / history_points

        for n in range(n_samples):
            new_changepoints = future['t'][future['t'] > 1].values
            sample = np.random.random(new_changepoints.shape)
            new_changepoints = new_changepoints[sample <=
                                                probability_changepoint]
            new_delta = np.r_[delta, sps.laplace(
                0, lambda_).rvs(new_changepoints.shape[0])]
            new_s = np.r_[self.s, new_changepoints]
            new_A = (future['t'].values[:, None] > new_s) * 1

            trend_forecast.append(self.trend.restore_trend(
                future['t'].values, new_s, new_A, new_delta, k, m, scale=self.scale))

        trend_forecast = np.array(trend_forecast)

        y_forecast = np.zeros_like(trend_forecast)
        y_forecast += trend_forecast

        if self.seasonality is not None:
            for period in self.seasonality:
                beta = self.posterior['beta'+'_'+period].mean(axis=(0, 1)).to_numpy(
                ) if self.mcmc_samples > 0 else self.posterior['beta'+'_'+period]
                seasonality_posterior = self.seasonality[period].restore_seasonality(
                    beta, None, self.scale, t=future['t'].values)
                y_forecast += seasonality_posterior

        future['y_hat'] = y_forecast.mean(0)

        loc = y_forecast.mean(0)
        scale = self.posterior['sigma']*self.scale+y_forecast.std(0)
        dist = sps.laplace(
            loc, scale) if self.likelihood == 'laplace' else sps.norm(loc, scale)

        future['y_hat_lower'] = dist.ppf(self.p/2)

        future['y_hat_upper'] = dist.ppf(1-self.p/2)

        future['trend'] = trend_forecast.mean(0)
        return future

    def make_plot(self, forecast, next_observation, title, threshold=0.001, fig_path=None):
        """
        Create and optionally save plot with predictions, trend, and changepoints.

        Parameters
        ----------
        forecast : pd.DataFrame
            Dataframe with forecast results from `predict()`.
        next_observation : float
            Actual observed value for comparison.
        title : str
            Title for the plot.
        threshold : float
            Threshold for detecting changepoints.
        fig_path : str, optional
            If provided, saves figure to this path.

        Returns
        -------
        None
        """

        delta = self.posterior['delta'].mean(axis=(0, 1)).to_numpy(
        ) if self.mcmc_samples > 0 else self.posterior['delta']

        s = self.s[np.abs(delta) > threshold]
        self.changepoints = s * \
            (self.df['ds'].max() - self.df['ds'].min()) + \
            self.df['ds'].min()
        changepoints = pd.to_datetime(self.changepoints)

        trend_posterior = forecast['trend'].values

        color = 'k' if forecast.y_hat_lower.values[-1] <= next_observation else 'r'

        plt.figure(figsize=(15, 6))

        plt.plot(forecast.ds.values, forecast['y_hat'].values,
                 label='Prediction', linewidth=1, color='blue')

        plt.fill_between(
            forecast.ds.values, forecast['y_hat_lower'], forecast['y_hat_upper'], color='blue', alpha=0.2)

        plt.plot(forecast.ds.values,
                 np.concatenate(
                     [self.df['y'].values, np.array([next_observation])]),
                 linestyle='--', color='k', label='Observations')

        plt.scatter(forecast.ds.values[-1],
                    np.array(next_observation), color=color, label='Alert')

        plt.plot(forecast.ds.values, trend_posterior, label='Trend',
                 color='red', linewidth=1.5)

        for cp in changepoints:
            plt.axvline(cp, color='red', linestyle='dashed', alpha=0.6)

        plt.xticks(forecast.ds.values, rotation=45)
        plt.xlim(forecast.ds.values.min(),
                 forecast.ds.values.max()+pd.Timedelta(days=5))
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Conversion')
        plt.legend()
        plt.grid(True)
        if fig_path:
            plt.savefig(fig_path)
