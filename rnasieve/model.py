"""
Defines a class that can be used to run RNA-Sieve in an object-oriented manner.
"""
import numpy as np
import pandas as pd
import altair as alt
from rnasieve.algo import find_mixtures
from rnasieve.stats import compute_marginal_confidence_intervals as compute_marginal_cis
import numbers


class RNASieveModel:
    """
    RNASieveModel is a class that wraps the core algorithm of RNA-Sieve, and provides
    additional instance functions for plotting and computing confidence intervals.

    Initialize with reference data (phi, sigma, m) and run predict on bulk vector(s).
    """

    def __init__(self, observed_phi, observed_sigma, observed_m):
        assert observed_phi.shape == observed_sigma.shape, 'Reference mean matrix and variance matrix must have the same dimensions'
        assert observed_m.shape[1] == observed_phi.shape[1], 'Number of cell types must be consistent with reference matrix'

        self.observed_phi = observed_phi
        self.observed_sigma = observed_sigma
        self.observed_m = observed_m
        self._reset()

    def _reset(self):
        self.psis = None
        self.filter_idxs = None
        self.alpha_hats = None
        self.n_hats = None
        self.phi_hat = None
        self.marginal_ci_errs = None

    def predict(self, psis):
        assert psis.shape[0] == self.observed_phi.shape[0], 'Bulks must have the same number of genes as the reference matrix'
        self._reset()

        self.filter_idxs = np.where(psis.any(axis=1))
        alpha_hats, n_hats, phi_hat = find_mixtures(
            self.observed_phi.to_numpy()[self.filter_idxs],
            self.observed_sigma.to_numpy()[self.filter_idxs],
            self.observed_m.to_numpy(),
            psis.to_numpy()[self.filter_idxs])[0]

        ref_labels = self.observed_phi.columns.values.tolist()
        bulk_labels = psis.columns.values.tolist()
        self.psis = psis
        self.alpha_hats = pd.DataFrame(data=alpha_hats,
                                       index=bulk_labels, columns=ref_labels)
        self.n_hats = pd.DataFrame(
            n_hats, index=bulk_labels, columns=['n_hat'])
        self.phi_hat = pd.DataFrame(phi_hat, columns=ref_labels)

        return self.alpha_hats

    def compute_marginal_confidence_intervals(self, sig=.05):
        assert self.alpha_hats is not None, 'Please run predict before computing CIs'

        marginal_ci_errs = np.empty(
            (0, self.alpha_hats.shape[1]), dtype=np.float32)
        marginal_cis = []
        for i in range(self.alpha_hats.shape[0]):
            marginal_ci = compute_marginal_cis(
                self.phi_hat.to_numpy(),
                self.observed_sigma.to_numpy()[self.filter_idxs],
                self.alpha_hats.to_numpy()[i].reshape(1, -1),
                self.n_hats.to_numpy()[i],
                self.psis.to_numpy()[self.filter_idxs, i].reshape(-1, 1),
                sig
            )
            marginal_cis.append(marginal_ci)
            marginal_ci_errs = np.vstack((marginal_ci_errs, np.array(
                [(hi - low) / 2 for low, hi in marginal_ci])))

        ref_labels = self.observed_phi.columns.values.tolist()
        bulk_labels = self.psis.columns.values.tolist()
        self.marginal_ci_errs = pd.DataFrame(
            marginal_ci_errs, index=bulk_labels, columns=ref_labels)

        return marginal_cis

    def plot_proportions(self, plot_type='bar'):
        alpha_hats_melt = pd.melt(
            self.alpha_hats.reset_index(),
            id_vars=['index'],
            var_name='cell_type',
            value_name='proportion')

        if self.marginal_ci_errs is not None:
            marginal_ci_errs_melt = pd.melt(
                self.marginal_ci_errs.reset_index(),
                id_vars=['index'],
                var_name='cell_type',
                value_name='ci_err')
            alpha_hats_melt = pd.merge(
                alpha_hats_melt, marginal_ci_errs_melt, left_on=[
                    'index', 'cell_type'], right_on=[
                    'index', 'cell_type'])
            alpha_hats_melt['ci_low'] = alpha_hats_melt.apply(lambda row: np.clip(
                row['proportion'] - row['ci_err'], 0, 1), axis=1)
            alpha_hats_melt['ci_high'] = alpha_hats_melt.apply(lambda row: np.clip(
                row['proportion'] + row['ci_err'], 0, 1), axis=1)

        if plot_type == 'bar':
            bars = alt.Chart().mark_bar().encode(
                x=alt.X(
                    'cell_type:N', axis=alt.Axis(
                        title='Cell Type', labels=False)), y=alt.Y(
                    'proportion:Q', axis=alt.Axis(
                        title='Proportion')), color='cell_type:N', )

            if self.marginal_ci_errs is not None:
                error_bars = alt.Chart().mark_errorbar().encode(
                    x=alt.X(
                        'cell_type:N',
                        axis=alt.Axis(
                            title='Cell Type',
                            labels=False)),
                    y='ci_low:Q',
                    y2='ci_high:Q',
                )
                chart = alt.layer(
                    bars, error_bars, data=alpha_hats_melt).facet(
                    column=alt.Column(
                        'index:N', title='Bulk'), )
            else:
                chart = alt.layer(bars, data=alpha_hats_melt).facet(
                    column=alt.Column('index:N', title='Bulk'),
                )

        if plot_type == 'scatter':
            assert isinstance(
                alpha_hats_melt['index'].iloc[0], numbers.Number), 'Bulk labels must be quantitative'

            avg_line = alt.Chart(alpha_hats_melt).mark_line().encode(
                x=alt.X(
                    'index:Q', axis=alt.Axis(
                        title='Bulk Metric')), y=alt.Y(
                    'mean(proportion):Q', axis=alt.Axis(
                        title='Proportion')), color='cell_type:N', order='index:Q', )

            ind_scatter = alt.Chart(alpha_hats_melt).mark_point().encode(
                x=alt.X('index:Q', axis=alt.Axis(title='Bulk Metric')),
                y=alt.Y('proportion:Q', axis=alt.Axis(title='Proportion')),
                color='cell_type:N',
            )

            chart = avg_line + ind_scatter

        if plot_type == 'stacked':
            chart = alt.Chart(alpha_hats_melt).mark_bar().encode(
                x=alt.X(
                    'index:N', axis=alt.Axis(
                        title='Bulk')), y=alt.Y(
                    'sum(proportion):Q', axis=alt.Axis(
                        title='Proportion'), stack='normalize'), color='cell_type:N', )

        return chart
