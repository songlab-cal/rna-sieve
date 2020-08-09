import numpy as np
import pandas as pd
import altair as alt
from rnasieve.algo import find_mixtures
import numbers


class RNASieveModel:
    def __init__(self, observed_phi, observed_sigma, observed_m):
        assert observed_phi.shape == observed_sigma.shape, 'Reference mean matrix and variance matrix must have the same dimensions'
        assert observed_m.shape[1] == observed_phi.shape[1], 'Number of cell types must be consistent with reference matrix'

        self.observed_phi = observed_phi
        self.observed_sigma = observed_sigma
        self.observed_m = observed_m
        self.psis = None
        self.filter_idxs = None
        self.alpha_hats = None
        self.n_hats = None
        self.phi_hat = None

    def predict(self, psis):
        assert psis.shape[0] == self.observed_phi.shape[0], 'Bulks must have the same number of genes as the reference matrix'

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
        self.n_hats = pd.DataFrame(n_hats, index=bulk_labels, columns=['n_hat'])
        self.phi_hat = pd.DataFrame(phi_hat, columns=ref_labels)

        return self.alpha_hats

    def plot_proportions(self, plot_type='bar'):
        alpha_hats_melt = pd.melt(self.alpha_hats.reset_index(), id_vars=['index'])

        if plot_type == 'bar':
            chart = alt.Chart(alpha_hats_melt).mark_bar().encode(
                x=alt.X('variable:N', axis=alt.Axis(title='Cell Type', labels=False)),
                y=alt.Y('value:Q', axis=alt.Axis(title='Proportion')),
                color='variable:N',
                column=alt.Column('index:N', title='Bulk'),
            )

        if plot_type == 'scatter':
            assert isinstance(alpha_hats_melt['index'].iloc[0], numbers.Number), 'Bulk labels must be quantitative'

            avg_line = alt.Chart(alpha_hats_melt).mark_line().encode(
                x=alt.X('index:Q', axis=alt.Axis(title='Bulk Metric')),
                y=alt.Y('mean(value):Q', axis=alt.Axis(title='Proportion')),
                color='variable:N',
                order='index:Q',
            )

            ind_scatter = alt.Chart(alpha_hats_melt).mark_point().encode(
                x=alt.X('index:Q', axis=alt.Axis(title='Bulk Metric')),
                y=alt.Y('value:Q', axis=alt.Axis(title='Proportion')),
                color='variable:N',
            )

            chart = avg_line + ind_scatter

        if plot_type == 'stacked':
            chart = alt.Chart(alpha_hats_melt).mark_bar().encode(
                x=alt.X('index:N', axis=alt.Axis(title='Bulk')),
                y=alt.Y('sum(value):Q', axis=alt.Axis(title='Proportion'), stack='normalize'),
                color='variable:N',
            )

        return chart
