# RNA-Sieve

RNA-Sieve is a method for the deconvolution of bulk cell samples via single-cell RNA expression data.

![RNA-sieve-viz](https://raw.githubusercontent.com/songlab-cal/rna-sieve/v_0_1_1/readme_figures/deconvolution_plot_3d.png)

## Associated Work
Our preprint is currently available at [https://www.biorxiv.org/content/10.1101/2020.10.01.322867v1](https://www.biorxiv.org/content/10.1101/2020.10.01.322867v1).

If you find RNA-Sieve useful, please cite our work at:
> *RNA-Sieve: A likelihood-based deconvolution of bulk gene expression data using single-cell references*<br />
> <small>Dan D. Erdmann-Pham, Jonathan Fischer, Justin Hong, Yun S. Song<br /></small>
> bioRxiv 2020.10.01.322867; doi: [https://doi.org/10.1101/2020.10.01.322867](https://doi.org/10.1101/2020.10.01.322867)

## Installation

For Python 3, we recommend that you install `rna-sieve` via `pip`.
```bash
$ pip3 install rna-sieve
```

## Example Usage

![muscle-FD-example](https://raw.githubusercontent.com/songlab-cal/rna-sieve/v_0_1_1/readme_figures/muscle_age_FD.png)

For example usage, please reference [the example Jupyter notebook](https://github.com/songlab-cal/rna-sieve/blob/master/examples/example.ipynb) for Python 3 usage,
or [the Mathematica notebook](https://github.com/songlab-cal/rna-sieve/blob/master/mathematica/rnasieve.nb) for Mathematica usage.

The core algorithm is called `find_mixtures`/`findMixtures` which takes in a vector of bulk expressions to be deconvolved and reference matrices of means, variances, and sample counts.
Currently, only the Python library takes allows for multiple bulks to be jointly deconvolved with a single set of reference matrices.
