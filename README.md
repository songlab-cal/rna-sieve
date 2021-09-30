# RNA-Sieve

RNA-Sieve is a method for the deconvolution of bulk cell samples via single-cell RNA expression data.

![RNA-sieve-viz](https://raw.githubusercontent.com/songlab-cal/rna-sieve/v_0_1_1/readme_figures/deconvolution_plot_3d.png)

## Associated Work
Our work has been published in [Genome Research](https://genome.cshlp.org/content/early/2021/07/22/gr.272344.120).
The manuscript is also available on [bioRxiv](https://www.biorxiv.org/content/10.1101/2020.10.01.322867v1).

If you find RNA-Sieve useful, please cite our work at ([Google Scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&authuser=1&q=A+likelihood-based+deconvolution+of+bulk+gene+expression+data+using+single-cell+references&btnG=#d=gs_cit&u=%2Fscholar%3Fq%3Dinfo%3AKSRCjNjl8Y4J%3Ascholar.google.com%2F%26output%3Dcite%26scirp%3D0%26hl%3Den%26authuser%3D1)):
> Erdmann-Pham, D. D., Fischer, J., Hong, J., & Song, Y. S. (2021).<br />
> *A likelihood-based deconvolution of bulk gene expression data using single-cell references*.<br />
> Genome Research, gr-272344.

## Installation

For Python 3, we recommend that you install `rnasieve` via `pip`.
```bash
$ pip3 install rnasieve
```

## Example Usage

![muscle-FD-example](https://raw.githubusercontent.com/songlab-cal/rna-sieve/v_0_1_1/readme_figures/muscle_age_FD.png)

For example usage, please reference [the example Jupyter notebook](https://github.com/songlab-cal/rna-sieve/blob/master/examples/example.ipynb) for Python 3 usage,
or [the Mathematica notebook](https://github.com/songlab-cal/rna-sieve/blob/master/mathematica/rnasieve.nb) for Mathematica usage.

The core algorithm is called `find_mixtures`/`findMixtures` which takes in a vector of bulk expressions to be deconvolved and reference matrices of means, variances, and sample counts.
Currently, only the Python library takes allows for multiple bulks to be jointly deconvolved with a single set of reference matrices.
