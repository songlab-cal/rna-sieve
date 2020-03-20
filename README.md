# BEDLAM

This repo contains scripts for deconvolution of bulk cell samples via bulk RNA expression. For example usage, run the jupyter notebook called examples/example.ipynb.

Input format to find_mixtures (with G = number of genes, K = number of cell types, B = number of bulks to be deconvolved):
  - phi: G x K matrix populated with the mean RNA expression for a given gene g, and cell type k for each phi(g,k).
  - sigma: G x K matrix populated with the sample variances for a given gene g, and cell type k for each sigma(g,k).
  - m: 1 x K matrix populated with the number of cells used for the phi matrix reference for each cell type k.
  - psis: G x B matrix populated with the bulk RNA expression for a sample which will be deconvolved.

Additional paramters:
  - eps: Threshold parameter for determining convergence of optimization steps for alpha, the proportion estimates.
  - delta: Threshold parameter for determining termination of likelihood optimization.
  - max_iter: Maximum iterations for a single alpha optimization cycle.
  - uniform_init: Initialization with a uniform proportion estimate rather than the NNLS estimate.
  - parallelized: Flag for paralellization during phi minimization steps.
  - num_process: Number of threads used for parallelization if parallelized = True.

Output format:
  - alphaLS: B x K matrix where the ith row is the proportion estimate vector for the ith bulk.
