# Gaussbock

#### Fast parallel-iterative cosmological parameter estimation with Bayesian nonparametrics

<img src="/logo.png" alt="logo" width="200px"/>

Gaussbock is a general-purpose tool for parameter estimation with computationaly expensive likelihood calculations, developed for high-dimensional cosmological parameter estimation for the [Dark Energy Survey (DES)](https://www.darkenergysurvey.org/), and with upcoming surveys like the [Large Synoptic Sky Telescope (LSST)](https://www.lsst.org/) and [ESA's Euclid mission](http://sci.esa.int/euclid/) in mind. Current efforts in cosmological parameter estimation often suffer from both the computational costs of approximating distributions in high-dimensional parameter spaces and the wide-spread need for model tuning. Specifically, calculating the likelihoods of parameter proposals through anisotropic simulations of the universe imposes high computational costs, which leads to excessive time requirements per experiment in an era of comparatively cheap parallel computing resources.

### Installation

Gaussbock can be installed via [PyPI](https://pypi.org), with a single command in the terminal:

```
pip install gaussbock
```

Alternatively, the file `gaussbock.py` can be downloaded from the folder `gaussbock` in this repository and used locally by placing the file into the working directory for a given project. An installation via the terminal is, however, highly recommended, as the installation process will check for the package requirements and automatically update or install any missing dependencies, thus sparing the user the effort of troubleshooting and installing them themselves.

### Quickstart guide


<br></br>

| Variables                        | Explanations                                              | Default                  |
|:---------------------------------|:----------------------------------------------------------|:-------------------------|
| parameter_ranges                 | The lower and upper limit for each parameter              |                          |
| posterior_evaluation             | Evaluation function handle for the posterior              |                          |
| output_samples                   | Number of posterior samples that are required             |                          |
| initial_samples (optional)       | Choice of 'emcee' or a provided start sample              | ['automatic', 50, 1000]  |
| gaussbock_iterations (optional)  | Maximum number of Gaussbock iterations                    | 10                       |
| mixture_samples (optional)       | Number of samples drawn for importance sampling           | 1e5                      |
| em_iterations (optional)         | Maximum number of EM iterations for the mixture model     | False                    |
| tolerance_range (optional)       | The range for the shrinking convergence threshold         | [1e-2, 1e-7]             |
| model_components (optional)      | Maximum number of Gaussians fitted to samples             | ⌈(2 / 3) * N⌉            |
| model_covariance (optional)      | Type of covariance for the GMM fitting process            | 'full'                   |
| parameter_init (optional)        | How to intialize model weights, means and covariances     | 'random'                 |
| model_verbosity (optional)       | The amount of information printed during runtime          | 1                        |
| mpi_parallelization (optional)   | Whether to parallelize Gaussbock using an MPI pool        | False                    |
| processes (optional)             | Number of processes Gaussbock should parallelize over     | 1                        |
| weights_and_model (optional)     | Whether to return importance weights and the model        | False                    |
| truncation_alpha (optional)      | Truncation value for importance probability re-weighting  | 2.0                      |
| model_selection (optional)       | Type of model used for the fitting process                | None                     |
| kde_bandwidth (optional)         | Kernel bandwidth used when fitting via KDE                | 0.5                      |

<br></br>
