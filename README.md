# Gaussbock

#### Fast parallel-iterative cosmological parameter estimation with Bayesian nonparametrics

<img src="/logo.png" alt="logo" width="200px"/>

Gaussbock is a general-purpose tool for parameter estimation with computationally expensive likelihood calculations, developed for high-dimensional cosmological parameter estimation for the [Dark Energy Survey (DES)](https://www.darkenergysurvey.org/), and with upcoming surveys like the [Large Synoptic Sky Telescope (LSST)](https://www.lsst.org/) and [ESA's Euclid mission](http://sci.esa.int/euclid/) in mind. Current efforts in cosmological parameter estimation often suffer from both the computational costs of approximating distributions in high-dimensional parameter spaces and the wide-spread need for model tuning. Specifically, calculating the likelihoods of parameter proposals through anisotropic simulations of the universe imposes high computational costs, which leads to excessive time requirements per experiment in an era of comparatively cheap parallel computing resources.

Making use of parallelization and an iterative approach to Bayesian nonparametrics, the provided method starts with a set of data points from a distribution that roughly approximates the true posterior. In order to obtain such a sample, the code either uses an affine-invariant MCMC ensemble as intoduced by [Goodman and Weare (2004)](https://projecteuclid.org/euclid.camcos/1513731992) via [emcee](http://dfm.io/emcee/current/) or an initial set of data points provided by the user. After that, a variational Bayesian non-parametric Gaussian mixture model, which makes use of an infinite Dirichlet process mixtures approximated with a stick-breaking representation (see [Blei and Jordan, 2006](https://projecteuclid.org/euclid.ba/1340371077)), is fitted to the data points and  aconsistent number of data points is sampled from the model.

These data points are then re-sampled via probabilities derived from truncated importance sampling, an extension of importance sampling less sensitive to proposal distributions to avoid dominating high-posterior samples and introduced by [Ionides (2008)](https://amstat.tandfonline.com/doi/abs/10.1198/106186008X320456), for which the computation is spread over the provided cores as an embarrassingly parallel problem. The model fitting and importance sampling steps are then iteratively repeated, shifting towards an ever-narrower fit around the true posterior distribution, and leading to a significant speed-up in comparison to other established cosmological parameter estimation approaches. At the end, Gaussbock returns a user-specified number of samples and, if the user wishes, the importance weights and the final model itself, allowing for the sampling of an unlimited number of data points and to investigate the importance probabilities to ensure a reasonable distribution of the latter.

### Installation

Gaussbock can be installed via [PyPI](https://pypi.org), with a single command in the terminal:

```
pip install gaussbock
```

Alternatively, the file `gaussbock.py` can be downloaded from the folder `gaussbock` in this repository and used locally by placing the file into the working directory for a given project. An installation via the terminal is, however, highly recommended, as the installation process will check for the package requirements and automatically update or install any missing dependencies, thus sparing the user the effort of troubleshooting and installing them themselves.

### Quickstart guide

Only three inputs are required by Gaussbock: The list of parameter ranges, each as a tuple with the lower and upper limit for each parameter (`parameter_ranges`), the handle for a posterior function to evaluate samples (`posterior_evaluation`), and the required number of posterior samples in the output (`output_samples`).

A vaiety of additional inputs can bet set, but are not required. The user can decide whether to use an affine-invariate MCMC ensemble to obtain an initial rough approximation of the posterior, or whether to provide their own initial sample (`initial_samples`), which takes either the form `['automatic', int, int]` for the MCMC ensemble, with the integer referring to the number of walker and steps per walker, respectively, or `['custom', array-like]`, with the array-like object providing the initial samples in the parameter space. The maximum number of Gaussbock iterations (`gaussbock_iterations`) can be set as well as the number of samples drawn from the current posterior approximation before each truncated importance sampling step (`mixture_samples`) and the maximum number of expectation-maximization (EM) iterations to fit the variational Bayesian mixture model (`em_iterations`).



The required and optional inputs are listed below, with _D_ denoting the dimensionality of the parameter estimation problem, or the number of parameters.


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
| model_components (optional)      | Maximum number of Gaussians fitted to samples             | ceiling((2 / 3) * _D_)   |
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

After the installation via [PyPI](https://pypi.org), or using the `gaussbock.py` file locally, the usage looks like this:

```python
from gaussbock import gaussbock

output = gaussbock(parameter_ranges = your_posterior_ranges,
                   posterior_evaluation = your_posterior_function,
                   output_samples = your_required_samples,
                   mpi_parallelization = True,
                   weights_and_model = True)

samples, weights, model = output
                   
```

Note that, in the above example, we use two of the optional parameters to tell the tool to parallelize using MPI, for example for the use on a supercomputer, and to return the importance weights and the model, for example for weighting the returned samples and saving the model to draw further samples later. If we wouldn't set the weights and model return indicator to be true, there would be no need to split the output up, as the output would be just the list of samples.
