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
