# Gaussbock

#### Fast parallel-iterative cosmological parameter estimation with Bayesian nonparametrics

<img src="/logo.png" alt="logo" width="200px"/>

Gaussbock is a general-purpose tool for parameter estimation with computationaly expensive likelihood calculations, which was developed for high-dimensional cosmological parameter estimation for the [Dark Energy Survey (DES)](https://www.darkenergysurvey.org/), and with upcoming surveys like the [Large Synoptic Sky Telescope (LSST)](https://www.lsst.org/) and [ESA's Euclid mission](http://sci.esa.int/euclid/) in mind. Current efforts in cosmological parameter estimation often suffer from both the computational costs of approximating distributions in high-dimensional parameter spaces and the wide-spread need for model tuning. Specifically, calculating the likelihoods of parameter proposals through anisotropic simulations of the universe imposes high computational costs, leading to excessive time requirements per experiment in an era of comparatively cheap parallel computing resources.

