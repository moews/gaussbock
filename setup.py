from setuptools import setup

setup(
  name = 'gaussbock',
  packages = ['gaussbock'],
  version = '1.0.5',
  description = 'Fast cosmological parameter estimation with parallel iterative Bayesian nonparametrics',
  long_description='Preliminary documentation can be found in the docstrings, the paper will appear on arXiv shortly',
  author = 'Ben Moews',
  author_email = 'ben.moews@protonmail.com',
  url = 'https://github.com/moews/gaussbock',
  keywords = ['astronomy',
              'astrophysics',
              'cosmology',
              'bayesian',
              'parameter estimation'],
  classifiers = ['Programming Language :: Python :: 3 :: Only',
                 'Intended Audience :: Science/Research',
                 'License :: OSI Approved :: MIT License'],
  install_requires = ['numpy',
                      'emcee >= 2.0.0',
                      'schwimmbad >= 0.3.0',
                      'scikit-learn >= 0.18.1'],
)
