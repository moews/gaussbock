from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
  name = 'gaussbock',
  packages = ['gaussbock'],
  version = '1.0.9',
  description = 'Fast parallel-iterative cosmological parameter estimation with Bayesian nonparametrics',
  long_description = long_description,
  long_description_content_type = 'text/markdown',
  author = 'Ben Moews and Joe Zuntz',
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
