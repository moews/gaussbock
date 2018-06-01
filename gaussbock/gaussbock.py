"""Fast cosmological parameter estimation with parallel iterative Bayesian nonparametrics.

Introduction:
-------------

'gaussbock' is a tool for fast cosmological parameter estimation that is geared towards alleviating
the factor of computationally expensive likelihoods. Making use of parallelization and an iterative
approach to Bayesian nonparametrics, it's a general-purpose tool that can be used for problems outside
of the field it was originally developed for; i.e. any kind of problem in need of an approximation to
the posterior distribution, and especially when the calculation of likelihoods is a limiting factor.

The provided method starts with a set of data points from a distribution that roughly approximates the
true posterior. In order to obtain such a sample, the code either uses an affine-invariant MCMC ensemble
or an initial set of data points provided by the user. After that, a variational Bayesian non-parametric
Gaussian mixture model, which makes use of an infinite Dirichlet process mixtures approximated with a
stick-breaking representation, is fitted to the data points and  aconsistent number of data points is
sampled from the model. These data points are then re-sampled via probabilities derived from truncated
importance sampling, an extension of importance sampling less sensitive to proposal distributions to
avoid dominating high-posterior samples, for which the computation is spread over the provided cores as an
embarrassinglyparallel problem to solve one of the central issues in today's high-dimensional cosmology.

The two model fitting and importance sampling steps are then iteratively repeated for a specified number
of iterations, shifting towards an ever-narrower fit around the true posterior distribution and leading
to a significant speed-up in comparison to other established cosmological parameter estimation approaches.
At the end, and depending on the user's wishes, 'gausbock' returns either a specified number of samples
from the final approximation of the posterior, or additionally both the final step's fitted model and the
importance probabilities for the returned samples, allowing the user to sample an unlimited number of data
points and to investigate the importance probabilities to ensure a reasonable distribution of the latter.

If you use 'gaussbock' or derive methodology or implementations from it, please cite the authors listed in
this docstring and the paper, which will appear on arXiv shortly and, subsequently, in a suitable journal.

Quickstart:
-----------
If you want to start quickly, 'gaussbock' only NEEDS three inputs, although there are more than that:

(1) The set of allowed parameter ranges, one per parameter/dimension (name: "parameter_ranges")
(2) The function that takes a data point and returns a log-posterior (name: "posterior_evaluation")
(3) The number of posterior-distributed samples that you want returned (name: "output_samples")

To start using 'gaussbock', simply use "from gaussbock import gaussbock" to access it as a function. An
example with the three mentioned parameters, which uses the adaptive defaults, looks like this:

    ----------------------------------------------------------------
    |  from gaussbock import gaussbock                             |
    |                                                              |
    |  output = gaussbock(parameter_ranges = your_allowed_ranges,  |
    |                     posterior_evaluation = your_function,    |
    |                     output_samples = your_intended_number)   |
    |                                                              |
    ----------------------------------------------------------------

There are 15 additional parameters that you can access, e.g. for tuning the model fitting and sampling
process, to provide your own initial samples to make use of a significant speed-up, to return the final
fitted Gaussian mixture and its importance probabilities in addition to the samples, or for specifying
whether you wish to use multi-core processing or MPI for cluster and supercomputing. These are described
under "parameters" in the docstring of the 'gaussbock()' function right below this introductory text.

One parameter that is recommended to be specified in addition to the required ones is "model_components",
which defaults to the rounded-up product of 2 / 3 and the number of dimensions of the provided problem.
If you're able to estimate the complexity of the posterior distribution, set it to an integer representing
the maximal number of models to be fitted to approximate the posterior distribution, which will lead to a
further significant speed-up and better fit, e.g. for a lower number for low-dimensional simple problems.

Authors:
--------
Ben Moews & Joe Zuntz
Institute for Astronomy (IfA)
School of Physics & Astronomy
The University of Edinburgh

Libraries:
----------
The variational Bayesian non-parametric Gaussian mixture model used in this implementation utilizes
'scikit-learn' [1] (http://scikit-learn.org), a free and open-source software machine learning library.

The affine-invariant Markov chain Monte Carlo ensemble sampler for the generation of an initial sample
set is 'emcee' [2], a pure-Python implementation of a previous paper's proposal on the subject [3].

The parallelization of this code makes use of 'schwimmbad' [4], a , an interface to parallel processing
pools that includes the ability to use MPI for deployment on computing clusters and supercomputers.

'NumPy' [5] is a wide-spread Python library for operations on multi-dimensional arrays and matrices, as
well as high-level mathematical functions for the latter, and probably doesn't need any introduction.

References:
-----------
[1] Pedregosa, F. et al. (2011), "Scikit-learn: Machine Learning in Python", JMLR, 12:2825-2830
[2] Foreman-Mackey et al. (2013), "emcee: The MCMC Hammer" Publ. Astron. Soc. Pac., 12:306-312
[3] Goodman, J. and Weare, J. (2010), "Ensemble samplers with affine invariance" CAMCoS, 5:65-80
[4] Price-Whelan et al. (2017), "schwimmbad: A uniform interface to parallel [...]", JOSS, 2:00357
[5] van der Walt, S. et al. (2011), "The NumPy Array: A Structure for [...]", CS&E, 13:22-30

Versions:
---------
The versions listed below were used in the development of 'gaussbock', but aren't specifically required.

Python 3.4.5
Scikit-learn 0.19.1
emcee 2.2.1
schwimmbad 0.3.0
NumPy 1.14.0
"""
# Import the necessary libraries
import numpy as np
import emcee as ec
from schwimmbad import SerialPool, MultiPool, MPIPool
from sklearn.mixture import BayesianGaussianMixture as BGM
from sklearn.neighbors import KernelDensity as KD


class Gaussbock:
    def __init__(self, parameter_ranges,
              posterior_evaluation,
              output_samples,
              initial_samples = ['automatic', 50, 1000],
              gaussbock_iterations = 10,
              mixture_samples = 10000,
              em_iterations = 1000,
              tolerance_range = [1e-2, 1e-7],
              model_components = None,
              model_covariance = 'full',
              parameter_init = 'random',
              model_verbosity = 1,
              mpi_parallelization = False,
              processes = 1,
              truncation_alpha = 2.0,
              model_selection = None,
              kde_bandwidth = 0.5,
              pool = None):
        self.parameter_ranges = parameter_ranges
        self.posterior_evaluation = posterior_evaluation
        self.output_samples = output_samples
        self.initial_samples = initial_samples
        self.gaussbock_iterations = gaussbock_iterations
        self.mixture_samples = mixture_samples
        self.em_iterations = em_iterations
        self.tolerance_range = tolerance_range
        self.model_components = model_components
        self.model_covariance = model_covariance
        self.parameter_init = parameter_init
        self.model_verbosity = model_verbosity
        self.mpi_parallelization = mpi_parallelization
        self.processes = processes
        self.truncation_alpha = truncation_alpha
        self.model_selection = model_selection
        self.kde_bandwidth = kde_bandwidth
        self.pool = pool_type(mpi_parallelization = self.mpi_parallelization, 
                              processes = self.processes, 
                              pool=pool)

        # We should close the pool if we made it ourself and it is an MPI pool
        self.should_close_pool = (pool is None) and (self.pool is not None) and self.mpi_parallelization

        self.parameter_number = self.parameter_ranges.shape[0]

        # If not chosen, set the type of model for fitting depending on the number of dimensions
        if self.model_selection == None:
            if self.parameter_number <= 2:
                self.model_selection = 'kde'
            else:
                self.model_selection = 'gmm'
        # Assign a reasonable default value to be used as the maximal number of models to fit
        if self.model_components == None:
            self.model_components = int(np.ceil((2 / 3) * self.parameter_number))



    def parameter_check(self):
        """Check all parameters for conformance with the required parameter specifications.

        Sometimes, users might provide parameters in formats or with values that aren't
        expected and would, therefore, lead to a crash. In order to prevent waiting for the
        code to reach such a breaking point, all parameters are checked in the beginning.
        The code will terminate with a descriptive error message in case of discrepancies.
        """
        # Create a vector of boolean values to keep track of incorrect inputs
        incorrect_inputs = np.zeros(19, dtype = bool)
        # Check if the parameter 'parameter_ranges' satisfies the specifications
        if not (type(self.parameter_ranges) == np.ndarray and self.parameter_ranges.shape[1] == 2
                and np.all([self.parameter_ranges[i, 0] < self.parameter_ranges[i, 1]
                for i in range(0, self.parameter_ranges.shape[1])])):
            incorrect_inputs[0] = True
        # Check if the parameter 'posterior_evaluation' is a suitable function
        try:
            # Extract the provided minimum and maximum allowed values for all of the parameters
            low, high = self.parameter_ranges[:, 0], self.parameter_ranges[:, 1]
            # Draw a random test data point for the function from the allowed parameter ranges
            test_point = np.random.uniform(low = low, high = high)
            try:
                test_posterior = self.posterior_evaluation(test_point)
            except:
                incorrect_inputs[1] = True
        except:
            # Let the function resume, as a faulty 'parameter_ranges' will terminate
            pass
        # Check if the provided output parameter 'output_samples' is an integer
        if not (type(self.output_samples) == int and self.output_samples > 0):
            incorrect_inputs[2] = True
        # Check if the parameter 'input_samples' is withing the allowed set
        if not self.initial_samples[0] in ['automatic', 'custom']:
            incorrect_inputs[3] = True
        if self.initial_samples[0] == 'automatic' and (len(self.initial_samples) != 3
            or [type(self.initial_samples[i]) for i in range(0,3)] != [str, int, int]):
            incorrect_inputs[3] = True
        elif self.initial_samples[0] == 'custom' and (len(self.initial_samples) != 2
            or [type(self.initial_samples[i]) for i in range(0,2)] != [str, np.ndarray]):
            incorrect_inputs[3] = True
        # Check if the provided parameter 'gaussbock_iterations' is an integer
        if not (type(self.gaussbock_iterations) == int and self.gaussbock_iterations > 0):
            incorrect_inputs[4] = True
        # Check if the parameter 'mixture_samples' for the model is an integer
        if not (type(self.mixture_samples) == int and self.mixture_samples > 0):
            incorrect_inputs[5] = True
        # Check if the parameter 'em_iterations' for the model is an integer
        if not (type(self.em_iterations) == int and self.em_iterations > 0):
            incorrect_inputs[6] = True
        # Check if the parameter 'tolerance_range' satisfies the specifications
        if not (type(self.tolerance_range) == list and self.tolerance_range[0] > self.tolerance_range[1]):
            incorrect_inputs[7] = True
        # Check if the parameter 'model_components' for the model is an integer
        if not ((type(self.model_components) == int and self.model_components > 0) or self.model_components == None):
            incorrect_inputs[8] = True
        # Check if the parameter 'model_covariance' is within the allowed set
        if not self.model_covariance in ['full', 'tied', 'diag', 'spherical']:
            incorrect_inputs[9] = True
        # Check if the parameter 'parameter_init' is within the allowed set
        if not self.parameter_init in ['kmeans', 'random']:
            incorrect_inputs[10] = True
        # Check if the parameter 'model_verbosity' is within the allowed set
        if not self.model_verbosity in [0, 1, 2]:
            incorrect_inputs[11] = True
        # Check if the parameter 'mpi_parallelization' is within the allowed set
        if not self.mpi_parallelization in [True, False]:
            incorrect_inputs[12] = True
        # Check if the parameter 'processes' for parallelization is an integer
        if not (type(self.processes) == int and self.processes > 0):
            incorrect_inputs[13] = True
        # Check if the parameter 'weights_and_model' is within the allowed set
        # if not self.weights_and_model in [True, False]:
        #     incorrect_inputs[14] = True
        # Check whether the number of walkers is at least twice the parameter number
        if (self.initial_samples[0] == 'automatic' and len(self.initial_samples) == 3
            and [type(self.initial_samples[i]) for i in range(0,3)] == [str, int, int]):
            try:
                if not self.initial_samples[1] > (2 * len(self.parameter_ranges)):
                    incorrect_inputs[15] = True
            except:
                # Let the function resume, as a faulty 'parameter_ranges' will terminate
                pass
        # Check whether the truncation parameter falls within the allowed range
        if not (type(self.truncation_alpha) == float and (0.0 <= self.truncation_alpha <= 3.0)):
            incorrect_inputs[16] = True
        # Check whether the model selection parameter is within the allowed set
        if not self.model_selection in [None, 'gmm', 'kde']:
            incorrect_inputs[17] = True
        # Check whether the KDE bandwidth parameter is a positive float value
        if not (type(self.kde_bandwidth) == float and self.kde_bandwidth > 0.0):
            incorrect_inputs[18] = True
        # Define error messages for each parameter in case of unsuitable inputs
        errors = ['ERROR: parameter_ranges: 2-column numpy.ndarray, first column lower bounds',
                  'ERROR: posterior_evaluation: Must be a suitable function for this problem',
                  'ERROR: output_samples: Must be an integer above 0 to be a valid input',
                  'ERROR: initial_samples: ["automatic", int, int] or ["custom", array-like]',
                  'ERROR: gaussbock_iterations: Must be an integer above 0 to be a valid input',
                  'ERROR: mixture_samples: Must be an integer above 0 to be a valid input',
                  'ERROR: em_iterations: Must be an integer above zero to be a valid input',
                  'ERROR: tolerance_range: List of the form [float, float], first value higher',
                  'ERROR: model_components: Must be an integer above 0 to be a valid input',
                  'ERROR: model_covariance: Must be from {"full", "tied", "diag", "spherical"}',
                  'ERROR: parameter_init: Must be from {"kmeans", "random"} to be a valid input',
                  'ERROR: model_verbosity: Must be from {0, 1, 2} to select the information level',
                  'ERROR: mpi_parallelization: Must be from {True, False} to indicate MPI usage',
                  'ERROR: processes: Must be an integer above 0 to be a valid input',
                  'ERROR: weights_and_model: Must be from {True, False} to indicate the preference',
                  'ERROR: initial_samples: No. of walkers must be > twice the number of parameters',
                  'ERROR: truncation_alpha: Must be a float from [0.0, 3.0] to be a valid input',
                  'ERROR: model_selection: Must be from {"gmm", "kde"} to be a valid input',
                  'ERROR: kde_bandwidth: Must be a float above 0 to be a valid input']
        # If there are any unsuitable inputs, print error messages and terminate
        if any(incorrect_inputs):
            errs = []
            for i in range(0, len(errors)):
                if incorrect_inputs[i]:
                    errs.append(errors[i])
            raise ValueError("\n".join(errs))

    def starting_point(self):
        # In case of using MPI, let worker processes wait for tasks from the master process
        # Check whether to get initial samples via emcee or use a set
        if self.initial_samples[0] == 'automatic':

            emcee_walkers, emcee_steps = self.initial_samples[1:3]
            # Call the emcee-based function to get a set of MCMC samples
            print('PROCESS: Getting initial samples via an affine-invariant MCMC ensemble ...')
            print('--------------------------------------------------------------------------\n')
            samples = self.initial_emcee(
                                      emcee_walkers = emcee_walkers,
                                      emcee_steps = emcee_steps
                                      )
            print('===> DONE\n')
        elif self.initial_samples[0] == 'custom':
            print('NOTE: Using the user-provided set of initial samples\n')
            samples = self.initial_samples[1]
        return samples


    def run(self, initial_samples, weights_and_model=True):
        """
        Run the complete sampling process
        """
        # Test all parameters for validity and terminate if not true
        self.initial_samples = initial_samples
        self.parameter_check()
        # Get the number of dimensions that the algorithm operates in for the provided problem
        
        if (self.pool is not None) and self.mpi_parallelization:
            if not self.pool.is_master():
                pool.wait()
                return

        samples = self.starting_point()


        print('NOTE: Starting gaussbock iterations\n')
        # Loop over the provided number of iterations minus the last one
        for i in range(self.gaussbock_iterations - 1):
            print('ITERATION %d\n' % i)
            print('PROCESS: Fitting the model for iteration %d ...' % i)
            print('----------------------------------------------\n')
            samples = self.iterate(i, samples)

        return self.final_iteration(samples, weights_and_model=weights_and_model)


    def final_iteration(self, samples, weights_and_model=True):
        # Fit the final model to the data points provided by the loop
        # We do one final iteration so we can potentially generate more samples here.
        print('PROCESS: Fitting the final model ...')
        print('------------------------------------\n')
        tolerance = self.tolerance_range[1]
        mixture_model = self.mixture_fit(samples, tolerance)
        print('\n===> DONE\n')

        # Sample 150% of the required amount of samples from the model
        if self.model_selection == 'gmm':
            samples = mixture_model.sample(n_samples = self.output_samples * 1.5)[0]
        if self.model_selection == 'kde':
            samples = mixture_model.sample(n_samples = int(np.round(self.output_samples * 1.5)))
        # Cut all the data points falling outside of the allowed ranges
        samples = self.range_cutoff(samples)

        importance_weights = self.importance_sampling(samples, mixture_model, return_weights=True)

        # Extract the required amount of samples from the now clean set
        samples = samples[0:self.output_samples, :]
        print('PROCESS: Checking and preparing returns ...')
        print('-------------------------------------------\n')
        # Check whether to return the importance weights and the model

        if weights_and_model:
            print('NOTE: Successful termination; samples, weights and model are being returned')
            results = (samples, importance_weights, mixture_model)

        else:
            print('NOTE: Successful termination; samples are being returned')
            results = samples

        if self.should_close_pool:
            self.pool.close()


        return results

    def initial_emcee(self, emcee_walkers, emcee_steps):
        """
        Create an initial set of data points with an affine-invariant MCMC ensemble sampler.

        While the iteartive method utilized in this code also works for an initial sample that
        is drawn uniformly at random from high-dimensional parameter ranges, convergence of the
        approximations towards a good fit to the true posterior distribution takes some time.
        For this reason, an affine-invariant MCMC ensemble sampler is used to generate the first
        set of data points that represent a very rough approximation to the true posterior.

        Parameters:
        -----------
        emcee_walkers : int
            The number of separate walkers that emcee should deploy to gather first samples.

        emcee_steps : int
            The number of steps in a chain that emcee walkers should take before termination.

        Returns:
        --------
        samples : array-like
            An initial set of distribution-approximating samples as chains of MCMC walkers.

        Attributes:
        -----------
        None
        """
        # Extract the provided minimum and maximum allowed values for all of the parameters
        low, high = self.parameter_ranges[:, 0], self.parameter_ranges[:, 1]
        # Generate starting points for emcee by drawing uniformly-distributed random samples
        starting_points = [np.random.uniform(low = low, high = high) for i in range(emcee_walkers)]
        # Get the number of parameters, i.e. the number of dimensions, as an input for emcee
        # Create a sampler with the number of walkers, dimensions and evaluation function
        emcee_sampler = ec.EnsembleSampler(nwalkers = emcee_walkers,
                                           dim = self.parameter_number,
                                           lnpostfn = self.posterior_evaluation,
                                           pool = self.pool)
        # Run the sampler with the generated starting points and a step number per walker
        emcee_sampler.run_mcmc(pos0 = starting_points,
                               N = emcee_steps)
        # Access the sampler's chain flattened to the number of chain links and the dimension
        emcee_samples = emcee_sampler.flatchain
        return emcee_samples

    def iterate(self, i, samples):
        # Get the model-fitting tolerance level for the iteration
        step_tolerance = self.dynamical_tolerance(i)
        # Fit the model for the current data points and tolerance
        mixture_model = self.mixture_fit(samples, step_tolerance)

        # Sample a novel set of data points from the fitted model
        if self.model_selection == 'gmm':
            samples = mixture_model.sample(n_samples = self.mixture_samples * 1.5)[0]
        elif self.model_selection == 'kde':
            samples = mixture_model.sample(n_samples = int(np.round(self.output_samples * 1.5)))
        # Cut all of the data points outside of the allowed ranges
        samples = self.range_cutoff(samples)
        
        # Extract the required amount of samples from the now clean set
        print("JAZ NOT SURE ABOUT THIS BIT - DOES IT HURT TO HAVE EXTRA SAMPLES?")
        samples = samples[0:self.output_samples, :]
        
        # Generate a new set of equally weighted samples
        print('PROCESS: Importance sampling for re-weighted frequencies ...')
        print('------------------------------------------------------------\n')
        samples = self.importance_sampling(samples,mixture_model, return_weights=False)
        print('===> DONE\n')
        return samples


    def range_cutoff(self, samples):
        """Cut data points which fall outside of allowed parameter ranges from a data set.

        As some user-provided evaluation functions for calculating posterior probabilites of
        single data points might prohibit values outside of certain ranges, e.g. by resulting
        in minus infinity as the returned value, this function cuts all data points from a
        set of data points with such a parameter range violation for one or more variables.

        Parameters:
        -----------
        samples : array-like
            The set data points from which the ones outside the allowed ranges should be cut.

        Returns:
        --------
        samples : array-like
            A sub-set of the provided data points which falls within the allowed ranges.

        Attributes:
        -----------
        None
        """
        # Extract the provided minimum and maximum allowed values for all of the parameters
        low, high = self.parameter_ranges[:, 0], self.parameter_ranges[:, 1]
        # Retain only the data points that fall within the allowed range for each parameter
        samples = samples[np.all(np.logical_and(samples > low, samples < high), axis = 1), :]
        return samples


    def weight_truncation(self, importance_probability):
        """Transform the handed probabilities to combat dominating data points

        This function transforms provided probabilities via an implementation of truncated
        importance sampling and re-normalizing the results. The transformation leads to a
        downgrading of otherwise dominating high-probability samples for the re-sampling.

        Parameters:
        -----------
        importance_probability : array-like
            The one-dimensional array of importance probabilities that should be smoothed.


        Returns:
        --------
        normalized_result : array-like
            A transformed array of probabilities with slightly less pronounced inequalities.

        Attributes:
        -----------
        None
        """
        # Compute the mean of the importance probabilities provided as input
        mean_probability = np.mean(importance_probability)
        comparison_term = self.mixture_samples**(1./self.truncation_alpha) * mean_probability
        # Calculate the element-wise minimum comparison between the given terms
        truncated_probability = np.minimum(importance_probability, comparison_term)
        # Normalize the results to ensure their full usability as probabilities
        normalized_result = truncated_probability / truncated_probability.sum()
        return normalized_result


    def importance_sampling(self, samples, model, return_weights):
        """Re-sample provided data points according to their computed importance weights.

        Importance sampling is a general estimation approach for distributions if only samples
        from another distribution are available. In the given case of this code, that other
        distribution is the one approximated by a fitted model per iteration. By handing a
        set of data points, a model and the user-specified evaluation function for posterior
        probabilities to this function, it re-samples these data points with frequency values
        derived from their respective importance weights for a better fit of the samples.

        Parameters:
        -----------
        return_weights : boolean
            The boolean value indicating whether to simply return the importance weights.

        Returns:
        --------
        samples : array-like
            A set of data points re-sampled due to frequencies based on importance weights.

        Attributes:
        -----------
        None
        """
        # Apply the procided evaluation function to get the posteriors of all data points
        posteriors = list(self.pool.map(self.posterior_evaluation, samples))
        # Use the model's built-in scoring function to get the posteriors w.r.t. the model
        proposal = model.score_samples(X = samples)
        # Get the importance weights for all the data points via element-wise subtraction
        print("JAZ Need to do truncation here too")
        importance_weights = posteriors - proposal

        # Make the max weight always unity, for numerical convenience
        importance_weights -= importance_weights.max()

        # Check whether the function is called to calculate and return the importance weights
        if return_weights:
            return importance_weights
        else:
            # Calculate the importance probabilities for the purpose of subsequent resampling
            importance_probability = np.exp(importance_weights)
            # Smoothe the importance probabilities with truncation to penalize dominant samples
            smoothed_probability = self.weight_truncation(importance_probability)
            # Calculate the importance probabilities for the purpose of subsequent resampling
            importance_probability = np.divide(smoothed_probability, sum(smoothed_probability))
            # Create a vector of index frequencies as weighted by the calculated probabilities
            sampling_index = np.random.choice(a = samples.shape[0],
                                              size = self.mixture_samples,
                                              replace = True,
                                              p = importance_probability)
            # Build a set of data points in accordance with the created vector of frequencies
            importance_samples = samples[sampling_index]
            return importance_samples


    def mixture_fit(self, samples, tolerance):
        """Fit a variational Bayesian non-parametric Gaussian mixture model to samples.

        This function takes the parameters described below to initialize and then fit a
        model to a provided set of data points. It returns a Scikit-learn estimator object
        that can then be used to generate samples from the distribution approximated by the
        model and score the log-probabilities of data points based on the returned model.


        Parameters:
        -----------
        samples : array-like
            The set of provided data points that the function's model should be fitted to.


        Returns:
        --------
        model : sklearn estimator
            A variational Bayesian non-parametric Gaussian mixture model fitted to samples.

        Attributes:
        -----------

        fit(X) : Estimate a model's parameters with the expectation maximization algorithm.

        sample(n_samples=1) : Generate a new set of random data points from fitted Gaussians.

        score_samples(X) : Calculate the weighted log-probabilities for each data point.
        """
        # Check which type of model should be used for the iterative fitting process
        if self.model_selection == 'gmm':
            # Initialize a variational Bayesian non-parametric GMM for fitting
            model = BGM(n_components = self.model_components,
                        covariance_type = self.model_covariance,
                        tol = tolerance,
                        max_iter = self.em_iterations,
                        init_params = self.parameter_init,
                        verbose = self.model_verbosity,
                        verbose_interval = 10,
                        warm_start = False,
                        random_state = 42,
                        weight_concentration_prior_type = 'dirichlet_process')
        elif self.model_selection == 'kde':
            model = KD(bandwidth = self.kde_bandwidth,
                       kernel = 'gaussian',
                       metric = 'euclidean',
                       algorithm = 'auto',
                       breadth_first = True,
                       atol = 0.0,
                       rtol = tolerance)
        # Fit the previously initialized model to the provided data points
        model.fit(np.asarray(samples))
        return model


    def dynamical_tolerance(self, step):
        """Calculate the model's convergence threshold dynamically for the given iteration.

        The variational Bayesian non-parametric Gaussian mixture model used in this code
        requires a convergence threshold to terminate before reaching the maximum number
        of model-fitting iterations. As the initial sample isn't a great fit for the true
        posterior distribution, optimizing the fit to a very strict convergence threshold
        isn't very sensible. Instead, the convergence threshold decreases linearly with
        each full iteration, with the change from iteration to iteration being dependent
        on the range that is provided by the user to allow customization to a problem.

        Parameters:
        -----------
        step : int
            the index of the current iteration, i.e. a counter to keep track of progress.

        Returns:
        --------
        step_tolerance : float
            A desired convergence threshold for the given step's model-fitting process.

        Attributes:
        -----------
        None
        """
        # Difference between the high and low end of the tolerance range
        difference = self.tolerance_range[0] - self.tolerance_range[1]
        # Increase by which the tolerance level is tightened at each step
        increase = difference / (self.gaussbock_iterations - 1)
        # Tolerance for the current iteration the function was called for
        step_tolerance = self.tolerance_range[0] - (step * increase)
        return step_tolerance



def pool_type(mpi_parallelization,
              processes,
              pool=None):
    """
    Establish which kind of pool should be used by the code for parallelization purposes.

    When using larger computing clusters or supercomputing facilities, MPI is a sensible
    choice and should used via an MPIPool. For local computing clusters, the recommended
    choice is a MultiPool, whereas a SerialPool is a simple single-process implementation.

    The boolean parameter 'mpi', which is set as false by default, has to be set as true
    to use MPI. If more than one process is called for by the parameter 'processes', the
    code will try to fulfill this wish. If the latter fails or one process is explicitly
    indicated, the default of one process will result in the usage of a SerialPool.


    Parameters:
    -----------
    mpi_parallelization : boolean
        The boolean value indicating whether to parallelize the code via an MPIPool.

    processes : int
        The number of processes the code should invoke for its parallelizable parts.

    pool : pool-like object, optional
        If set, use this object as the pool instead of creating a new one

    Returns:
    --------
    pool : {'mpi', 'multi', 'serial'}
        A string denoting which kind of pool should be used for parallelizing code.

    Attributes:
    -----------
    None
    """

    if pool is not None:
        return pool

    # Check whether the utilization of an MPIPool is indicated via the input
    if mpi_parallelization:
        # If the use of an MPIPool is not enabled, print an error and terminate
        if not MPIPool.enabled():
            raise ValueError("ERROR: MPIPool must be enabled in order to parallelize via MPI")
        # If the use of an MPIPool is enabled, resume and wait if necessary
        else:
            pool = MPIPool()
            pool_indicator = 0
            print_variable = pool.size
    # Check whether the input allows for the utilization of a MultiPool instead
    elif processes > 1 and MultiPool.enabled():
        pool = MultiPool(processes = processes)
        pool_indicator = 1
        print_variable = processes
    # If one process or no process number is given in the input, use a SerialPool
    else:
        pool = SerialPool()
        pool_indicator = 2
        print_variable = 1
    # Create a list of possible notifications about the pool type that is used
    messages = ['NOTE: Running with MPI on %d cores',
                'NOTE: Running with MultiPool on %d cores',
                'NOTE: Running with SerialPool on %d core']
    # When using MPI, print the corresponding statement only once for pool rank 0
    if pool_indicator == 0:
        if pool.comm.Get_rank() == 0:
            print(messages[pool_indicator] % print_variable)
    else:
        print(messages[pool_indicator] % print_variable)
    print()
    return pool



def test():
    ranges = np.array([[0.0, 1.], [0.0, 1.0]])
    def posterior_evaluation(p):
        L = -0.5*((p[0]-0.5)**2 + (p[1]-0.5)**2)/0.05**2
        return L
    output_samples=5000
    G = Gaussbock(ranges, posterior_evaluation, output_samples)
    start = ['automatic', 16, 100]
    p, w, model = G.run(start)
    w = np.exp(w-w.max())
    # print(np.cov(p.T, aweights=w))
    # import pylab
    # print(p.shape)
    # pylab.hist2d(p[:,0], p[:,1], weights=w, bins=30)
    # pylab.show()
    # pylab.hist2d(p[:,0], p[:,1],            bins=30)
    # pylab.show()


if __name__ == '__main__':
    test()