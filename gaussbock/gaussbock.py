"""Fast parallel-iterative cosmological parameter estimation with Bayesian nonparametrics

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
import sys
import numpy as np
import emcee as ec
from scipy.stats import halfnorm
from schwimmbad import SerialPool, MultiPool, MPIPool
from sklearn.mixture import BayesianGaussianMixture as BGM
from sklearn.neighbors import KernelDensity as KD
from sklearn.preprocessing import minmax_scale as scaling

def gaussbock(parameter_ranges,
              posterior_evaluation,
              output_samples,
              initial_samples = None,
              gaussbock_iterations = None,
              convergence_threshold = None,
              mixture_samples = 10000,
              em_iterations = 1000,
              tolerance_range = [1e-2, 1e-7],
              model_components = None,
              model_covariance = 'full',
              parameter_init = 'random',
              model_verbosity = 1,
              mpi_parallelization = False,
              processes = 1,
              weights_and_model = False,
              truncation_alpha = 2.0,
              model_selection = None,
              kde_bandwidth = 0.5):
    """Sample data points from a parameter space's approximated probability distribution.

    This is the primary function of 'gaussbock', allowing access to its functionality
    with one simple function call. Most parameters default to reasonably well-behaved
    values and don't need to be customized to get good results. The only parameters that
    the user has to specify is the allowed range for each variable (i.e. dimension), an
    evaluation function that takes one data point and returns its posterior probability,
    and the desired number of samples from the approximated posterior distribution.

    Parameters:
    -----------
    parameter_ranges : array-like
        The allowed range for each parameter, signified by a lower and upper limit, with
        one row per parameter, and lower and upper limits in the first and second column.

    posterior_evaluation : function
        The evaluation function that takes a single data point in the parameter space and
        then returns the logarithmic posterior value for the specific given data point.

    output_samples : int
        The number of samples generated from the final model's posterior approximation.

    initial_samples : {['automatic', int, int], ['custom', array-like]},
                      defaults to ['automatic', 50, 1000]
        The specification whether 'emcee' should be used to generate the initial set of
        data points, with the number of walkers and the number of chain steps specified,
        or whether a custom pre-prepared set of data points should be used instead.

    gaussbock_iterations : int, defaults to None
        The number of iterations that gaussbock should circle through to achieve a fit.
        
    convergence_threshold : float, defaults to None
        The threshold to determine inter-iteration importance resampling convergence.

    mixture_samples : int, defaults to 10000
        The number of samples to be drawn from the model before each importance sampling.

    em_iterations : int, defaults to 1000
        The maximum number of expectation maximization iterations the model should run.

    tolerance_range : list, defaults to [1e-2, 1e-7]
        The two ends for the shrinking convergence threshold as a tuple [float, float].

    model_components : int, defaults to rounding up (2 / 3) * the number of dimenzsions
        The maximum number of Gaussians to be fitted to data points in each iteration.

    model_covariance : {'full', 'tied', 'diag', 'spherical'}, defaults to 'full'
        The type of covariance parameters the model should use for the fitting process.

    parameter_init :  {'kmeans', 'random'}, defaults to 'random'
        The method used to initialize the model's weights, the means and the covariances.

    model_verbosity : {0, 1, 2}, defaults to 1
        The amount of information that the model fitting should provide during runtime.

    mpi_parallelization : boolean, defaults to False
        The boolean value indicating whether to parallelize the code via an MPIPool.

    processes : int, defaults to 1
        The number of processes the code should invoke for its parallelizable parts.

    weights_and_model : boolean, defaults to False
        The boolean value indicating whether to return importance weights and the model.

    truncation_alpha : float from [1.0, 3.0], defaults to 2.0
        The truncation value for importance probability re-weighting. A higher value
        leads to a more general fitting with strong truncation, whereas a smaller value
        leads to a higher level of retaining dominant data points as such. It is generally
        recommended to only alter this parameter if the resulting posterior approximation
        is problematic and the issue can't be resolved by adjusting other parameters.

    model_selection : {None, 'gmm', 'kde'}, defaults to None
        The selection of the type of model that should be used for the fitting process,
        i.e. either a variational Bayesian non-parametric GMM or kernel density estimation.

    kde_bandwidth : float, defaults to 0.5
        The kernel bandwidth that should be used in the case of kernel density estimation.

    Returns:
    --------
    samples : array-like or list
        A set of samples of the size that is specified in the parameter output_samples if
        'weights_and_model' is false, which is the default. If 'weights_and_model' is true,
        a list with the above set of samples, the importance weights of the last model and
        the model itself is returned. The list elements always follow this order:

        0 : Samples generated with the final model
        1 : Importance weights for the final model
        2 : The final model itself to sample from

    Attributes:
    -----------
    None
    """
    # Set the random seed
    #np.random.seed(0)
    # If no initial sampling parameters are specified, assign defaults
    if initial_samples == None:
        walker_number = int(np.multiply(len(parameter_ranges), 2) + 2)
        initial_samples = ['automatic', walker_number, 1000]
    # Test all parameters for validity and terminate if not true
    parameter_check(parameter_ranges = parameter_ranges,
                    posterior_evaluation = posterior_evaluation,
                    output_samples = output_samples,
                    initial_samples = initial_samples,
                    gaussbock_iterations = gaussbock_iterations,
                    convergence_threshold = convergence_threshold,
                    mixture_samples = mixture_samples,
                    em_iterations = em_iterations,
                    tolerance_range = tolerance_range,
                    model_components = model_components,
                    model_covariance = model_covariance,
                    parameter_init = parameter_init,
                    model_verbosity = model_verbosity,
                    mpi_parallelization = mpi_parallelization,
                    processes = processes,
                    weights_and_model = weights_and_model,
                    truncation_alpha = truncation_alpha,
                    model_selection = model_selection,
                    kde_bandwidth = kde_bandwidth)
    # Check whether a number of iterations is given, otherwise fall back on defaults
    if gaussbock_iterations == None:
        if convergence_threshold == None:
            gaussbock_iterations = 10
            convergence_check = False
        else:
            gaussbock_iterations = 50
            convergence_check = True
    else:
        if convergence_threshold == None:
            convergence_check = False
        else:
            convergence_check = True
    # Get the number of dimensions that the algorithm operates in for the provided problem
    parameter_number = parameter_ranges.shape[0]
    # If not chosen, set the type of model for fitting depending on the number of dimensions
    if model_selection == None:
        if parameter_number <= 2:
            model_selection = 'kde'
        else:
            model_selection = 'gmm'
    # Assign a reasonable default value to be used as the maximal number of models to fit
    if model_components == None:
        model_components = int(np.ceil((2 / 3) * parameter_number))
        #model_components = 50
    # Establish the type of pool to be used for an eventual parallelization of the sampler
    with pool_type(mpi_parallelization = mpi_parallelization, processes = processes) as pool:
        # In case of using MPI, let worker processes wait for tasks from the master process
        if mpi_parallelization == True:
            if not pool.is_master():
                pool.wait()
                return
        # Check whether to get initial samples via emcee or use a set
        if initial_samples[0] == 'automatic':

            emcee_walkers, emcee_steps = initial_samples[1:3]
            # Call the emcee-based function to get a set of MCMC samples
            print('PROCESS: Getting initial samples via an affine-invariant MCMC ensemble ...')
            print('--------------------------------------------------------------------------\n')
            samples = gaussbock_emcee(parameter_ranges = parameter_ranges,
                                      emcee_walkers = emcee_walkers,
                                      emcee_steps = emcee_steps,
                                      posterior_evaluation = posterior_evaluation,
                                      mpi_parallelization = mpi_parallelization,
                                      processes = processes,
                                      pool = pool)
            print('===> DONE\n')
        elif initial_samples[0] == 'custom':
            print('NOTE: Using the user-provided set of initial samples\n')
            #sample_number = np.multiply(len(parameter_ranges), 2) + 2
            #random_drawing = np.random.randint(len(initial_samples[1]), size = sample_number)
            #samples = initial_samples[1][random_drawing, :]
            samples = initial_samples[1]
        print('NOTE: Starting gaussbock iterations\n')
        # Loop over the provided number of iterations minus the last one
        step_indicator = 0
        for i in range(gaussbock_iterations - 1):
            print('ITERATION %d\n' % i)
            # Store the previous inter-iteration variance and difference
            if convergence_check == True:
                if i > 0:
                    previous_weights = weights
                #if i > 1:
                #    previous_inter_iteration = inter_iteration
            print('PROCESS: Fitting the model for iteration %d ...' % i)
            print('----------------------------------------------\n')
            # Get the model-fitting tolerance level for the iteration
            step_tolerance = dynamical_tolerance(tolerance_range = tolerance_range,
                                                 gaussbock_iterations = gaussbock_iterations,
                                                 step = i)
            # Fit the model for the current data points and tolerance
            mixture_model = mixture_fit(samples = samples,
                                        model_components = model_components,
                                        model_covariance = model_covariance,
                                        tolerance = step_tolerance,
                                        em_iterations = em_iterations,
                                        parameter_init = parameter_init,
                                        model_verbosity = model_verbosity,
                                        model_selection = model_selection,
                                        kde_bandwidth = kde_bandwidth)
            print('===> DONE\n')
            # Sample a novel set of data points from the fitted model
            if model_selection == 'gmm':
                samples = mixture_model.sample(n_samples = mixture_samples * 1.5)[0]
            if model_selection == 'kde':
                samples = mixture_model.sample(n_samples = int(np.round(output_samples * 1.5)))
            # Cut all of the data points outside of the allowed ranges
            samples = range_cutoff(samples = samples,
                                   parameter_ranges = parameter_ranges)
            # Extract the required amount of samples from the now clean set
            samples = samples[0:output_samples, :]
            # Generate a new set of weighted and resampled data points
            print('PROCESS: Importance sampling for re-weighted frequencies ...')
            print('------------------------------------------------------------\n')
            samples, weights = importance_sampling(samples = samples,
                                                   model = mixture_model,
                                                   mixture_samples = mixture_samples,
                                                   posterior_evaluation = posterior_evaluation,
                                                   mpi_parallelization = mpi_parallelization,
                                                   processes = processes,
                                                   return_weights = False,
                                                   pool = pool,
                                                   parameter_number = parameter_number,
                                                   truncation_alpha = truncation_alpha)
            print('\n===> DONE\n')
            # Calculate the inter-iteration variance difference
            if convergence_check == True:
                if i > 0:
                    #scaled_values = np.var(scaling(weights)) - np.var(scaling(previous_weights))
                    #inter_iteration = np.absolute(scaled_values)
                    var_difference = np.absolute(np.var(weights) - np.var(previous_weights))
                    print('\nITERATION DIFFERENCE: %f\n' % var_difference)
                # If the convergence criterion is fulfilled, terminate the loop
                #if i > 1:
                #    iteration_difference = np.absolute(inter_iteration - previous_inter_iteration)
                #    print('\nITERATION DIFFERENCE: %f\n' % iteration_difference)
                #    if iteration_difference < convergence_threshold:
                    if var_difference < convergence_threshold:
                        break
            step_indicator = step_indicator + 1
        # Fit the final model to the data points provided by the loop
        print('PROCESS: Fitting the final model ...')
        print('------------------------------------\n')
        # Get the model-fitting tolerance level for the iteration
        step_tolerance = dynamical_tolerance(tolerance_range = tolerance_range,
                                             gaussbock_iterations = gaussbock_iterations,
                                             step = step_indicator)
        mixture_model = mixture_fit(samples = samples,
                                    model_components = model_components,
                                    model_covariance = model_covariance,
                                    #tolerance = tolerance_range[1],
                                    tolerance = step_tolerance,
                                    em_iterations = em_iterations,
                                    parameter_init = parameter_init,
                                    model_verbosity = model_verbosity,
                                    model_selection = model_selection,
                                    kde_bandwidth = kde_bandwidth)
        print('\n===> DONE\n')
        # Sample 150% of the required amount of samples from the model
        if model_selection == 'gmm':
            samples = mixture_model.sample(n_samples = output_samples * 1.5)[0]
        if model_selection == 'kde':
            samples = mixture_model.sample(n_samples = int(np.round(output_samples * 1.5)))
        # Cut all the data points falling outside of the allowed ranges
        samples = range_cutoff(samples = samples,
                               parameter_ranges = parameter_ranges)
        # Extract the required amount of samples from the now clean set
        samples = samples[0:output_samples, :]
        print('PROCESS: Checking and preparing returns ...')
        print('-------------------------------------------\n')
        # Check whether to return the importance weights and the model
        if weights_and_model == False:
            print('NOTE: Successful termination; samples are being returned')
            return samples
        if weights_and_model == True:
            # Get the importance weights for the final set of data points
            importance_weights = importance_sampling(samples = samples,
                                                     model = mixture_model,
                                                     mixture_samples = mixture_samples,
                                                     posterior_evaluation = posterior_evaluation,
                                                     mpi_parallelization = mpi_parallelization,
                                                     processes = processes,
                                                     return_weights = True,
                                                     pool = pool,
                                                     parameter_number = parameter_number,
                                                     truncation_alpha = truncation_alpha)
        # Stop the pool at this point to avoid simply letting the process hang indefinitely
        pool.close()
        print('NOTE: Successful termination; samples, weights and model are being returned')
        return [samples, importance_weights, mixture_model]

def gaussbock_emcee(parameter_ranges,
                    emcee_walkers,
                    emcee_steps,
                    posterior_evaluation,
                    mpi_parallelization,
                    processes,
                    pool):
    """
    Create an initial set of data points with an affine-invariant MCMC ensemble sampler.

    While the iteartive method utilized in this code also works for an initial sample that
    is drawn uniformly at random from high-dimensional parameter ranges, convergence of the
    approximations towards a good fit to the true posterior distribution takes some time.
    For this reason, an affine-invariant MCMC ensemble sampler is used to generate the first
    set of data points that represent a very rough approximation to the true posterior.

    Parameters:
    -----------
    parameter_ranges : array-like
        The allowed range for each parameter, signified by a lower and upper limit, with
        one row per parameter, and lower and upper limits in the first and second column.

    emcee_walkers : int
        The number of separate walkers that emcee should deploy to gather first samples.

    emcee_steps : int
        The number of steps in a chain that emcee walkers should take before termination.

    posterior_evaluation : function
        The evaluation function that takes a single data point in the parameter space and
        then returns the logarithmic posterior value for the specific given data point.

    mpi_parallelization : boolean
        The boolean value indicating whether to parallelize the code via an MPIPool.

    processes : int
        The number of processes the code should invoke for its parallelizable parts.

    pool : schwimmbad pool
        The pool created via 'schwimmbad' and returned by the pool_type() function.

    Returns:
    --------
    samples : array-like
        An initial set of distribution-approximating samples as chains of MCMC walkers.

    Attributes:
    -----------
    None
    """
    # Extract the provided minimum and maximum allowed values for all of the parameters
    low, high = parameter_ranges[:, 0], parameter_ranges[:, 1]
    # Generate starting points for emcee by drawing uniformly-distributed random samples
    starting_points = [np.random.uniform(low = low, high = high) for i in range(emcee_walkers)]
    # Get the number of parameters, i.e. the number of dimensions, as an input for emcee
    parameter_number = parameter_ranges.shape[0]
    # Create a sampler with the number of walkers, dimensions and evaluation function
    emcee_sampler = ec.EnsembleSampler(nwalkers = emcee_walkers,
                                       dim = parameter_number,
                                       lnpostfn = posterior_evaluation,
                                       pool = pool)
    # Run the sampler with the generated starting points and a step number per walker
    emcee_sampler.run_mcmc(pos0 = starting_points,
                           N = emcee_steps)
    # Access the sampler's chain flattened to the number of chain links and the dimension
    emcee_samples = emcee_sampler.flatchain
    return emcee_samples

def pool_type(mpi_parallelization,
              processes):
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

    Returns:
    --------
    pool : {'mpi', 'multi', 'serial'}
        A string denoting which kind of pool should be used for parallelizing code.

    Attributes:
    -----------
    None
    """
    # Check whether the utilization of an MPIPool is indicated via the input
    if mpi_parallelization == True:
        # If the use of an MPIPool is not enabled, print an error and terminate
        if not MPIPool.enabled():
            print('ERROR: MPIPool must be enabled in order to parallelize via MPI')
            sys.exit()
        # If the use of an MPIPool is enabled, resume and wait if necessary
        else:
            pool = MPIPool()
            pool_indicator = 0
            print_variable = pool.size
    # Check whether the input allows for the utilization of a MultiPool instead
    elif processes > 1 and MultiPool.enabled() == True:
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

def range_cutoff(samples,
                 parameter_ranges):
    """Cut data points which fall outside of allowed parameter ranges from a data set.

    As some user-provided evaluation functions for calculating posterior probabilites of
    single data points might prohibit values outside of certain ranges, e.g. by resulting
    in minus infinity as the returned value, this function cuts all data points from a
    set of data points with such a parameter range violation for one or more variables.

    Parameters:
    -----------
    samples : array-like
        The set data points from which the ones outside the allowed ranges should be cut.

    parameter_ranges : array-like
        The allowed range for each parameter, signified by a lower and upper limit, with
        one row per parameter, and lower and upper limits in the first and second column.

    Returns:
    --------
    samples : array-like
        A sub-set of the provided data points which falls within the allowed ranges.

    Attributes:
    -----------
    None
    """
    # Extract the provided minimum and maximum allowed values for all of the parameters
    low, high = parameter_ranges[:, 0], parameter_ranges[:, 1]
    # Retain only the data points that fall within the allowed range for each parameter
    samples = samples[np.all(np.logical_and(samples > low, samples < high), axis = 1), :]
    return samples

def importance_sampling(samples,
                        model,
                        mixture_samples,
                        posterior_evaluation,
                        mpi_parallelization,
                        processes,
                        return_weights,
                        pool,
                        parameter_number,
                        truncation_alpha):
    """Re-sample provided data points according to their computed importance weights.

    Importance sampling is a general estimation approach for distributions if only samples
    from another distribution are available. In the given case of this code, that other
    distribution is the one approximated by a fitted model per iteration. By handing a
    set of data points, a model and the user-specified evaluation function for posterior
    probabilities to this function, it re-samples these data points with frequency values
    derived from their respective importance weights for a better fit of the samples.

    Parameters:
    -----------
    samples : array-like
        The set of provided data points that importance sampling should be run on.

    mixture_samples : int
        The number of samples to be drawn from the model before each importance sampling.

    posterior_evaluation : function
        The evaluation function that takes a single data point in the parameter space and
        then returns the logarithmic posterior value for the specific given data point.

    mpi_parallelization : boolean
        The boolean value indicating whether to parallelize the code via an MPIPool.

    processes : int
        The number of processes the code should invoke for its parallelizable parts.

    return_weights : boolean
        The boolean value indicating whether to simply return the importance weights.

    pool : schwimmbad pool
        The pool created via 'schwimmbad' and returned by the pool_type() function.

    parameter_number : int
        The number of dimensions that the algorithm operates in for the given problem.

    truncation_alpha : float
        The truncation value for importance probability re-weighting. This parameter must
        be a float from the interval[1.0, 3.0]. A higher value leads to a looser fitting
        with strong truncation, whereas a smaller value leads to a higher level of retaining
        dominant data points as such. It is recommended to only alter this parameter if the
        resulting posterior approximation is problematic and can't be otherwise resolved.

    Returns:
    --------
    samples : array-like
        A set of data points re-sampled due to frequencies based on importance weights.

    Attributes:
    -----------
    None
    """
    # Apply the procided evaluation function to get the posteriors of all data points
    posteriors = list(pool.map(posterior_evaluation, samples))
    # Use the model's built-in scoring function to get the posteriors w.r.t. the model
    proposal = model.score_samples(X = samples)
    # Get the importance weights for all the data points via element-wise subtraction
    importance_weights = posteriors - proposal
    # Calculate the importance probabilities for the purpose of subsequent resampling
    importance_probability = np.exp(importance_weights)
    # Smooth the importance probabilities with truncation to penalize dominant samples
    smoothed_probability = weight_truncation(mixture_samples = mixture_samples,
                                             importance_probability = importance_probability,
                                             truncation_alpha = truncation_alpha)
    # Calculate the importance probabilities for the purpose of subsequent resampling
    importance_probability = np.divide(smoothed_probability, sum(smoothed_probability))
    # Check whether the function is called to calculate and return the importance weights
    if return_weights == True:
        importance_weights = np.log(importance_probability)
        return importance_weights
    if return_weights == False:
        # Create a vector of index frequencies as weighted by the calculated probabilities
        sampling_index = np.random.choice(a = samples.shape[0],
                                          size = mixture_samples,
                                          replace = True,
                                          p = importance_probability)
        # Build a set of data points in accordance with the created vector of frequencies
        importance_samples = samples[sampling_index]
        return importance_samples, importance_weights

def weight_truncation(mixture_samples,
                      importance_probability,
                      truncation_alpha):
    """Transform the handed probabilities to combat dominating data points

    This function transforms provided probabilities via an implementation of truncated
    importance sampling and re-normalizing the results. The transformation leads to a
    downgrading of otherwise dominating high-probability samples for the re-sampling.

    Parameters:
    -----------
    importance_probability : array-like
        The one-dimensional array of importance probabilities that should be smoothed.

    mixture_samples : int
        The number of samples to be drawn from the model before each importance sampling.

    truncation_alpha : float
        The truncation value for importance probability re-weighting. This parameter must
        be a float from the interval[1.0, 3.0]. A higher value leads to a looser fitting
        with strong truncation, whereas a smaller value leads to a higher level of retaining
        dominant data points as such. It is recommended to only alter this parameter if the
        resulting posterior approximation is problematic and can't be otherwise resolved.

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
    # Create the necessary comparison term for truncated importance sampling
    if truncation_alpha == 2.0:
        comparison_term = np.multiply(np.sqrt(mixture_samples), mean_probability)
    elif truncation_alpha == 3.0:
        comparison_term = np.multiply(np.cbrt(mixture_samples), mean_probability)
    else:
        comparison_term = np.multiply(np.power(mixture_samples, (1 / truncation_alpha)),
                                      mean_probability)
    # Calculate the element-wise minimum comparison between the given terms
    truncated_probability = np.minimum(importance_probability, comparison_term)
    # Normalize the results to ensure their full usability as probabilities
    normalized_result = np.divide(truncated_probability, np.sum(truncated_probability))
    return normalized_result

def mixture_fit(samples,
                model_components,
                model_covariance,
                tolerance,
                em_iterations,
                parameter_init,
                model_verbosity,
                model_selection,
                kde_bandwidth):
    """Fit a variational Bayesian non-parametric Gaussian mixture model to samples.

    This function takes the parameters described below to initialize and then fit a
    model to a provided set of data points. It returns a Scikit-learn estimator object
    that can then be used to generate samples from the distribution approximated by the
    model and score the log-probabilities of data points based on the returned model.


    Parameters:
    -----------
    samples : array-like
        The set of provided data points that the function's model should be fitted to.

    model_components : int, defaults to rounding up (2 / 3) * the number of dimensions
        The maximum number of Gaussians to be fitted to data points in each iteration.

    model_covariance : {'full', 'tied', 'diag', 'spherical'}
        The type of covariance parameters the model should use for the fitting process.

    tolerance : float
        The model's convergence threshold at which the model's fit is deemed finalized.

    em_iterations : int
        The maximum number of expectation maximization iterations the model should run.

    parameter_init :  {'kmeans', 'random'}
        The method used to initialize the model's weights, the means and the covariances.

    model_verbosity : {0, 1, 2}
        The amount of information that the model fitting should provide during runtime.

    model_selection : {'gmm', 'kde'}
        The selection of the type of model that should be used for the fitting process,
        i.e. either a variational Bayesian non-parametric GMM or kernel density estimation.

    kde_bandwidth : float
        The kernel bandwidth that should be used in the case of kernel density estimation.

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
    if model_selection == 'gmm':
        # Initialize a variational Bayesian non-parametric GMM for fitting
        model = BGM(n_components = model_components,
                    covariance_type = model_covariance,
                    tol = tolerance,
                    max_iter = em_iterations,
                    init_params = parameter_init,
                    verbose = model_verbosity,
                    verbose_interval = 10,
                    warm_start = False,
                    random_state = 42,
                    weight_concentration_prior_type = 'dirichlet_process')
    if model_selection == 'kde':
        model = KD(bandwidth = kde_bandwidth,
                   kernel = 'gaussian',
                   metric = 'euclidean',
                   algorithm = 'auto',
                   breadth_first = True,
                   atol = 0.0,
                   rtol = tolerance)
    # Fit the previously initialized model to the provided data points
    model.fit(np.asarray(samples))
    return model

def dynamical_tolerance(tolerance_range,
                        gaussbock_iterations,
                        step):
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
    tolerance_range : list
        The two ends for the shrinking convergence threshold as a tuple [float, float].

    gaussbock_iterations : int
        The number of iterations that gaussbock should circle through to achieve a fit.

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
    difference = tolerance_range[0] - tolerance_range[1]
    # Increase by which the tolerance level is tightened at each step
    increase = difference / (gaussbock_iterations - 1)
    # Tolerance for the current iteration the function was called for
    step_tolerance = tolerance_range[0] - (step * increase)
    return step_tolerance

def parameter_check(parameter_ranges,
                    posterior_evaluation,
                    output_samples,
                    initial_samples,
                    gaussbock_iterations,
                    convergence_threshold,
                    mixture_samples,
                    em_iterations,
                    tolerance_range,
                    model_components,
                    model_covariance,
                    parameter_init,
                    model_verbosity,
                    mpi_parallelization,
                    processes,
                    weights_and_model,
                    truncation_alpha,
                    model_selection,
                    kde_bandwidth):
    """Check all parameters for conformance with the required parameter specifications.

    Sometimes, users might provide parameters in formats or with values that aren't
    expected and would, therefore, lead to a crash. In order to prevent waiting for the
    code to reach such a breaking point, all parameters are checked in the beginning.
    The code will terminate with a descriptive error message in case of discrepancies.

    Parameters:
    -----------
    parameter_ranges : array-like
        The allowed range for each parameter, signified by a lower and upper limit, with
        one row per parameter, and lower and upper limits in the first and second column.

    posterior_evaluation : function
        The evaluation function that takes a single data point in the parameter space and
        then returns the logarithmic posterior value for the specific given data point.

    output_samples : int
        The number of samples generated from the final model's posterior approximation.

    initial_samples : {['automatic', int, int], ['custom', array-like]}
        The specification whether 'emcee' should be used to generate the initial set of
        data points, with the number of walkers and the number of chain steps specified,
        or whether a custom pre-prepared set of data points should be used instead.

    gaussbock_iterations : int
        The number of iterations that gaussbock should circle through to achieve a fit.

    mixture_samples : int
        The number of samples to be drawn from the model before each importance sampling.

    em_iterations : int
        The maximum number of expectation maximization iterations the model should run.

    tolerance_range : list
        The two ends for the shrinking convergence threshold as a tuple [float, float].

    model_components : int
        The maximum number of Gaussians to be fitted to data points in each iteration.

    model_covariance : {'full', 'tied', 'diag', 'spherical'}
        The type of covariance parameters the model should use for the fitting process.

    parameter_init :  {'kmeans', 'random'}
        The method used to initialize the model's weights, the means and the covariances.

    model_verbosity : {0, 1, 2}
        The amount of information that the model fitting should provide during runtime.

    mpi_parallelization : boolean
        The boolean value indicating whether to parallelize the code via an MPIPool.

    processes : int
        The number of processes the code should invoke for its parallelizable parts.

    truncation_alpha : float
        The truncation value for importance probability re-weighting. This parameter must
        be a float from the interval[1.0, 3.0]. A higher value leads to a looser fitting
        with strong truncation, whereas a smaller value leads to a higher level of retaining
        dominant data points as such. It is recommended to only alter this parameter if the
        resulting posterior approximation is problematic and can't be otherwise resolved.

    model_selection : {'gmm', 'kde'}
        The selection of the type of model that should be used for the fitting process,
        i.e. either a variational Bayesian non-parametric GMM or kernel density estimation.

    kde_bandwidth : float
        The kernel bandwidth that should be used in the case of kernel density estimation.

    Returns:
    --------
    None

    Attributes:
    -----------
    None
    """
    # Create a vector of boolean values to keep track of incorrect inputs
    incorrect_inputs = np.zeros(20, dtype = bool)
    # Check if the parameter 'parameter_ranges' satisfies the specifications
    if not (type(parameter_ranges) == np.ndarray and parameter_ranges.shape[1] == 2
            and np.all([parameter_ranges[i, 0] < parameter_ranges[i, 1]
            for i in range(0, parameter_ranges.shape[1])])):
        incorrect_inputs[0] = True
    # Check if the parameter 'posterior_evaluation' is a suitable function
    try:
        # Extract the provided minimum and maximum allowed values for all of the parameters
        low, high = parameter_ranges[:, 0], parameter_ranges[:, 1]
        # Draw a random test data point for the function from the allowed parameter ranges
        test_point = np.random.uniform(low = low, high = high)
        try:
            test_posterior = posterior_evaluation(test_point)
        except:
            incorrect_inputs[1] = True
    except:
        # Let the function resume, as a faulty 'parameter_ranges' will terminate
        pass
    # Check if the provided output parameter 'output_samples' is an integer
    if not (type(output_samples) == int and output_samples > 0):
        incorrect_inputs[2] = True
    # Check if the parameter 'input_samples' is withing the allowed set
    if not initial_samples[0] in ['automatic', 'custom']:
        incorrect_inputs[3] = True
    if initial_samples[0] == 'automatic' and (len(initial_samples) != 3
        or [type(initial_samples[i]) for i in range(0,3)] != [str, int, int]):
        incorrect_inputs[3] = True
    elif initial_samples[0] == 'custom' and (len(initial_samples) != 2
        or [type(initial_samples[i]) for i in range(0,2)] != [str, np.ndarray]):
        incorrect_inputs[3] = True
    # Check if the provided parameter 'gaussbock_iterations' is an integer or None
    if not gaussbock_iterations is None:
        if not (type(gaussbock_iterations) == int and gaussbock_iterations > 0):
            incorrect_inputs[4] = True
    # Check if the parameter 'mixture_samples' for the model is an integer
    if not (type(mixture_samples) == int and mixture_samples > 0):
        incorrect_inputs[5] = True
    # Check if the parameter 'em_iterations' for the model is an integer
    if not (type(em_iterations) == int and em_iterations > 0):
        incorrect_inputs[6] = True
    # Check if the parameter 'tolerance_range' satisfies the specifications
    if not (type(tolerance_range) == list and tolerance_range[0] > tolerance_range[1]):
        incorrect_inputs[7] = True
    # Check if the parameter 'model_components' for the model is an integer
    if not ((type(model_components) == int and model_components > 0) or model_components == None):
        incorrect_inputs[8] = True
    # Check if the parameter 'model_covariance' is within the allowed set
    if not model_covariance in ['full', 'tied', 'diag', 'spherical']:
        incorrect_inputs[9] = True
    # Check if the parameter 'parameter_init' is within the allowed set
    if not parameter_init in ['kmeans', 'random']:
        incorrect_inputs[10] = True
    # Check if the parameter 'model_verbosity' is within the allowed set
    if not model_verbosity in [0, 1, 2]:
        incorrect_inputs[11] = True
    # Check if the parameter 'mpi_parallelization' is within the allowed set
    if not mpi_parallelization in [True, False]:
        incorrect_inputs[12] = True
    # Check if the parameter 'processes' for parallelization is an integer
    if not (type(processes) == int and processes > 0):
        incorrect_inputs[13] = True
    # Check if the parameter 'weights_and_model' is within the allowed set
    if not weights_and_model in [True, False]:
        incorrect_inputs[14] = True
    # Check whether the number of walkers is at least twice the parameter number
    if (initial_samples[0] == 'automatic' and len(initial_samples) == 3
        and [type(initial_samples[i]) for i in range(0,3)] == [str, int, int]):
        try:
            if not initial_samples[1] > (2 * len(parameter_ranges)):
                incorrect_inputs[15] = True
        except:
            # Let the function resume, as a faulty 'parameter_ranges' will terminate
            pass
    # Check whether the truncation parameter falls within the allowed range
    if not (type(truncation_alpha) == float and (0.0 <= truncation_alpha)):# <= 3.0)):
        incorrect_inputs[16] = True
    # Check whether the model selection parameter is within the allowed set
    if not model_selection in [None, 'gmm', 'kde']:
        incorrect_inputs[17] = True
    # Check whether the KDE bandwidth parameter is a positive float value
    if not (type(kde_bandwidth) == float and kde_bandwidth > 0.0):
        incorrect_inputs[18] = True
    # Check if the provided parameter 'convergence_threshold' is a flaot or None
    if not convergence_threshold is None:
        if not (type(convergence_threshold) == float and convergence_threshold > 0):
            incorrect_inputs[19] = True
    # Define error messages for each parameter in case of unsuitable inputs
    errors = ['ERROR: parameter_ranges: 2-column numpy.ndarray, first column lower bounds',
              'ERROR: posterior_evaluation: Must be a suitable function for this problem',
              'ERROR: output_samples: Must be an integer above 0 to be a valid input',
              'ERROR: initial_samples: ["automatic", int, int] or ["custom", array-like]',
              'ERROR: gaussbock_iterations: Must be an integer above 0 or None to be a valid input',
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
              'ERROR: kde_bandwidth: Must be a float above 0 to be a valid input',
              'ERROR: convergence_threshold: Must be a float above 0 or None to be a valid input']
    # If there are any unsuitable inputs, print error messages and terminate
    if any(value == True for value in incorrect_inputs):
        for i in range(0, len(errors)):
            if incorrect_inputs[i] == True:
                print(errors[i])
        sys.exit()
