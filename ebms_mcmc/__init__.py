default_params = {
    # parameters for the data generation
    'data': 
        {
        # name of the data set in data directory
        'name' : 'toy_data',
        # how many data sets to generate
        'num_data_sets' : 1,
        # maximal polynomial degree of the underlying model
        'max_poly_degree' : 5,
        # how much noise
        'noise_scale' : 0.1,
        # how many data points
        'num_points' : 300,
        # range of the x values
        'x_range' : [-1, 1],
        # how to sample the parameters of the model
        'params' : 
            {
            # either fixed, random, leaky
            # random: random uniform values in range for every possible monomial
            # leaky: random uniform values in range for some monomials
            # fixed: fixed values for some monomials
            'method' : 'fixed',
            # prior range for the parameters if sampled
            'range' : [-1, 1],
            # if method is fixed then these are the parameters, otherwise they are ignored
            # length of bin must match max_poly_degree+1
            'bin' : [0, 1, 1, 0, 1, 0],
            # params for each monomial
            'params' : [0.1 , 0.3, -0.4],
            },
        # whether to plot and save the data
        'plot' : False,
        'save' : True
        },
    
    # parameters for the mcmc run
    'run' : 
        {
        # base name of the run, date is added
        'name' : 'run/',
        # which data, can be toy_data or supernova
        'data_type' : 'toy_data',
        # if toy_data, which data set
        'data_path' : 'data/toy_data/0/toy_data.npz',
        # reuse already calculated evidences, give file path
        'resume_evidence_file' : None,
        # max degree which is explored, should be signifactly larger than the true or expected on
        'max_poly_degree' : 7,
        # how many mcmc steps
        'n_iterations' : 100000,
        # which model prior, either normalisable, AIC, one_over or uniform
        'prior_kind' : 'normalisable',
        # proposal distribution of the mcmc step, either poisson, uniform, or single
        'prop_dist' : 'poisson',
        
        # prior for the parameters
        # warning: if toy data and uniform, then multinest is needed 
        'param_prior' : 'gaussian', 
        'param_prior_range' : [-10, 10],
        'prior_gaussian_mean' : 0,
        'prior_gaussian_inv_cov' : 1,
        'multinest_params' : 
            {
            'evidence_tolerance' : 0.5,
            'n_live_points' : 400,
            }
        },
    
    # parameters for the plotting
    'plot' : 
        {
        # where is the output of the run, base_name is sufficient
        'name' : 'run/',
        # where the plots in the run_dir are saved
        'plot_dir' : 'plots/',
        'burn_in' : 1000,
        'log_plot' : False,
        'plots_to_do' :
            [
            '2d_color',
            '2d_circle',
            'data',
            'marginals'
            ]          
        }
    }
            