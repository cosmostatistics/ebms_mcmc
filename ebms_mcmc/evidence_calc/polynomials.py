from typing import Tuple
import numpy as np
import logging
import os
from ..util.logger import separator

class Polynomials:
    """
    A class for calculating the log evidence for a given polynomial model and polynomial toy data.

    Attributes:
        x_data (numpy.ndarray): The x data.
        y_data (numpy.ndarray): The y data.
        y_err (numpy.ndarray): The y error.
        n_data (int): The number of data points.
        params (dict): Dictionary of parameters.
        max_poly_degree (int): The maximum polynomial degree.
        x_pow (numpy.ndarray): The power of x data.
        covariance (numpy.ndarray): The covariance matrix.
        inv_covariance (numpy.ndarray): The inverse covariance matrix.
        yCy (float): The product of y_data, inv_covariance, and y_data.
    """

    def __init__(self, params: dict) -> None:
        """
        Initializes the Polynomials class.

        Args:
            params (dict): A dictionary containing the parameters for the class.
        """
        
        # Data handling
        data_path = params['data_path']
        data = np.load(data_path)
        self.x_data = data['x_data']
        self.y_data = data['y_data']
        self.y_err = data['y_err']
        self.n_data = len(self.x_data)
        
        # Param handling
        self.params = params
        self.max_poly_degree = params['max_poly_degree']
        
        # Linear model
        self.x_pow = np.power(self.x_data[:, None], np.arange(0, self.max_poly_degree+1, 1, dtype=np.int64))
        
        # Singular calculations
        self.covariance = np.diag(self.y_err**2)
        self.inv_covariance = np.diag(1 / self.y_err**2)
        self.yCy = self.y_data.T @ self.inv_covariance @ self.y_data      
    
        # Evidence file handling
        bin_model = []
        log_evidence = []
        log_evidence_error = []
        try:
            with open(params['resume_evidence_file'], 'r') as f:
                lines = f.readlines()
            for line in lines[1:]:
                bin_model = np.array([int(i) for i in line.split()[0].split(',')])
                if len(bin_model) > self.max_poly_deg + 1:
                    bin_model = bin_model[:self.max_poly_deg + 1]
                bin_model.append(bin_model)
                log_evidence.append(float(line.split()[1]))
                log_evidence_error.append(float(line.split()[2]))
            logging.info('Resuming evidence calculation from file {}'.format(self.params['resume_evidence_file']))
        except:
            pass
        with open(params['name'] + 'evidence.txt', 'w') as f:
            f.write('Binary representation' + '    ' + 'log evidence' + '    ' + 'log_evidence_error \n')
        for i, b in enumerate(bin_model):
            self.write_evidence_file(b, log_evidence[i], log_evidence_error[i])
    
    def write_evidence_file(self, bin_rep: np.array, log_evidence: float, log_evidence_error: float) -> None:
        """
        Writes evidence data to a file.

        Args:
            bin_rep (numpy.ndarray): Binary representation.
            log_evidence (float): Log evidence.
            log_evidence_error (float): Log evidence error.
        """
        with open(self.params['name'] + 'evidence.txt', 'a') as f:
            bin_rep_write = ','.join([str(i) for i in bin_rep])
            f.write(str(bin_rep_write) + '    ' + str(log_evidence) + '    ' + str(log_evidence_error) + '\n')
    
    def log_evidence(self, bin_model: np.array) -> Tuple[float, float]:
        """
        Calculates the log evidence.

        Args:
            bin_model (numpy.ndarray): The binary model.

        Returns:
            Tuple[float, float]: The log evidence and log evidence error.
        """
        
        active = bin_model == 1
        x_model = self.x_pow[:, active]
        Fisher_matrix = x_model.T @ self.inv_covariance @ x_model
        Q = self.y_data @ self.inv_covariance @ x_model
        
        if self.params['param_prior'] == "one":
            log_evidence, log_evidence_error = self.log_evidence_one(bin_model, Fisher_matrix, Q)
        elif self.params['param_prior'] == "gaussian":
            log_evidence, log_evidence_error = self.log_evidence_gaussian(bin_model, Fisher_matrix, Q, active)
        elif self.params['param_prior'] == "uniform":
            assert self.params['param_prior_range'] is not None, "Please provide prior ranges"
            log_evidence, log_evidence_error = self.log_evidence_uniform(bin_model, Fisher_matrix, Q)
        else:
            logging.error('Parameter prior not implemented.')
            
        self.write_evidence_file(bin_model, log_evidence, log_evidence_error)
        logging.info('log_evidence: {}'.format(log_evidence))
        separator()
        return log_evidence, log_evidence_error        
        
    def log_evidence_one(self, bin_model: np.array, Fisher_matrix: np.ndarray, Q: np.array) -> float:
        """
        Calculates the log evidence using a one parameter prior.

        Args:
            bin_model (numpy.ndarray): The binary model.
            Fisher_matrix (numpy.ndarray): The Fisher matrix.
            Q (numpy.ndarray): The Q matrix.

        Returns:
            float: The log evidence.
        """
        FQQ = Q.T @ np.linalg.inv(Fisher_matrix) @ Q
        log_evidence = 0.5 * bin_model.sum() * np.log(2*np.pi) - 0.5*np.log(np.linalg.det(Fisher_matrix)) + 0.5 * FQQ
        return log_evidence, 0
    
    def log_evidence_gaussian(self, bin_model: np.array, Fisher_matrix: np.ndarray, Q: np.array, active: np.ndarray[np.bool_]) -> float:
        """
        Calculates the log evidence using a Gaussian parameter prior.

        Args:
            bin_model (numpy.ndarray): The binary model.
            Fisher_matrix (numpy.ndarray): The Fisher matrix.
            Q (numpy.ndarray): The Q matrix.
            active (numpy.ndarray): The active array.

        Returns:
            float: The log evidence.
        """
        try:
            prior_Fisher_Gaussian = self.params['prior_gaussian_inv_cov'] * np.eye(self.max_poly_degree+1)[active][:,active]  
        except:
            logging.info('No prior Gaussian inverse covariance provided. Using identity matrix.')
            prior_Fisher_Gaussian = np.eye(bin_model.sum())            
        try:
            prior_exp_Gaussian = self.params['prior_gaussian_exp'] * np.ones(bin_model.sum())
        except:
            logging.info('No prior Gaussian expectation provided. Using zero.')    
            prior_exp_Gaussian = np.zeros(bin_model.sum())
        log_norm_prior = bin_model.sum()/2 * np.log(2*np.pi) - 1/2 * np.log(np.linalg.det(prior_Fisher_Gaussian))
        Gexpexp = prior_exp_Gaussian.T @ prior_Fisher_Gaussian @ prior_exp_Gaussian
        FPQQ = (Q + prior_Fisher_Gaussian @ prior_exp_Gaussian).T @ np.linalg.inv(Fisher_matrix + prior_Fisher_Gaussian) @ (Q + prior_Fisher_Gaussian @ prior_exp_Gaussian)
        log_evidence = - log_norm_prior  - 0.5 * Gexpexp + bin_model.sum()/2 * np.log(2*np.pi) - 0.5 * np.log(np.linalg.det(Fisher_matrix + prior_Fisher_Gaussian)) + 0.5 * FPQQ
        return log_evidence, 0
    
    def log_evidence_uniform(self, bin_model: np.array, Fisher_matrix: np.ndarray, Q: np.array) -> float:
        """
        Calculates the log evidence using a uniform parameter prior.

        Args:
            bin_model (numpy.ndarray): The binary model.
            Fisher_matrix (numpy.ndarray): The Fisher matrix.
            Q (numpy.ndarray): The Q matrix.

        Returns:
            float: The log evidence.
        """
        try:
            import pymultinest
        except ImportError:
            logging.error('PyMultiNest import failed. Please install it to use the uniform parameter prior for the polynomials.')
            raise
        
        def loglike(cube, ndim, nparams):
            thetas = np.array([cube[i] for i in range(ndim)])
            loglikelihood = Q.T @ thetas - 0.5 * thetas.T @ Fisher_matrix @ thetas
            return loglikelihood
        try:
            theta_min, theta_max = self.params['param_prior_range']
        except:
            logging.info('No prior range provided. Using -1 and 1 as default.')
            theta_min, theta_max = -1, 1
                    
        def prior_uniform(cube, ndim, nparams):
            for i in range(ndim):
                cube[i] = theta_min + cube[i] * (theta_max - theta_min)
        
        n_params = bin_model.sum()
        bin_model_str = ''.join([str(i) for i in bin_model])
        output_multinest = self.params['name'] + 'chains_multinest/' + bin_model_str + '/'
        os.makedirs(output_multinest, exist_ok=True)
        
        pymultinest.run(loglike, prior_uniform, n_params, outputfiles_basename=output_multinest, resume = False, **self.params['multinest_params'])
        json_data = pymultinest.Analyzer(n_params=n_params, outputfiles_basename=output_multinest)
        stats = json_data.get_stats()        
        log_evidence = stats['nested importance sampling global log-evidence']
        log_evidence_error = stats['nested importance sampling global log-evidence error']
        return log_evidence, log_evidence_error