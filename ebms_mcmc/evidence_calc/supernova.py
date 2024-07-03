import os
import logging
from typing import Tuple
import numpy as np
from scipy.special import binom, factorial
from scipy.integrate import solve_ivp
from scipy.stats import norm

import pymultinest

from ebms_mcmc.util.logger import separator

class Supernova:
    """
    Class representing a supernova object to calculate the log evidence for a given polynomial model
    for dark energy equation of state with the Pantheon+ dataset.

    Attributes:
        z_data (numpy.ndarray): Array of redshift values.
        z_hel (numpy.ndarray): Array of heliocentric redshift values.
        m_obs (numpy.ndarray): Array of observed apparent magnitudes.
        covmat (numpy.ndarray): Covariance matrix of the observed data.
        n_data (int): Number of data points.
        y_data_inv_covariance (numpy.ndarray): Inverse covariance matrix of the observed data.
        params (dict): Dictionary of parameters.
        max_poly_deg (int): Maximum polynomial degree.
    
    Methods:
        __init__(self, params: dict) -> None: Initializes the Supernova object.
        write_evidence_file(self, bin_rep: numpy.ndarray, log_evidence: float, log_evidence_error: float) -> None: Writes evidence data to a file.
        distance_lum_ode(self, z: float, d_L: float, bin_model: numpy.ndarray, theta: numpy.ndarray, omega_m: float) -> float: Calculates the luminosity distance.
        calc_dist_mod(self, bin_model: numpy.ndarray, theta: numpy.ndarray, omega_m: float) -> numpy.ndarray: Calculates the distance modulus.
        log_likelihood(self, bin_model: numpy.ndarray, thetas: numpy.ndarray, omega_m: float, M: float) -> float: Calculates the log likelihood.
        log_evidence(self, bin_model: numpy.ndarray) -> Tuple[float, float]: Calculates the log evidence.
        append_to_evidence_file(self, bin_model: numpy.ndarray, log_evidence: float, log_evidence_error: float) -> None: Appends evidence data to a file.
    """
    def __init__(self, params: dict) -> None:
        """
        Initializes the Supernova object.

        Args:
            params (dict): Dictionary of parameters.
        """
        # Data handling
        data_path = 'data/pantheon_data.npz'
        data = np.load(data_path)
        self.z_data = data['z_cmb']
        self.z_hel = data['z_hel']
        self.m_obs = data['m_obs']
        self.covmat = data['covmat']
        self.n_data = len(self.z_data)
        self.y_data_inv_covariance = np.linalg.inv(self.covmat)
        
        # Param handling
        self.params = params
        self.max_poly_deg = params['max_poly_degree']
        
        # Evidence file handling
        bin_model = []
        log_evidence = []
        log_evidence_error = []
        try:
            with open(params['resume_evidence_file'], 'r') as f:
                lines = f.readlines()
            for line in lines[1:]:
                bin_rep = np.array([int(i) for i in line.split()[0].split(',')])
                if len(bin_rep) > self.max_poly_deg + 1:
                    bin_rep = bin_rep[:self.max_poly_deg + 1]
                bin_model.append(bin_rep)
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
    
    def distance_lum_ode(self, z: float, d_L: float, bin_model: np.array, theta: np.array, omega_m: float) -> float:
        """
        Calculates the luminosity distance.

        Args:
            z (float): Redshift value.
            d_L (float): Luminosity distance.
            bin_model (numpy.ndarray): Binary model.
            theta (numpy.ndarray): Parameter values.
            omega_m (float): Omega_m value.

        Returns:
            float: Calculated luminosity distance.
        """
        HUBBLE = 70
        SPEED_OF_LIGHT = 299792458
        a = 1 / (1 + z)
        factors_dlumi = np.zeros(self.max_poly_deg)
        for j in range(1, self.max_poly_deg + 1):
            if bin_model[j] == 1:
                for k in range(1, j + 1):
                    factors_dlumi[j - 1] += (-1) ** k / k * binom(j, k) * (a ** k - 1)
        
        k_value = np.arange(0, self.max_poly_deg + 1, dtype=int)
        I1 = np.log(a) * (1 + np.sum(theta / factorial(k_value)))
        I2 = np.sum(factors_dlumi * theta[1:] / factorial(k_value[1:]))
        exponent = -3 * (I1 + I2)
        H = HUBBLE * np.sqrt(omega_m * (1 + z) ** 3 + (1 - omega_m) * np.exp(exponent))
        dd_L = d_L / (1 + z) + (1 + z) * SPEED_OF_LIGHT / H
        return dd_L
    
    def calc_dist_mod(self, bin_model: np.array, theta: np.array, omega_m: float) -> np.array:
        """
        Calculates the distance modulus.

        Args:
            bin_model (numpy.ndarray): Binary model.
            theta (numpy.ndarray): Parameter values.
            omega_m (float): Omega_m value.

        Returns:
            numpy.ndarray: Array of calculated distance moduli.
        """
        # Calculate distance modulus
        t_span = (0, self.z_data[-1])
        y0 = np.array([0.])
        sol = solve_ivp(self.distance_lum_ode, t_span, y0, t_eval=self.z_data, method='RK45', args=(bin_model, theta, omega_m))
        try:
            d_L = sol.y[0]
        except:
            d_L = np.inf
        redshift_corr = (1 + self.z_data) / (1 + self.z_hel)
        modulus = 5 * np.log10(d_L * redshift_corr) + 10
        return modulus
    
    def log_likelihood(self, bin_model: np.array, thetas: np.array, omega_m: float, M: float) -> float:
        """
        Calculates the log likelihood.

        Args:
            bin_model (numpy.ndarray): Binary model.
            thetas (numpy.ndarray): Parameter values.
            omega_m (float): Omega_m value.
            M (float): M value.

        Returns:
            float: Calculated log likelihood.
        """
        for i, b in enumerate(bin_model):
            if b == 0:
                thetas = np.insert(thetas, i, 0)
        y_pred = self.calc_dist_mod(bin_model, thetas, omega_m)
        residuals = self.m_obs - M - y_pred
        return -0.5 * (residuals).T @ self.y_data_inv_covariance @ residuals
    
    def log_evidence(self, bin_model: np.array) -> Tuple[float, float]:
        """
        Calculates the log evidence.

        Args:
            bin_model (numpy.ndarray): Binary model.

        Returns:
            Tuple[float, float]: Calculated log evidence and log evidence error.
        """
        logging.info('Calculating evidence for bin model: {}'.format(bin_model))
        loglike = lambda cube, ndim, nparams: self.log_likelihood(bin_model, np.array([cube[i] for i in range(ndim - 2)]), cube[ndim - 2], cube[ndim - 1])
        w_min, w_max = self.params['param_prior_range']
        
        def prior_uniform(cube, ndim, nparams):
            for i in range(ndim - 2):
                cube[i] = w_min + (w_max - w_min) * cube[i]
            omega_m_min, omega_m_max = 0, 1
            M_min, M_max = -22, -17
            cube[ndim - 2] = omega_m_min + (omega_m_max - omega_m_min) * cube[ndim - 2]
            cube[ndim - 1] = M_min + (M_max - M_min) * cube[ndim - 1]
        
        def prior_gaussian(cube, ndim, nparams):
            mu = (w_min + w_max) / 2
            sigma = w_max - mu
            rvs = norm(loc=mu, scale=sigma)
            for i in range(0, ndim - 2):
                cube[i] = rvs.ppf(cube[i])
            omega_m_min, omega_m_max = 0, 1
            M_min, M_max = -22, -17
            cube[ndim - 2] = omega_m_min + (omega_m_max - omega_m_min) * cube[ndim - 2]
            cube[ndim - 1] = M_min + (M_max - M_min) * cube[ndim - 1]
        
        n_params = bin_model.sum() + 2
        bin_model_str = ''.join([str(i) for i in bin_model])
        output_multinest = self.params['name'] + 'chains_multinest/' + bin_model_str + '/'
        os.makedirs(output_multinest, exist_ok=True)
        
        if self.params['param_prior'] == 'uniform':
            prior = prior_uniform
        elif self.params['param_prior'] == 'gaussian':
            prior = prior_gaussian
        else:
            logging.error('Prior not implemented')
        
        pymultinest.run(loglike, prior, n_params, outputfiles_basename=output_multinest, resume=False, **self.params['multinest_params'])
        json_data = pymultinest.Analyzer(n_params=n_params, outputfiles_basename=output_multinest)
        stats = json_data.get_stats()
        log_evidence = stats['nested importance sampling global log-evidence']
        log_evidence_error = stats['nested importance sampling global log-evidence error']
        self.write_evidence_file(bin_model, log_evidence, log_evidence_error)
        logging.info('log_evidence: {}'.format(log_evidence))
        separator()
        return log_evidence, log_evidence_error
    
    def append_to_evidence_file(self, bin_model: np.array, log_evidence: float, log_evidence_error: float) -> None:
        """
        Appends evidence data to a file.

        Args:
            bin_model (numpy.ndarray): Binary model.
            log_evidence (float): Log evidence.
            log_evidence_error (float): Log evidence error.
        """
        evi = np.load(self.params['name'] + 'evidence.npz')
        bin_model = np.append(evi['bin_model'], bin_model)
        log_evidence = np.append(evi['log_evidence'], log_evidence)
        log_evidence_error = np.append(evi['log_evidence_error'], log_evidence_error)
        np.savez(self.params['name'] + 'evidence.npz', bin_model=bin_model, log_evidence=log_evidence, log_evidence_error=log_evidence_error)
        
        
        
    