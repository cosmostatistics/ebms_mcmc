import logging
from typing import Tuple, List, Callable

import numpy as np
from scipy.special import binom


class MCMC:
    """
    Class representing a Markov Chain Monte Carlo (MCMC) algorithm on a discrete polynomial
    model space.

    Args:
        params (dict): A dictionary containing the parameters for the MCMC algorithm.

    Attributes:
        params (dict): A dictionary containing the parameters for the MCMC algorithm.
        max_poly_deg (int): The maximum polynomial degree.
        binairies (list): A list of binary representations of models.
        log_evidences (list): A list of log evidences for each model.
        log_evidences_error (list): A list of errors in log evidences for each model.

    Methods:
        run: Runs the MCMC algorithm.
        get_evidence: Calculates the evidence for a given model.
        model_log_prior: Calculates the log prior for a given model.
        corrections: Calculates the correction factor for a proposed model.
        find_new_propostion: Finds a new proposition for the next iteration.
        single_step: Performs a single step in the MCMC algorithm.
        acceptance_step: Determines whether to accept or reject a proposed model.
        save: Saves the MCMC results to files.
    """

    def __init__(self, params: dict) -> None:
        """
        Initializes the MCMC object.

        Args:
            params (dict): A dictionary containing the parameters for the MCMC algorithm.
        """
        self.params = params
        self.max_poly_deg = params['max_poly_degree']
        evi = np.load(params['name']+'evidence.npz')
        self.binairies = evi['binaires']
        self.log_evidences = evi['log_evidence'] 
        self.log_evidences_error = evi['log_evidence_error']

    def run(self, 
            evidence_calculator: Callable,
            rep_init: np.array = None) -> None:
        """
        Runs the MCMC algorithm.

        Args:
            evidence_calculator (Callable): A function that calculates the evidence for a given model.
            rep_init (np.array, optional): The initial proposition for the MCMC algorithm. Defaults to None.
        """
        path = []
        if rep_init is None:
            rep_init = np.zeros(self.max_poly_deg+1, dtype=np.int8)
            rep_init[0] = 1
        rep_act = rep_init
        self.get_evidence(rep_act, evidence_calculator)
        path.append(rep_act)
        accept = 0
        for i in range(self.params['n_iterations']):
            rep_prop = self.find_new_propostion(rep_act, self.params['prop_dist'])
            self.get_evidence(rep_prop, evidence_calculator)
            if self.acceptance_step(rep_act, rep_prop):
                rep_act = rep_prop
                accept += 1
            if i % 100 == 0:
                logging.info(f'Iteration {i} of {self.params["n_iterations"]}')
                self.save(path)
            path.append(rep_act)
        logging.info(f'Acceptance rate: {accept/self.params["n_iterations"]}')
        self.save(path)
          
    def get_evidence(self, rep_act: np.array, evidence_calculator: Callable) -> None:
        """
        Calculates the evidence for a given model and appends it to the list.

        Args:
            rep_act (np.array): The binary representation of the model.
            evidence_calculator (Callable): A function that calculates the evidence for a given model.
        """
        if rep_act not in self.binairies:
            self.binairies.append(rep_act)
            log_evidence_act, log_evidence_error = evidence_calculator(rep_act)
            self.log_evidences.append(log_evidence_act)
            self.log_evidences_error.append(log_evidence_error)
        else:
            pass
    
    def model_log_prior(self, rep: np.array, kind: str = 'normalisable') -> float:
        """
        Calculates the log prior for a given model.

        Args:
            rep (np.array): The binary representation of the model.
            kind (str, optional): The type of log prior to calculate. Defaults to 'normalisable'.

        Returns:
            float: The log prior value.
        """
        degree = np.max(np.where(rep == 1))
        freedom = rep.sum()
        if kind == 'normalisable':
            return -(freedom+2)*np.log(degree+1)
        elif kind == 'AIC':
            return -freedom
        # TODO: implement BIC n_data problem
        # elif kind == 'BIC':
        #     return -freedom
        elif kind == 'one_over':
            return -np.log(freedom)
        elif kind == 'uniform':
            return 0
    
    def corrections(self,
                    rep_act: np.array,
                    rep_prop: np.array) -> float:
        """
        Calculates the correction factor for a proposed model.

        Args:
            rep_act (np.array): The binary representation of the current model.
            rep_prop (np.array): The binary representation of the proposed model.

        Returns:
            float: The correction factor.
        """
        pos_highest_power_act = np.max(np.where(rep_act == 1))
        pos_highest_power_prop = np.max(np.where(rep_prop == 1))
        n_initial, k_initial = pos_highest_power_act, np.sum(rep_act)-1
        n_final, k_final =  pos_highest_power_prop, np.sum(rep_prop)-1
        #Check, whether act model is 10000 (i.e. top of Pascal's triangle )
        if n_initial == 0:
            borders_initial = 3
        #check if it is left
        elif n_initial == k_initial:
            borders_initial = 1
        #check if it is right
        elif k_initial == 0:
            borders_initial = 1
        else:
            borders_initial = 0
        # same for final state
        if n_final == 0:
            borders_final = 3
        elif n_final == k_final:
            borders_final = 1
        elif k_final == 0:
            borders_final = 1
        else:
            borders_final = 0
        return binom(n_final, k_final) / binom(n_initial, k_initial) * (0.5)**(borders_initial-borders_final)
    
    
    def find_new_propostion(self, rep_act: np.array, prop_dist: str = 'poisson') -> np.array:
        """
        Finds a new proposition for the next iteration of the MCMC algorithm.

        Args:
            rep_act (np.array): The binary representation of the current model.
            prop_dist (str, optional): The proposal distribution to use. Defaults to 'poisson'.

        Returns:
            np.array: The binary representation of the proposed model.
        """
        if prop_dist == 'poisson':
            steps = np.random.poisson(0.5) + 1
        elif prop_dist == 'uniform':
            steps = np.random.randint(1, 4)
        else:
            raise ValueError('Invalid proposal distribution')
        print(steps)
        for s in range(steps):
            rep_act = self.single_step(rep_act)
        return rep_act

    
    def single_step(self, rep_act: np.array) -> np.array:
        """
        Performs a single step in the MCMC algorithm.

        Args:
            rep_act (np.array): The binary representation of the current model.

        Returns:
            np.array: The binary representation of the proposed model.
        """
        rep_prop = rep_act.copy()
        rep_act_loop = rep_act.copy()
        #Position of highest power in binary representation            
        pos_highest_power = np.max(np.where(rep_act_loop == 1))
        #Choose direction to persue (diagonal vs. horizontal)
        dir = np.random.choice(['d', 'h'])
        while np.all(rep_prop == rep_act):
            print(rep_prop)
            #Performing diagonal move
            if dir == 'd':
                #Choose whether to increase/ decrease polynomial degree
                up_down = np.random.choice([-1,1])
                #Perform increase in polynopmial degree
                if up_down == 1:
                    if pos_highest_power < self.max_poly_deg:
                        rep_prop[pos_highest_power+1] = 1
                        rep_prop[np.random.choice(np.where(rep_act_loop == 1)[0])] = 0
    
            #Perform decrease in polynomial degree (IF: (i) down (ii) not 1000000 (iii) make sure diagonal exists)
                else: # up_down == -1:
                    if pos_highest_power > 0 and rep_act_loop.sum() != (pos_highest_power+1):
                        rep_prop[pos_highest_power] = 0
                        if rep_prop[pos_highest_power-1] == 1:
                            rep_prop[np.random.choice(np.where(rep_act_loop[:pos_highest_power] == 0)[0])] = 1
                        else:
                            rep_prop[pos_highest_power-1] = 1
            #Perform horizontal move
            else:
                add_remove = np.random.choice([-1,1])
                #Add one degree
                if add_remove == 1:
                    if rep_prop.sum() < pos_highest_power +1:
                        rep_prop[np.random.choice(np.where(rep_act_loop[:pos_highest_power] == 0)[0])] = 1
                #Remove one degree
                else: #add_remove == -1
                    if rep_prop.sum() > 1:
                        rep_prop[np.random.choice(np.where(rep_act_loop[:pos_highest_power] == 1)[0])] = 0
        return rep_prop
    
    def acceptance_step(self,
                       rep_act: np.array,
                       rep_prop: np.array,
                       ) -> bool:
        """
        Determines whether to accept or reject a proposed model.

        Args:
            rep_act (np.array): The binary representation of the current model.
            rep_prop (np.array): The binary representation of the proposed model.

        Returns:
            bool: True if the proposed model is accepted, False otherwise.
        """
        # Find the evidences for both models
        log_evi_act = self.log_evidences[self.binairies.index(rep_act)]
        log_evi_act_error = self.log_evidences_error[self.binairies.index(rep_act)]
        log_evi_prop = self.log_evidences[self.binairies.index(rep_prop)]
        log_evi_prop_error = self.log_evidences_error[self.binairies.index(rep_prop)]
        #Sample log evidences
        log_evi_act_sample = np.random.normal(log_evi_act, log_evi_act_error)
        log_evi_prop_sample = np.random.normal(log_evi_prop, log_evi_prop_error)
        #Calculate log prior for both models
        log_prior_act = self.model_log_prior(rep_act, kind = self.params['prior_kind'])
        log_prior_prop = self.model_log_prior(rep_prop, kind = self.params['prior_kind'])
        proba = np.exp( log_evi_act_sample + log_prior_act - log_evi_prop_sample - log_prior_prop)
        proba = proba * self.corrections(rep_act, rep_prop)
        if proba > 1:
            return True
        else:
            return np.random.choice([True, False], p = [proba, 1-proba])
    
    def save(self, path: List) -> None:
        """
        Saves the MCMC results to files.

        Args:
            path (List): A list of binary representations of models visited during the MCMC algorithm.
        """
        output_dir = self.params['output_dir']
        np.savez(output_dir + 'evidences.npz',
                 binaires = self.binairies,
                 log_evidence = self.log_evidences,
                 log_evidence_error = self.log_evidences_error)
        np.save(output_dir + 'path.npy', path)

