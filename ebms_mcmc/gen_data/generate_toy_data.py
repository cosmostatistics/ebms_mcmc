import logging
import os
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

def gen_toy_data(max_poly_degree: int = 2,
                 noise_scale: float = 0.1,
                 num_points: int = 300,
                 x_range: List = [-1, 1],
                 params: dict = {'method': 'random', 'bin': None, 'range': [-1, 1]},
                 plot: bool = False,
                 save: bool = True,
                 name: str = 'data/',
                 **kwargs) -> Tuple[np.array, np.array, np.array]:
    """
    Generate toy data for polynomial model selection.
    
    Parameters:
    - max_poly_degree (int): The maximum degree of the polynomial to generate.
    - noise_scale (float): The scale of the Gaussian noise added to the data.
    - num_points (int): The number of data points to generate.
    - x_range (List): The range of x values.
    - params (dict): The parameters for generating the polynomial coefficients.
        - method (str): The method for generating the coefficients. Options are 'random', 'leaky', and 'fixed'.
            (1) 'random': Randomly generate the coefficients within the specified range.
            (2) 'leaky': Randomly generate the coefficients within the specified range, but with some coefficients set to zero.
            (3) 'fixed': Use the specified coefficients and binary mask.
        - bin (List or None): The binary mask for the coefficients. If None, all coefficients are set accordingly to the method specified.
        - range (List): The range of values for generating the coefficients.
    - plot (bool): Whether to plot the generated data.
    - save (bool): Whether to save the generated data.
    - name (str): The directory name for saving the data.
    - **kwargs: Additional keyword arguments.
    
    Returns:
    - toy_data (np.array): The generated toy data.
    - parameter (np.array): The polynomial coefficients.
    - bin (np.array): The binary mask for the coefficients.
    """
    
    x = np.linspace(x_range[0], x_range[1], num_points)
    shape = x.shape
    noise = np.random.normal(loc=0, scale=noise_scale, size=shape)
    
    if params['method'] == 'random':
        parameter = np.random.uniform(params['range'][0], params['range'][1], max_poly_degree+1)
        bin = np.ones(max_poly_degree+1)
        
    elif params['method'] == 'leaky':
        parameter = np.zeros(max_poly_degree+1)
        while np.all(params == 0):
            bin = np.ones(max_poly_degree+1)
            for i in range(max_poly_degree+1):
                if np.random.choice([True, False]):
                    parameter[i] = np.random.uniform(params['range'][0], params['range'][1], 1)
                else:
                    parameter[i] = 0
                    bin[i] = 0
                    
    elif params['method'] == 'fixed':
        parameter = np.array(params['params'])
        bin = np.array(params['bin'])
        assert len(parameter) == np.sum(bin), 'Parameter and bin length mismatch'
        for i, b in enumerate(bin):
            if b == 0:
                parameter = np.insert(parameter, i, 0)
                 
    if not params['bin'] is None:
        parameter = parameter * params['bin']
        
    x_pow = np.power(x[:, None], np.arange(0, parameter.shape[0], 1, dtype=np.int64))
    toy_data = x_pow @ parameter + noise
    y_err = np.full(shape, noise_scale)
    
    if save:
        os.makedirs(name, exist_ok=True)
        np.savez(name+'toy_data.npz', x_data=x, y_data=toy_data, y_err=y_err,
                params=parameter, bin=bin)
        
    if plot:
        text = r'$y = '
        for i in range(len(parameter)):
            if bin[i] == 1:
                param = parameter[i]
                if i == 0:
                    # No variable term
                    text += f'{param:.2f} ' if param >= 0 else f'- {-param:.2f} '
                elif i == 1:
                    # First-degree term
                    text += f'+ {param:.2f}x ' if param >= 0 else f'- {-param:.2f}x '
                else:
                    # Higher-degree terms with proper LaTeX exponent formatting
                    text += f'+ {param:.2f}x^{{{i}}} ' if param >= 0 else f'- {-param:.2f}x^{{{i}}} '
        text = text.strip() + '$'

        plt.text(0.1, 0.9, text, fontsize=12, ha='left', va='center', transform=plt.gca().transAxes,
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
        plt.errorbar(x, toy_data, yerr=y_err, fmt='.', ms=5, color='teal', label='Data', alpha=0.5)
        y_true = x_pow @ parameter
        plt.plot(x, y_true, label='True model', zorder=10, color='darkred')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(loc='upper right')
        plt.savefig(name+'toy_data.pdf')
        
    return toy_data, parameter, bin