from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

def gen_toy_data(max_poly_degree: int = 2,
                 noise_scale: float = 0.1,
                 num_points: int = 300,
                 x_range: List = [-1, 1],
                 method: str = 'random',
                 bin: List = None,
                 param_range: List = [-1, 1],
                 plot: bool = False,
                 save: bool = True,
                 save_path: str = 'data/') -> Tuple[np.array, np.array, List]:
    
    """Funtion to generate toy data including Gaussian noise

    Args:
        x (1d float np array): linspace to generate polynomial on
        max_poly_degree (int, optional): desired polynomial degree. Defaults to 2.
        noise_scale (int, optional): standard deviation of noise. Defaults to 1.
        params (_type_, optional): parameters for polynomial. Defaults to None.
        param_range (int, optional): range for random parameters [-param_range:param_range]. Defaults to 100.

    Returns:
        1d float array, 1d float array: generated polynomial data, used parameter to generate polynomial
    """
    x
    shape = x.shape
    noise = np.random.normal(loc = 0, scale = noise_scale, size=shape)
    
    if method == 'Random':
        params = np.random.rand(max_poly_degree+1)*2*param_range - param_range
        bin = np.ones(max_poly_degree+1)
        
    elif method == 'leaky':
        params = np.zeros(max_poly_degree+1)
        while np.all(params == 0):
            bin = np.ones(max_poly_degree+1)
            for i in range(max_poly_degree+1):
                if np.random.choice([True, False]):
                    params[i] = np.random.rand()*2*param_range - param_range
                else:
                    params[i] = 0
                    bin[i] = 0
    if not bin is None:
        params = params * bin
        
    if params.shape[0] != max_poly_degree+1:
        print("Shape of given parameters doesn't fit degree, random parameters created.")
        params = np.random.rand(max_poly_degree+1)*2*param_range - param_range
    x_pow = np.power(x[:, None], np.arange(0, max_poly_degree+1, 1, dtype=np.int64))
    toy_data = x_pow @ params + noise
    
    #Visualisze toy data
    if plot:
        plt.scatter(x, toy_data, marker=".")
        plt.xlabel("x")
        plt.ylabel("y_gt")
        plt.title("Plot of Toy Data")
        plt.show()
    
    if save:
        np.save(data_directory+'toy_data.npy', toy_data)
        np.save(data_directory+'params.npy', params)
        np.save(data_directory+'bin.npy', bin)
        
        plt.scatter(x, toy_data, marker=".")
        plt.xlabel("x")
        plt.ylabel("y_gt")
        plt.title("Plot of Toy Data")
        plt.savefig(data_directory+'toy_data.pdf')
        plt.close()
    
    return toy_data, params, bin