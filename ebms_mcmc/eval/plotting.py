import logging
import os
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_pdf import PdfPages

from ..util.logger import separator

class Plotting:
    """
    A class for performing plotting routines.

    Args:
        params (dict): A dictionary containing the parameters.

    Attributes:
        params (dict): A dictionary containing the parameters.
        path (numpy.ndarray): The path data.
        max_poly_deg (int): The maximum polynomial degree.
        chain_2d (numpy.ndarray): The 2D chain data.
        data (numpy.ndarray): The data.
        type (str): The type of data.

    Methods:
        analyse(): Analyzes the data and returns the results.
        bin_to_2d(bin): Converts a binary array to 2D coordinates.
        visualise_chain_2d_color(): Visualizes the chain in 2D with color.
        visualise_chain_2d_circle(): Visualizes the chain in 2D with circles.
        plot_polynomial_data(): Plots the polynomial data.
        plot_supernova_data(): Plots the supernova data.
        save_posterior_list(): Saves the posterior list.
        plot_marginals(): Plots the marginals.
    """

    def __init__(self, params: dict) -> None:
        """
        Initializes the Plotting class.

        Args:
            params (dict): A dictionary containing the parameters.
        """
        logging.info("Performing the Plotting Routine")
        separator()
        self.params = params
        self.path = np.load(self.params['name'] + 'path.npy')[params['burn_in']:]
        self.max_poly_deg = self.path[0].shape[0] - 1
        n_samples = self.path.shape[0]
        self.chain_2d = np.zeros((n_samples,2))
        for i in range(n_samples):
            self.chain_2d[i] = self.bin_to_2d(self.path[i])
        data_files = [f for f in os.listdir(self.params['name']) if f.endswith('.npz')]
        if data_files:
            if len(data_files) > 1:
                logging.warning("Multiple data files found, using the first one.")
            self.data = np.load(os.path.join(self.params['name'], data_files[0]))
            self.type = 'toy_data'
        else:
            pantheon_data_path = 'data/pantheon_data.npz'
            if os.path.exists(pantheon_data_path):
                self.data = np.load(pantheon_data_path)
                self.type = 'supernova'
            else:
                logging.error("No valid data files found.")
                self.data = None
                self.type = None

        os.makedirs(self.params['plot_dir'], exist_ok=True)
            
    def analyse(self) -> Tuple[np.array, np.array, np.array, np.array, np.array]:
        """
        Analyzes the data and returns the results.

        Returns:
            Tuple[np.array, np.array, np.array, np.array, np.array]: A tuple containing the model binary, relative counts, bin mean, polynomial degree probability, and degrees of freedom probability.
        """
        model_binairy,counts = np.unique(self.path,return_counts=True,axis=0)
        total_counts = np.sum(counts)
        relative_counts = counts/total_counts
        bin_mean = np.zeros(self.max_poly_deg+1)
        poly_degree_prob = np.zeros(self.max_poly_deg+1)
        dof_prob = np.zeros(self.max_poly_deg+2)
        
        for i,b in enumerate(model_binairy):
            bin_mean += b*relative_counts[i]
            max_poly_deg = np.max(np.where(b == 1))
            poly_degree_prob[max_poly_deg] += relative_counts[i]
            dof = int(np.sum(b))
            dof_prob[dof] += relative_counts[i]
        return model_binairy, relative_counts, bin_mean, poly_degree_prob, dof_prob
    
    def bin_to_2d(self, bin: np.array) -> Tuple[int,int]:
        """
        Converts a binary array to 2D coordinates.

        Args:
            bin (numpy.ndarray): The binary array.

        Returns:
            Tuple[int,int]: The 2D coordinates.
        """
        y_coord = np.max(np.where(bin == 1))
        x_coord = np.sum(bin[:y_coord+1])
        return x_coord, y_coord
    
    def visualise_chain_2d_color(self) -> None:

        """
        Visualizes the chain in 2D with color.
        """
        logging.info("Visualising the chain in 2D with color")

        max_x = int(np.max(self.chain_2d[:,0])+1)
        max_y = int(np.max(self.chain_2d[:,1])+1)
        plt.figure()
        if self.params['log_plot']:
            plt.hist2d(self.chain_2d[:,0],self.chain_2d[:,1],norm=LogNorm(vmin = 1),
                            bins=[np.arange(0.5,max_x,1),np.arange(-0.5,max_y,1)], density=True)
        else:
            plt.hist2d(self.chain_2d[:,0],self.chain_2d[:,1],
                            bins=[np.arange(0.5,max_x,1),np.arange(-0.5,max_y,1)], density=True)
        cbar = plt.colorbar()
        cbar.set_label('Posterior')   
        try:
            bin_model = self.data['bin']
            x, y = self.bin_to_2d(bin_model)
            plt.scatter(x, y, color='darkred', s=100, marker='X', label='True model')
            plt.legend()
        except:
            pass
        # set the bins which do not exist to white
        for i in range(max_y):
            for j in range(i,max_x):
                plt.fill_between([j+2 - 0.5, j+2 + 0.5], [i - 0.5, i - 0.5],
                                 [i + 0.5, i + 0.5], color='white')    
        plt.ylabel('Polynomial degree')
        plt.xlabel('Degrees of freedom')
        plt.xticks(np.arange(1,max_x,1))
        plt.yticks(np.arange(0,max_y,1))
        plt.gca().invert_yaxis()
        plt.savefig(self.params['plot_dir'] + 'chain_2d_color.pdf')
        plt.show()
        plt.close()
        
    def visualise_chain_2d_circle(self) -> None:
        """
        Visualizes the chain in 2D with circles.
        """
        logging.info("Visualising the chain in 2D with circles")
        max_x = int(np.max(self.chain_2d[:,0])+1)
        max_y = int(np.max(self.chain_2d[:,1])+1)
        counts,_,_ = np.histogram2d(self.chain_2d[:,0], self.chain_2d[:,1], bins=[np.arange(0.5,max_x,1),np.arange(-0.5,max_y,1)], density=False)
        counts_norm = counts/np.sum(counts)
        if self.params['log_plot']:
            counts_norm = np.log10(counts_norm, out=np.zeros_like(counts_norm), where=(counts_norm!=0))
        plot_dot = np.where(counts != 0, 1, 0)
        x = np.arange(1,max_x,1)
        y = np.arange(0,max_y,1)
        X,Y = np.meshgrid(x,y)
        scale_factor = 5000
        plt.figure(figsize=(max_x, max_y))
        sc = plt.scatter(X, Y, s=scale_factor*counts_norm.T, c=counts_norm.T, cmap='viridis',edgecolors='black')
        plt.scatter(X, Y, s=plot_dot.T, c='black')
        try:
            bin_model = self.data['bin']
            x, y = self.bin_to_2d(bin_model)
            plt.scatter(x, y, color='darkred', s=100, marker='X', label='True model')
        except:
            pass
        cbar = plt.colorbar(sc)
        if self.params['log_plot']:
            ticks = cbar.get_ticks()
            ticks = ticks[1::2]
            tick_labels = [tick.get_text() for tick in cbar.ax.get_yticklabels()]
            tick_labels = [f"{10**float(tick):.1e}" if float(tick) < 0 else f"{int(10**float(tick))}" for tick in ticks]
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(tick_labels)
        plt.ylabel('Polynomial degree')
        plt.xlabel('Degrees of freedom')
        # set lables only to integers
        plt.xticks(np.arange(1,max_x,1))
        plt.yticks(np.arange(0,max_y,1))
        plt.gca().invert_yaxis()
        plt.legend()
        plt.savefig(self.params['plot_dir'] + 'chain_2d_circle.pdf')
        plt.show()
        plt.close()
    
    def find_map_params(self) -> Tuple[np.array, np.array]:
        """
        Finds the Maximum A Posteriori (MAP) parameters for the model.
        This function attempts to retrieve 'x_data' and 'y_data' from the instance's data attribute.
        If the data is not found, it logs an error and returns None. It then analyzes the model
        to find the MAP parameters by identifying the model with the highest relative counts.
        The function constructs the model matrix using the active terms and computes the MAP
        parameters (theta_hat) using the normal equation. If any polynomial degrees are not active,
        their corresponding coefficients in theta_hat are set to zero.
        Returns:
            Tuple[np.array, np.array]: A tuple containing:
            - theta_hat (np.array): The MAP parameters for the model.
            - map_model_bin (np.array): The binary representation of the model with the highest relative counts.
        """
        try:
            x_data = self.data['x_data']
            y_data = self.data['y_data']

        except:
            logging.error("No toy data found. Cannot find MAP parameters.")
            logging.error("Analysis of Multinest output not yet implemented")
            # TODO: Implement analysis of multinest output
            return
        model_binairy, relative_counts, _, _, _ = self.analyse()
        max_index = np.argmax(relative_counts)
        map_model_bin = model_binairy[max_index]
        
        active_terms = np.where(map_model_bin == 1)[0]
        x_pow = np.power(x_data[:, None], np.arange(0, active_terms[-1]+1, 1))
        x_model = x_pow[:, active_terms]
        theta_hat = np.linalg.inv(x_model.T @ x_model) @ x_model.T @ y_data
        for i in range(self.max_poly_deg+1):
            if i not in active_terms:
                theta_hat = np.insert(theta_hat, i, 0)
                
        return theta_hat, map_model_bin
    
    def get_map_model_function(self) -> callable:
        """
        Returns the MAP model function.

        Returns:
            callable: The MAP model function.
        """
        parameter, _ = self.find_map_params()
        def model(x: np.array) -> np.array:
            return np.sum([parameter[i]*x**i for i in range(len(parameter))], axis=0)
        return model
        
    def plot_toy_data(self) -> None:
        """
        Plots the polynomial data.
        """
        logging.info("Plotting the polynomial data")
        x_data = self.data['x_data']
        y_data = self.data['y_data']
        y_err = self.data['y_err']
        try:
            parameter = self.data['params']
            bin = self.data['bin']
            source = 'True'
        except:
            parameter, bin = self.find_map_params()
            source = 'MAP'
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

        plt.errorbar(x_data, y_data, yerr=y_err, fmt='.', ms=5, color = 'teal', label='Data', alpha=0.5)
        x_lin_values = np.linspace(np.min(x_data), np.max(x_data), 1000)
        x_pow = np.power(x_lin_values[:, None], np.arange(0, parameter.shape[0], 1, dtype=np.int64))
        y_model = x_pow @ parameter
        plt.plot(x_lin_values, y_model, label=source+' model', zorder = 10, color = 'darkred')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(loc = 'upper right')
        text_x = 0.03
        text_y = 0.93
        text_fs = 12
        text_box = plt.text(text_x, text_y, text, fontsize=text_fs, ha='left', va='center', transform=plt.gca().transAxes,
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
        renderer = plt.gcf().canvas.get_renderer()
        
        bbox_text = text_box.get_window_extent(renderer)
        bbox_text = bbox_text.expanded(1.1, 1.1)  # Add margin
        bbox_legend = plt.gca().get_legend().get_window_extent(renderer)
        while bbox_text.overlaps(bbox_legend):
            text_fs -= 1
            text_box.set_fontsize(text_fs)
            bbox_text = text_box.get_window_extent(renderer)
            bbox_text = bbox_text.expanded(1.1, 1.1)  # Add margin
        plt.savefig(self.params['plot_dir'] + 'polynomial_data.pdf')
        plt.show()
        plt.close()
        
    def plot_supernova_data(self) -> None:
        """
        Plots the supernova data.
        """
        logging.info("Plotting the supernova data")
        z = self.data['z_cmb']
        m_obs = self.data['m_obs']
        cov = self.data['covmat']
        std = np.sqrt(np.linalg.inv(cov).diagonal())
        plt.errorbar(z, m_obs, yerr=std, fmt='.', ms=5, color = 'teal', label='Pantheon+', alpha=0.5)
        plt.legend()
        plt.xlabel('z')
        plt.ylabel('m')
        plt.savefig(self.params['plot_dir'] + 'supernova_data.pdf')
        plt.show()
        plt.close()
        
    def save_posterior_list(self, print_list: bool = False) -> None:
        """
        Saves the posterior list.
        """
        model_binairy, relative_counts, _, _, _ = self.analyse()
        sort_index = np.argsort(relative_counts)[::-1]
        with open(self.params['plot_dir'] + 'posterior_list.txt', 'w') as f:
            f.write('----------------\n')
            f.write('Model\tPosterior\n')
            f.write('----------------\n')
            for i in sort_index:
                f.write(f'{model_binairy[i]}\t{relative_counts[i]}\n')
            f.write('----------------')
        if print_list:
            with open(self.params['plot_dir'] + 'posterior_list.txt', 'r') as f:
                print(f.read())
      
    def plot_marginals(self) -> None:
        """
        Plots the marginals.
        """
        logging.info("Plotting the marginals")
        _, _, bin_mean, poly_degree_prob, dof_prob = self.analyse()
        max_x = int(np.max(self.chain_2d[:,0])+1)
        max_y = int(np.max(self.chain_2d[:,1])+1)
        color = 'teal'
        xlabel = [r'$\theta_ix^{i}\,,i=...$','Polynomial degree (d)','Degrees of freedom (n)']
        ylabel = ['Probability per Term','Probability', 'Probability']       
        fs = 22
        fig, ax = plt.subplots(3,1,figsize=(8,22))
        bar_width = 0.6
        ax[0].bar(np.arange(0,self.max_poly_deg+1), bin_mean, bar_width, color=color,)
        for value in np.arange(0,max_x-1,1) :
            ax[0].axvline(x=value+0.5, color='gray', linestyle='--', linewidth=1)
            
        ax[1].stairs(poly_degree_prob,edges=np.arange(poly_degree_prob.shape[0] + 1) - 0.5, fill=True, alpha=0.2, color=color)
        ax[1].stairs(poly_degree_prob,edges=np.arange(poly_degree_prob.shape[0] + 1) - 0.5, fill=False, 
                linewidth=2, color=color)
        
        ax[2].stairs(dof_prob,edges=np.arange(dof_prob.shape[0] + 1) - 0.5, fill=True, alpha=0.2, color=color)
        ax[2].stairs(dof_prob,edges=np.arange(dof_prob.shape[0] + 1) - 0.5, fill=False, 
                linewidth=2, color=color, label='Posterior')
        try:
            bin_model = self.data['bin']
            correct_terms = np.where(bin_model == 1)[0]
            for term in correct_terms:
                ax[0].axvline(x=term, color='darkred', linestyle='--', linewidth=4)
            ax[1].axvline(x=np.max(term), color='darkred', linestyle='--', linewidth=4)
            ax[2].axvline(x=np.sum(bin_model), color='darkred', linestyle='--', linewidth=4, label='True model')
        except:
            pass
        for i, a in enumerate(ax):
            a.set_xticks(np.arange(0,self.max_poly_deg+1,1))
            a.tick_params(axis='both', labelsize=fs)
            if i == 2:
                a.set_xlim(0.5, max_x+0.5)
                a.set_xticks(np.arange(1,self.max_poly_deg+2,1))
            else:
                a.set_xlim(-0.5, max_y+0.5)
                a.set_xticks(np.arange(0,self.max_poly_deg+1,1))
            a.set_ylim(1e-3,None)
            a.set_xlabel(xlabel[i],fontsize=fs)
            a.set_ylabel(ylabel[i],fontsize=fs)
            if self.params['log_plot']:
                a.set_yscale('log')
        # get handles and labels for the legend
        handles, labels = ax[2].get_legend_handles_labels()
        # make the legend
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=2, fontsize=fs)
        
        plt.savefig(self.params['plot_dir'] + 'marginals.pdf', bbox_inches='tight')
        plt.show()
        plt.close()
        
    def main(self):
        self.save_posterior_list()
        plots_to_do = self.params['plots_to_do']
        if '2d_color' in plots_to_do:
            self.visualise_chain_2d_color()
        if '2d_circle' in plots_to_do:
            self.visualise_chain_2d_circle()
        if 'data' in plots_to_do:
            if self.type == 'toy_data':
                self.plot_toy_data()
            elif self.type == 'supernova':
                self.plot_supernova_data()
        if 'marginals' in plots_to_do:
            self.plot_marginals()
        
        
            
            
        
            
            
       
        