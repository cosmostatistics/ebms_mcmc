import logging
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_pdf import PdfPages

from ..util.logger import separator

class Plotting:
    def __init__(self, params: dict) -> None:
        self.params = params
        self.path = np.load(self.params['name'] + 'path.npy')
        self.max_poly_deg = self.path[0].shape[0] - 1
        n_samples = self.path.shape[0]
        self.chain_2d = np.zeros((n_samples,2))
        for i in range(n_samples):
            self.chain_2d[i] = self.bin_to_2d(self.path[i])
     
    def analyse(self) -> Tuple[np.array, np.array, np.array, np.array]:
        
        model_binairy,counts = np.unique(self.path,return_counts=True,axis=0)
        total_counts = np.sum(counts)
        relative_counts = counts/total_counts
        bin_mean = np.zeros(self.max_poly_deg+1)
        poly_degree_prob = np.zeros(self.max_poly_deg+1)
        dof_prob = np.zeros(self.max_poly_deg+1)
        
        for i,b in enumerate(model_binairy):
            bin_mean += b*relative_counts[i]
            max_poly_deg = np.max(np.where(b == 1))
            poly_degree_prob[max_poly_deg] += relative_counts[i]
            dof = int(np.sum(b))
            dof_prob[dof] += relative_counts[i]
        return relative_counts, bin_mean, poly_degree_prob, dof_prob
    
    def bin_to_2d(self, bin: np.array) -> Tuple[int,int]:
        y_coord = np.max(np.where(bin == 1))
        x_coord = np.sum(bin[:y_coord+1])
        return x_coord,y_coord
    
    def visualise_chain_3d_color(self) -> None:
        logging.info("Visualising the chain in 3D with color")

        max_x = int(np.max(self.chain_2d[:,0])+1)
        max_y = int(np.max(self.chain_2d[:,1])+1)
        plt.figure()
        plt.hist2d(self.chain_2d[:,0],self.chain_2d[:,1],norm=LogNorm(vmin = 1),
                        bins=[np.arange(0.5,max_x,1),np.arange(-0.5,max_y,1)], density=False)
        cbar = plt.colorbar()
        cbar.set_label('Posterior')
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
        plt.savefig(self.params['plot_dir'] + 'chain_3d_color.pdf')
        plt.close()
        separator()
        
    def visualise_chain_3d_circle(self) -> None:
        logging.info("Visualising the chain in 3D with circles")
        max_x = int(np.max(self.chain_2d[:,0])+1)
        max_y = int(np.max(self.chain_2d[:,1])+1)
        counts,_,_ = np.histogram2d(self.chain_2d[:,0], self.chain_2d[:,1], bins=[np.arange(0.5,max_x,1),np.arange(-0.5,max_y,1)], density=False)
        counts_norm = counts/np.sum(counts)
        log_counts_norm = np.log10(counts_norm, out=np.zeros_like(counts_norm), where=(counts_norm!=0))
        plot_dot = np.where(counts != 0, 1, 0)
        x = np.arange(1,max_x,1)
        y = np.arange(0,max_y,1)
        X,Y = np.meshgrid(x,y)
        scale_factor = 5000
        plt.figure(figsize=(max_x, max_y))
        sc = plt.scatter(X, Y, s=scale_factor*counts_norm.T, c=log_counts_norm.T, cmap='viridis',edgecolors='black')
        plt.scatter(X, Y, s=plot_dot.T, c='black')
        plt.colorbar(sc, label=r'$log_{10}p(M|y)$')

        plt.ylabel('Polynomial degree')
        plt.xlabel('Degrees of freedom')
        # set lables only to integers
        plt.xticks(np.arange(1,max_x,1))
        plt.yticks(np.arange(0,max_y,1))
        plt.gca().invert_yaxis()
        plt.savefig(self.params['plot_dir'] + 'chain_3d_circle.pdf')
        plt.close()
        separator()
    
    # def plot_marginals(self) -> None:
    #     logging.info("Plotting the marginals")
    #     _, bin_mean, poly_degree_prob, dof_prob = self.analyse()
    #     max_x = int(np.max(self.chain_2d[:,0])+1)
    #     max_y = int(np.max(self.chain_2d[:,1])+1)
    #     fs = 18
    #     with PdfPages(self.params['plot_dir'] + 'marginals.pdf') as pdf:
    #         plt.figure()
    #         plt.bar(np.arange(self.max_poly_deg+1),poly_degree_prob,
    #                        lw=2,color='darkred', edgecolor = 'black')
    #         plt.xticks(np.arange(0,self.max_poly_deg+1,1),fontsize=fs)
    #         plt.yticks(fontsize=fs)
    #         plt.xlim(-0.5,max_x+0.5)
    #         plt.ylim(1e-3,None)
    #         plt.xlabel('$w_i(a-1)^{i}\,,i=...$', fontsize=fs)
    #         plt.ylabel('Probability per Term', fontsize=fs)
    #         if self.params['log_plot']:
    #             plt.yscale('log')
    #         pdf.savefig()
    #         plt.close()
    #         plt.figure()
            
            
       
        