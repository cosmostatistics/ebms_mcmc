<h2 align="center">ebms-mcmc: Evidence Based Model Selection with MCMC</h2>

<p align="center">
<!-- <a href="https://arxiv.org/abs/2401.04174"><img alt="Arxiv" src="https://img.shields.io/badge/arXiv-2401.04174-b31b1b.svg"></a> -->


**ebms-mcmc** is a tool for exploring polynomial model spaces with Markov walks and performing Bayesian model selection. It can extract the model posterior for (i) a toy model with polynomial data and for (ii) the dark energy equation of state in a polynomial parametrisation from the [Pantheon+][Pantheon+] dataset (needs to be manually included).

[Pantheon+]: https://github.com/PantheonPlusSH0ES/DataRelease

## Installation

```sh
# clone the repository
git clone https://github.com/cosmostatistics/ebms_mcmc
# then install in dev mode
cd ebms_mcmc
pip install -e .
```
If anything besides Gaussian priors in the toy model is used you also need [PyMulitNest][PyMulitNest], follow their installation, and then
```sh
# final installation step for pymultinest
pip install -e . [non_anlytical_evidence]
```

[PyMulitNest]: https://github.com/JohannesBuchner/PyMultiNest


## Usage
There are four basic commands implemented, which all have the same structure:
```
ebms_mcmc task params/paramcard.yaml --verbose
```
The ```task``` specifies what to do, ```paramcard.yaml``` sets the parameters different from the default values (defined in ```__init__.py```) and ```--verbose``` prints the output in the console.

### 1. Toy data generation:

```
ebms_mcmc data params/paramcard.yaml --verbose
```

### 2. Perform the MCMC run:

```
ebms_mcmc run params/paramcard.yaml --verbose
```

### 3. Analyse and plot the results:
```
ebms_mcmc plot params/paramcard.yaml --verbose
```

### 4. All of the above in one call:
```
ebms_mcmc all params/paramcard.yaml --verbose
```

## Include your own models

To add a new model, create a file named **'new_model.py'** in the **'ebms_mcmc/evidence_calc/'** directory. This file should contain a function that calculates the log evidence. The function signature should be:
```
def log_evidence(bin_model: np.array) -> Tuple[float, float]:
    """
    Calculate the log evidence.

    Parameters:
    bin_model (np.array): The binary model array.

    Returns:
    Tuple[float, float]: A tuple containing the log evidence and its error.
    """
```
Additionally, you need to ensure that your function writes the calculated log evidence to a file. You can follow the example provided by the **'write_evidence_file'** function in **'ebms_mcmc/evidence_calc/polynomials.py'**.


## Acknowledgements

If you use any part of this repository please cite the following paper:
<!-- 
```
@article{Schosser:2024aic,
    author = "Schosser, Benedikt and Heneka, Caroline and Plehn, Tilman",
    title = "{Optimal, fast, and robust inference of reionization-era cosmology with the 21cmPIE-INN}",
    eprint = "2401.04174",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    month = "1",
    year = "2024"
}
``` -->


