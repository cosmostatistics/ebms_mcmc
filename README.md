<h2 align="center">ebms-mcmc: Evidence based model selection with MCMC</h2>

<p align="center">
<!-- <a href="https://arxiv.org/abs/2401.04174"><img alt="Arxiv" src="https://img.shields.io/badge/arXiv-2401.04174-b31b1b.svg"></a> -->


ebms-mcmc is a tool for exploring polynomial model spaces with markov walks and performing bayesian model selection. It can extract the model posterior for a toy model with polynomial data and the dark energy equation of state in a polynomial parametrisation from the Pantheon+ dataset.

## Installation

```sh
# clone the repository
git clone https://github.com/cosmostatistics/ebms_mcmc
# then install in dev mode
cd ebms_mcmc
pip install -e .
```
if non-analytical evidences are need, you also need [PyMulitNest][PyMulitNest], follow their installation, and then
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
The ```task``` specifies what to do, the param card sets the parameters, which should not be the default value and ```--verbose``` prints the output in the console, too/

1. Toy data generation:

```
ebms_mcmc data params/paramcard.yaml --verbose
```

2. Perform the mcmc run:

```
ebms_mcmc run params/paramcard.yaml --verbose
```

3. Analysis and plot the output:
```
ebms_mcmc plot params/paramcard.yaml --verbose
```

4. All of the above in one go:
```
ebms_mcmc all params/paramcard.yaml --verbose
```

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


