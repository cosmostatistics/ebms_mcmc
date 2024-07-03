import argparse
import logging
import shutil
from typing import Tuple, Callable

from .util.parse import parse, setup_dir, prep_output
from .util.logger import init_logger, separator
from .gen_data import generate_toy_data
from .mcmc.mcmc import MCMC


def main():
    """
    Entry point of the program.
    Parses command line arguments and executes the corresponding function based on the subcommand.

    Usage:
        python __main__.py data <paramcard> [--verbose]
        python __main__.py run <paramcard> [--verbose]
        python __main__.py plot <paramcard> [--verbose]
    """

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)
    
    data_parser = subparsers.add_parser("data")
    data_parser.add_argument("paramcard")
    data_parser.add_argument("--verbose", action="store_true")
    data_parser.set_defaults(func=data)
    
    train_parser = subparsers.add_parser("run")
    train_parser.add_argument("paramcard")
    train_parser.add_argument("--verbose", action="store_true")
    train_parser.set_defaults(func=run)

    plot_parser = subparsers.add_parser("plot")
    plot_parser.add_argument("paramcard")
    plot_parser.add_argument("--verbose", action="store_true")
    plot_parser.set_defaults(func=plot)

    args = parser.parse_args()
    args.func(args)
    
def data(args: argparse.Namespace) -> None:
    """
    Function to generate polynomial toy data
    """
    params = parse(args.paramcard)
    dir_name = 'data/'+params['name']+'/'
    init_logger(fn=dir_name, verbose=args.verbose)
    logging.info(f"Generating {params['num_data_sets']} toy data sets")
    logging.info("with the following settings:")
    separator()
    for key, value in params.items():
        logging.info(f"{key}: {value}")
    for i in range(params['num_data_sets']):
        params['name'] = dir_name+'/'+str(i)+'/'
        data, param, bin = generate_toy_data.gen_toy_data(**params)
    
def setup_evidence_calculator(params: dict) -> Callable:
    """
    Function to setup the evidence calculator
    """
    if params['data_type'] == 'supernova':
        from .evidence_calc.supernova import Supernova
        return Supernova(params).log_evidence
    elif params['data_type'] == 'toy_data':
        from .evidence_calc.polynomials import Polynomials
        return Polynomials(params).log_evidence
    else:
        raise logging.error(f"Data type {params['data_type']} not implemented.")
    
def run(args: argparse.Namespace) -> None:
    """
    Function to run the MCMC
    """
    params = parse(args.paramcard)
    run_name = setup_dir(args.paramcard)
    params['name'] = run_name
    init_logger(fn=run_name, verbose=args.verbose)
    logging.info("Starting MCMC run")
    separator()
    params.update({'multinest_params': {'verbose': args.verbose}})  
    evidence_calculator = setup_evidence_calculator(params)
    MCMC(params).run(evidence_calculator)
    
def plot(args: argparse.Namespace) -> None:
    pass

if __name__ == "__main__":  
    main()
    
    
    
    