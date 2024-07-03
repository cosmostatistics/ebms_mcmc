import argparse
import logging
import shutil
from typing import Tuple, Callable

from .util.parse import parse, setup_dir, prep_output
from .util.logger import init_logger, separator
from .gen_data import generate_toy_data
from .mcmc import MCMC


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
    
    train_parser = subparsers.add_parser("train")
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
    init_logger(fn=params['logging_name'], verbose=args.verbose)
    generate_toy_data(params)
    
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
    params.update({'multinest_params': {'verbose': args.verbose}})  
    evidence_calculator = setup_evidence_calculator(params)
    MCMC(params, evidence_calculator).run()
    
def plot(args: argparse.Namespace) -> None:
    pass

if __name__ == "__main__":  
    main()
    
    
    
    