import argparse
import logging
import shutil
import yaml
from typing import Tuple, Callable

from . import default_params
from .util.parse import parse, setup_dir, prep_output
from .util.logger import init_logger, separator
from .gen_data import generate_toy_data
from .mcmc.mcmc import MCMC
from .eval.plotting import Plotting


def main():
    """
    Entry point of the program.
    Parses command line arguments and executes the corresponding function based on the subcommand.
    """
    
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)
    
    data_parser = subparsers.add_parser("data")
    data_parser.add_argument("paramcard", nargs='?', default=None)
    data_parser.add_argument("--verbose", action="store_true")
    data_parser.set_defaults(func=data)
    
    train_parser = subparsers.add_parser("run")
    train_parser.add_argument("paramcard", nargs='?', default=None)
    train_parser.add_argument("--verbose", action="store_true")
    train_parser.set_defaults(func=run)

    plot_parser = subparsers.add_parser("plot")
    plot_parser.add_argument("paramcard", nargs='?', default=None)
    plot_parser.add_argument("--verbose", action="store_true")
    plot_parser.set_defaults(func=plot)
    
    plot_parser = subparsers.add_parser("all")
    plot_parser.add_argument("paramcard",nargs='?', default=None)
    plot_parser.add_argument("--verbose", action="store_true")
    plot_parser.set_defaults(func=all)

    args = parser.parse_args()
    args.func(args)
    
def data(args: argparse.Namespace) -> None:
    """
    Function to generate polynomial toy data
    """
    ind_params = parse(args.paramcard)
    try:
        params = {**default_params['data'], **ind_params['data']}
    except:
        params = default_params['data']
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
    ind_params = parse(args.paramcard)
    try:
        params = {**default_params['run'], **ind_params['run']}
    except:
        params = default_params['run']
    print(params)
    run_name = setup_dir(params)
    params['name'] = run_name
    init_logger(fn=run_name, verbose=args.verbose)
    separator()
    params.update({'multinest_params': {'verbose': args.verbose}})  
    evidence_calculator = setup_evidence_calculator(params)
    MCMC(params).run(evidence_calculator)
    
def plot(args: argparse.Namespace) -> None:
    ind_params = parse(args.paramcard)
    try:
        params = {**default_params['plot'], **ind_params['plot']}
    except:
        params = default_params['plot']
    run_name, plot_dir = prep_output(params)
    params["name"] = run_name
    params['plot_dir'] = plot_dir    
    init_logger(fn=plot_dir, verbose=args.verbose)
    Plotting(params).main()
    
def all(args: argparse.Namespace) -> None:
    data(args)
    run(args)
    plot(args)
    
if __name__ == "__main__":  
    main()
    
    
    
    