import yaml
import os
import logging
import sys
import shutil
from typing import Tuple
from datetime import datetime

def parse(file: str) -> dict:
    """
    Parse a YAML file and return the contents as a dictionary.

    Args:
        file (str): The path to the YAML file.

    Returns:
        dict: The contents of the YAML file as a dictionary.

    Raises:
        yaml.YAMLError: If there is an error while parsing the YAML file.
    """
    try:
        with open(file, 'r') as stream:
            try:
                params = yaml.safe_load(stream)
                return params
            except yaml.YAMLError as exc:
                print(exc)
    except:
        print("Wrong path or no file given, continue with default parameters")

def find_run(dir_snippet: str) -> Tuple[str, bool]:
    """
    Find a run directory that matches the given directory snippet.

    Args:
        dir_snippet (str): The snippet to match against the directory names.

    Returns:
        Tuple[str, bool]: A tuple containing the matching directory path and a boolean
        indicating whether a match was found.

    Raises:
        SystemExit: If multiple runs are found with the same name.

    """
    dir = ['output/' + x for x in os.listdir('output/') if dir_snippet in x]
    if len(dir) > 1:
        sys.exit("Error: Multiple runs found with the same name")
    elif len(dir) == 0:
        return None, False
    else:
        print("Warning: Continuing in an existing directory")
        return dir[0]+'/', True

def setup_dir(params: dict) -> str:
    """
    Set up the directory structure for the training.

    Args:
        params (dict): The parameter for the runs.

    Returns:
        str: The full path to the created directory.
    """
    print(params)
    full_run_name, run_exists = find_run(params['name'])
    if not run_exists:
        now = datetime.now()
        full_run_name = "output/" + params['name'] + "_" + now.strftime("%Y%m%d")+'/'
    os.makedirs(full_run_name, exist_ok=True)
    with open(full_run_name+'params.yaml', 'w') as file:
        yaml.dump({'run': params}, file)
    return full_run_name

def prep_output(params: dict) -> Tuple[str, str]:
    """
    Prepare the output directory for a specific run.

    Args:
        params (dict): The parameters for the ploting routing.

    Returns:
        Tuple[str, str]: A tuple containing the run name and the plot directory.

    Raises:
        AssertionError: If multiple or no runs are found with the same name.
    """
    dir_snippet = params['name']
    plot_dir = params['plot_dir']
    run_name, run_exists = find_run(dir_snippet)
    if not run_exists:
        sys.exit("Error: Run not found")
    plot_dir = run_name + '/' + plot_dir + '/'
    os.makedirs(plot_dir, exist_ok=True)
    with open(plot_dir+'params.yaml', 'w') as file:
        yaml.dump({'plot': params}, file)
    return run_name, plot_dir
    
    