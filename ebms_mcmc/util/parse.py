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
    with open(file, 'r') as stream:
        try:
            params = yaml.safe_load(stream)
            return params
        except yaml.YAMLError as exc:
            print(exc)

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
        logging.info("Multiple runs found with the same name")
        sys.exit()
    elif len(dir) == 0:
        return None, False
    else:
        logging.info("Warning: Continuing in an existing directory")
        return dir[0], True

def setup_dir(file: str) -> str:
    """
    Set up the directory structure for the training.

    Args:
        file (str): The path to the input file.

    Returns:
        str: The full path to the created directory.
    """
    params = parse(file)
    full_run_name, run_exists = find_run(params['name'])
    if not run_exists:
        now = datetime.now()
        full_run_name = "output/" + params['name'] + "_" + now.strftime("%Y%m%d")
    os.makedirs(full_run_name, exist_ok=True)
    os.makedirs(full_run_name+'/models/', exist_ok=True)
    shutil.copy(file, full_run_name)
    return full_run_name

def prep_output(file: str) -> Tuple[str, str]:
    """
    Prepare the output directory for a specific run.

    Args:
        file (str): The file path of the input file.

    Returns:
        Tuple[str, str]: A tuple containing the run name and the plot directory.

    Raises:
        AssertionError: If multiple or no runs are found with the same name.
    """
    params = parse(file)
    dir_snippet = params['name']
    plot_dir = params['plot']['plot_dir']
    run_name, run_exists = find_run(dir_snippet)
    if not run_exists:
        logging.info("No run found with the same name")
        sys.exit()
    plot_dir = run_name + '/' + plot_dir + '/'
    os.makedirs(plot_dir, exist_ok=True)
    shutil.copy(file, plot_dir)
    return run_name, plot_dir
    
    