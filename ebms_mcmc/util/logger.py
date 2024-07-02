import logging
import os
from importlib import reload 

def separator()->None:
    logging.info( "-----------------------------------------------" )

def init_logger(*, fn: str, 
                verbose: bool = True) -> None:
    """
    Initialize the logger with the specified log file path and verbosity level.

    Args:
        fn (str): The path to the log file directory.
        verbose (bool, optional): Whether to enable verbose logging. Defaults to True.
    """
    os.makedirs(fn, exist_ok=True)
    reload(logging)
    logger = logging.getLogger()
    logger.handlers.clear()  # Clear existing handlers

    # Common formatter for both handlers
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler = logging.FileHandler(f"{fn}/log", mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    logger.addHandler(file_handler)

    if verbose:
        # Console handler (added when verbose is True)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Set the logging level to INFO for the logger
    logger.setLevel(logging.INFO)