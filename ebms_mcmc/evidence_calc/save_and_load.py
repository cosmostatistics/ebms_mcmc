import numpy as np
from typing import Tuple, List

def load_evidence(file_path: str, max_poly_degree: int) -> Tuple[List, List, List]:
    """
    Load evidence from a file.

    Args:
        file_path (str): The path to the file containing the evidence data.
        max_poly_degree (int): The maximum polynomial degree.

    Returns:
        Tuple[List, List, List]: A tuple containing three lists:
            - bin_models: A list of binary models.
            - log_evidence: A list of log evidence values.
            - log_evidence_error: A list of log evidence error values.
    """
    bin_models = []
    log_evidence = []
    log_evidence_error = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
    for line in lines[1:]:
        bin_model = np.array([int(i) for i in line.split()[0].split(',')])
        if (len(bin_model) > (max_poly_degree + 1)) & np.all(bin_model[max_poly_degree + 1:] == 0):
            bin_model = bin_model[:max_poly_degree + 1]
        elif len(bin_model) < (max_poly_degree + 1):
            bin_model = np.concatenate((bin_model, np.zeros(max_poly_degree + 1 - len(bin_model))))
        elif len(bin_model) == (max_poly_degree + 1):
            pass
        else:
            continue
        bin_models.append(bin_model)
        log_evidence.append(float(line.split()[1]))
        log_evidence_error.append(float(line.split()[2]))
    return bin_models, log_evidence, log_evidence_error

def evidence_file_setup(file_path: str) -> None:
    """
    Sets up a new evidence file by creating a new file and writing the header.

    Args:
        file_path (str): The path to the evidence file.

    Returns:
        None
    """
    with open(file_path, 'w') as f:
        f.write('Binary representation' + '    ' + 'log evidence' + '    ' + 'log_evidence_error \n')
          
def write_evidence_file(file_path: str,
                        bin_rep: np.array,
                        log_evidence: float,
                        log_evidence_error: float) -> None:
    """
    Writes evidences to a file.

    Args:
        bin_rep (numpy.ndarray): Binary representation.
        log_evidence (float): Log evidence.
        log_evidence_error (float): Log evidence error.
    """
    with open(file_path, 'a') as f:
        bin_rep_write = ','.join([str(i) for i in bin_rep])
        f.write(str(bin_rep_write) + '    ' + str(log_evidence) + '    ' + str(log_evidence_error) + '\n') 

