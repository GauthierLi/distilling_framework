import os 
from typing import List, Dict, Any, Callable

def batch_call(a: List[Callable], b: List[List[Any]]) -> List[Any]:
    """
        Batch operation, which a is a list of callable functions, and b stored
        the arguments for each callable function

        a: List[Callable]: List of callable functions
        b: List[List[Any]]: List of arguments for each callable function
        return: List[Any]: List of results from each callable function
    """
    assert len(a) == len(b), "The length of the two lists must be the same"
    results = []
    for caller, value in zip(a, b):
        res = caller(*value)
        results.append(res)
    return results

def make_save_dir(dir: str) -> None:
    """
        Create a directory if it does not exist

        dir: str: Directory to be created
    """
    if not os.path.exists(dir):
        os.makedirs(dir)
    ckpt_dir = os.path.join(dir, "ckpt")
    log_dir = os.path.join(dir, "logs")
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
