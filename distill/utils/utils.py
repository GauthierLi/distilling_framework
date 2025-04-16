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

