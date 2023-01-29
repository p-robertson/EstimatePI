# main.py
# --
# Python Version: Python 3.11
# Author: Peter M. Robertson
# Date: 2023-01-29
# License: MIT
# --

from concurrent.futures import ProcessPoolExecutor
import math
import random

import numpy as np


def old_monte_carlo(n: int = 10**6) -> float:
    """
    Estimate the value of pi using Monte Carlo method with built-ins.

    Parameters:
    n (int): Number of random points to generate. Default is 10^6.

    Returns:
    float: Estimated value of pi.
    """
    x = [random.random() for _ in range(n)]
    y = [random.random() for _ in range(n)]
    xy_norm = [x[i]**2 + y[i]**2 for i in range(n)]
    return len([norm for norm in xy_norm if norm <= 1]) / n * 4


def _monte_carlo(n: int = 10**6) -> float:
    """
    Estimate the value of pi using Monte Carlo method.

    Parameters:
    n (int): Number of random points to generate. Default is 10^6.

    Returns:
    float: Estimated value of pi.
    """
    # get L2 Norm of a random x and a random y and return calculated pi estimate
    xy = np.add(np.square(np.random.rand(n)), np.square(np.random.rand(n)))
    return xy[xy <= 1].size / n * 4


def monte_carlo(n: int = 10**6, num_limit: int = 10**6) -> float:
    """
    Estimate the value of pi using the Monte Carlo method with parallel processing,
    without overflowing memory.

    Parameters:
    n (int): Number of random points to generate. Default is 10^6.
    num_limit (int): Maximum number of random points to generate in a single process.
                     Default is 10^6.

    Returns:
    float: Estimated value of pi.
    """
    # just run the basic helper function if n is less than our number limit
    # to save compute resources and time.
    if n <= num_limit:
        return _monte_carlo(n)

    # ensure we use safe numbers for chunking
    n0 = n if n >= num_limit else round(n, -math.log10(num_limit))
    n_range = n0 // num_limit

    # use parallel processing to perform n_range monte carlo operations and return their average
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(_monte_carlo, num_limit) for _ in range(n_range)]
        executor.shutdown(wait=True)
    return np.average([future.result() for future in futures])


if __name__ == "__main__":
    import time

    print(f"Real PI: {math.pi}")
    estimation_depth: int = 10**8
    timer0 = time.perf_counter()
    print(f"(NumPy) Estimated PI: {monte_carlo(estimation_depth)}")
    timer0 = time.perf_counter() - timer0
    print(f"{monte_carlo.__name__} executed at a depth of"
          f" {estimation_depth:,} n in {timer0} seconds")

    timer1 = time.perf_counter()
    print(f"(Built-in) Estimated PI: {old_monte_carlo(estimation_depth)}")
    timer1 = time.perf_counter() - timer1
    print(f"{old_monte_carlo.__name__} executed at a depth of"
          f" {estimation_depth:,} n in {timer1} seconds")

    print(f"\nThe Numpy method is {timer1 / timer0} times faster than the built-in method.")
