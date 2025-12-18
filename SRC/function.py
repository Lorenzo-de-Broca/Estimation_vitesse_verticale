import numpy as np
import matplotlib.pyplot as plt


# %% defining functions for manovariable regression
def linear(x, m, b): # linear function
    return m*x + b

def exponential(x, a, b, c): # exponential function
    return a * np.exp(b * x) + c

def logarithmic(x, a, b): # logarithmic function
    return a * np.log(x) + b    