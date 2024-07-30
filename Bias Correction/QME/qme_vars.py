import numpy as np
from qme_utils import qme_var

def get_qme_var(var):
    """
    Return a default set of options for a given variable name.
    This is convenient for commonly used variables and can be added to or adjusted as required.
    Example comments are included for pr. It is important to note that simply changing max_bin
    will be ineffective without also adjusting the scaling and unscaling functions to compensate.
    Additionally, scaling and unscaling should keep in mind the given lower and upper lims.
    """

    # var is already an instance of the qme_var class
    # this is here to save checking at other times
    if isinstance(var, qme_var):
        return var

    # Dictionary to hold variable configurations
    var_configs = {
        "pr": {
            "lower_lim": 0,
            "upper_lim": 1250,
            "scaling": lambda x: np.log(x + 1) * 70,
            "unscaling": lambda x: np.exp(x / 70) - 1
        },
        "tasmax": {
            "lower_lim": -30,
            "upper_lim": 60,
            "scaling": lambda x: (x + 35) * 5,
            "unscaling": lambda x: (x / 5) - 35
        },
        "tasmin": {
            "lower_lim": -50,
            "upper_lim": 40,
            "scaling": lambda x: (x + 55) * 5,
            "unscaling": lambda x: (x / 5) - 55
        },
        "wswd": {
            "lower_lim": 0,
            "upper_lim": 45,
            "scaling": lambda x: x * 10,
            "unscaling": lambda x: x / 10
        },
        "FFDI": {
            "lower_lim": 0,
            "upper_lim": 200,
            "scaling": lambda x: x,
            "unscaling": lambda x: x
        },
        "rsds": {
            "lower_lim": 0,
            "upper_lim": 45,
            "scaling": lambda x: x * 10,
            "unscaling": lambda x: x / 10
        },
        "rh": {
            "lower_lim": 0,
            "upper_lim": 110,
            "scaling": lambda x: x * 4,
            "unscaling": lambda x: x / 4
        }
    }

    # Debugging print statement
    print(f"Getting QME variable configuration for: {var}")

    # Check if the variable name is in the dictionary
    if var not in var_configs:
        raise ValueError("Cannot find a matching variable from given input, check qme_vars.py")

    # Extract the variable configuration
    config = var_configs[var]

    # Default max_bin value
    max_bin = 500

    return qme_var(config["lower_lim"], config["upper_lim"], max_bin, config["scaling"], config["unscaling"])
