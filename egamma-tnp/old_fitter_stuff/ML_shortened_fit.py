import argparse
import os
from functools import partial

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import uproot
from numpy.polynomial.chebyshev import Chebyshev
from scipy.optimize import minimize
from scipy.special import voigt_profile
from scipy.stats import poisson
import numdifftools as nd # Added for Hessian calculation

def load_histogram(root_file, hist_name):
    keys = {key.split(";")[0]: key for key in root_file.keys()}
    if hist_name in keys:
        obj = root_file[keys[hist_name]]
        if isinstance(obj, uproot.behaviors.TH1.Histogram):
            values, edges = obj.to_numpy()
            return {"values": values, "edges": edges, "errors": obj.errors()}
    return None

def create_fixed_param_wrapper(func, fixed_params):
    def wrapped(x, *free_params):
        full_params = []
        free_idx = 0
        for i in range(len(fixed_params) + len(free_params)):
            if i in fixed_params:
                full_params.append(fixed_params[i])
            else:
                full_params.append(free_params[free_idx])
                free_idx += 1
        return func(x, *full_params)
    return wrapped

def double_crystal_ball(x, A, mu, sigma, alphaL, nL, alphaR, nR):
    z = (x - mu) / sigma
    result = np.zeros_like(z)
    abs_alphaL, abs_alphaR = np.abs(alphaL), np.abs(alphaR)
    
    # Core
    mask = (z > -abs_alphaL) & (z < abs_alphaR)
    result[mask] = np.exp(-0.5 * z[mask]**2)
    
    # Left tail
    mask = z <= -abs_alphaL
    NL = (nL/abs_alphaL)**nL * np.exp(-0.5*abs_alphaL**2)
    result[mask] = NL * (nL/abs_alphaL - abs_alphaL - z[mask])**-nL
    
    # Right tail
    mask = z >= abs_alphaR
    NR = (nR/abs_alphaR)**nR * np.exp(-0.5*abs_alphaR**2)
    result[mask] = NR * (nR/abs_alphaR - abs_alphaR + z[mask])**-nR
    
    norm = np.trapezoid(result, x)
    return A * result / norm

def double_voigtian(x, A, mu, sigma1, gamma1, sigma2, gamma2):
    result = (voigt_profile(x-mu, sigma1, gamma1) + 
              voigt_profile(x-mu, sigma2, gamma2))
    return A * result / np.trapezoid(result, x)

def double_gaussian(x, A, mu, sigma):
    gauss = lambda x_val, s: np.exp(-0.5*((x_val-mu)/s)**2)/(s*np.sqrt(2*np.pi))
    return A * (gauss(x, sigma)) / np.trapezoid(gauss(x, sigma), x)


def phase_space(x, B, a, b, x_min=70, x_max=110):
    safe_x = np.clip(x, x_min, x_max)
    result = (safe_x-x_min)**a * (x_max-safe_x)**b
    result[(x<=x_min)|(x>=x_max)] = 0
    return B * result / np.trapezoid(result, x)

def linear(x, B, b, C):
    if b == 0 and C == 0:
        return B * np.ones_like(x) # Modified line
    else:
        return B * (b + C*x) / np.trapezoid(b + C*x, x)

def exponential(x, B, C):
    return B * np.exp(C*x) / np.trapezoid(np.exp(C*x), x)

def chebyshev_background(x, B, *coeffs, x_min=70, x_max=110):
    x_norm = 2*(x-x_min)/(x_max-x_min) - 1
    return B * Chebyshev(coeffs)(x_norm) / np.trapezoid(Chebyshev(coeffs)(x_norm), x)


def create_combined_model(signal_func, background_func):
    signal_param_count = signal_func.__code__.co_argcount - 1 # Exclude 'x'
    
    # Handle background function
    if isinstance(background_func, partial):
        original_func = background_func.func
        # For Chebyshev, we need to pass B plus the coefficients
        if original_func.__name__ == "chebyshev_background":
            num_bg_params_to_pass = 1 + 3 # B plus 3 coefficients
        else:
            num_bg_params_to_pass = original_func.__code__.co_argcount - 1 - len(background_func.keywords)
    else:
        if background_func.__name__ == "chebyshev_background":
            num_bg_params_to_pass = 1 + 3 # B plus 3 coefficients
        else:
            num_bg_params_to_pass = background_func.__code__.co_argcount - 1
    
    return lambda x, *params: (
        signal_func(x, *params[:signal_param_count]) + 
        background_func(x, *params[signal_param_count:signal_param_count + num_bg_params_to_pass]))

# Predefine combined models
MODELS = {
    'dcb': {
        'signal': double_crystal_ball,
        'backgrounds': {
            'ps': partial(phase_space, x_min=70, x_max=110),
            'lin': linear,
            'exp': exponential,
            'cheb': partial(chebyshev_background, x_min=70, x_max=110) # Note the partial here
        }
    },
    'dv': {
        'signal': double_voigtian,
        'backgrounds': {
            'ps': partial(phase_space, x_min=70, x_max=110),
            'lin': linear,
            'exp': exponential,
            'cheb': partial(chebyshev_background, x_min=70, x_max=110) # Note the partial here
        }
    },
    'dg': {
        'signal': double_gaussian,
        'backgrounds': {
            'ps': partial(phase_space, x_min=70, x_max=110),
            'lin': linear,
            'exp': exponential,
            'cheb': partial(chebyshev_background, x_min=70, x_max=110) # Note the partial here
        }
    }
}

for prefix in MODELS:
    for bg in MODELS[prefix]['backgrounds']:
        func_name = f"{prefix}_{bg}"
        bg_func = MODELS[prefix]['backgrounds'][bg]
        
        # Determine number of background parameters needed
        if isinstance(bg_func, partial):
            original_func = bg_func.func
            if original_func.__name__ == "chebyshev_background":
                num_bg_params = 3
            else:
                num_bg_params = original_func.__code__.co_argcount - 1 - len(bg_func.keywords)
        else:
            if bg_func.__name__ == "chebyshev_background":
                num_bg_params = 3
            else:
                num_bg_params = bg_func.__code__.co_argcount - 1
            
        globals()[f"{func_name}_model"] = create_combined_model(
            MODELS[prefix]['signal'], 
            bg_func
        )

def negative_log_likelihood(params_array, model, param_names, fixed_params, centers, values):
    """
    Calculates the negative log-likelihood for a given set of parameters.
    
    params_array : array of free parameters
    model : function(centers, *params) -> expected counts
    param_names : list of all parameter names in order
    fixed_params : dict of fixed parameter values {param_name: value}
    centers : bin centers of the histogram
    values : observed counts in each bin
    """
    # Reconstruct the full parameter array including fixed parameters
    full_params = []
    free_idx = 0
    for i, name in enumerate(param_names):
        if name in fixed_params:
            full_params.append(fixed_params[name])
        else:
            full_params.append(params_array[free_idx])
            free_idx += 1
            
    expected = model(centers, *full_params)
    
    # Ensure expected values are non-negative for Poisson distribution
    expected[expected < 0] = 1e-9 # Small positive value to avoid log(0) or poisson with negative lambda
    
    # Calculate NLL
    nll = -np.sum(poisson.logpmf(values, expected))
    return nll

def compute_errors_from_hessian(nll_func, res_x, param_names, fixed_params):
    """
    Computes parameter errors from the inverse of the Hessian matrix.
    """
    
    # Create a wrapper for nll_func that only takes free parameters
    def free_nll_wrapper(free_params_array):
        return nll_func(free_params_array)

    # Calculate the Hessian matrix
    hessian_func = nd.Hessian(free_nll_wrapper)
    hessian = hessian_func(res_x)
    cov_matrix = np.linalg.inv(hessian)
    errors = np.sqrt(np.diag(cov_matrix))

    full_perr = np.zeros(len(param_names))
    free_idx = 0
    for i, name in enumerate(param_names):
        if name in fixed_params:
            full_perr[i] = 0.0 # Error for fixed parameters is zero
        else:
            full_perr[i] = errors[free_idx]
            free_idx += 1
    return full_perr, cov_matrix # Return covariance matrix as well

def perform_fit(fit_type, hist, hist_name, fixed_params=None):
    fixed_params = fixed_params or {}
    centers = (hist["edges"][:-1] + hist["edges"][1:]) / 2
    values, errors = hist["values"], hist["errors"]
    
    x_min, x_max = 70, 110
    mask = (centers >= x_min) & (centers <= x_max)
    centers_masked, values_masked = centers[mask], values[mask]
    
    # Get model configuration
    config = FIT_CONFIGS[fit_type]
    param_names = config["param_names"]
    
    # Prepare initial guesses (p0) and bounds for free parameters
    initial_guesses = []
    bounds_list = [] # List of tuples for bounds (min, max)
    
    for name in param_names:
        if name not in fixed_params:
            initial_guesses.append(config["bounds"][name][1]) # Use initial guess from config
            bounds_list.append((config["bounds"][name][0], config["bounds"][name][2])) # Use bounds from config

    # Create the model function (might have fixed parameters)
    model_func = globals()[f"{fit_type}_model"]
    
    nll_objective = lambda params_array: negative_log_likelihood(
        params_array, model_func, param_names, fixed_params, centers_masked, values_masked
    )

    # Perform the minimization
    print(f"Starting minimization for {hist_name} with type {fit_type}...")
    res_1 = minimize(
        nll_objective,
        initial_guesses,
        method='L-BFGS-B',
        bounds=bounds_list,
        options={'maxiter': 20000}
    )
    
    print(f"After first fit: {res_1.x}")

    res = minimize(
        nll_objective,
        res_1.x,
        method='nelder-mead',
        bounds=bounds_list,
        options={'maxiter': 20000}
    )

    print(f"After second fit: {res.x}")
    

    popt = res.x
    convergence = "Converged" if res.success else f"Did NOT converge: {res.message}"
    print(f"Minimization status for {hist_name}: {convergence}")
    
    # Reconstruct full popt and compute errors using Hessian
    full_popt = np.zeros(len(param_names))
    free_idx = 0
    for i, name in enumerate(param_names):
        if name in fixed_params:
            full_popt[i] = fixed_params[name]
        else:
            full_popt[i] = popt[free_idx]
            free_idx += 1

    full_perr, full_pcov = compute_errors_from_hessian(nll_objective, popt, param_names, fixed_params)
    
    # Calculate chi-squared for comparison/plotting
    expected_values = model_func(centers_masked, *full_popt)
    errors_masked = errors[mask]
    errors_masked[errors_masked == 0] = 1.0 # Avoid division by zero for chi2 calculation
    
    chi2 = np.sum(((values_masked - expected_values) / errors_masked)**2)
    reduced_chi2 = chi2 / (len(values_masked) - len(popt)) # len(popt) is number of free parameters

    # Extract A and B and their errors
    A_index = param_names.index("A")
    B_index = param_names.index("B")

    A_value = full_popt[A_index]
    A_error = full_perr[A_index]

    B_value = full_popt[B_index]
    B_error = full_perr[B_index]
    
    print(f" - Estimated number of signal events: {A_value:.2f} ± {A_error:.2f}")
    print(f" - Estimated number of background events: {B_value:.2f} ± {B_error:.2f}")
    print(f" - Chi-squared: {chi2:.2f}, Reduced Chi-squared: {reduced_chi2:.2f}")
    print(f" - Status: {convergence}")
    
    return {
        "centers": centers_masked, "values": values_masked, "errors": errors_masked, # Pass back masked data
        "x_min": x_min, "x_max": x_max, "popt": full_popt, "pcov": full_pcov, "full_perr": full_perr,
        "A_value": A_value, "A_error": A_error, "B_value": B_value, "B_error": B_error,
        "chi_squared": chi2, "reduced_chi_squared": reduced_chi2,
        "type": fit_type, "hist_name": hist_name, "convergence": convergence
    }

def plot_fit(results, efficiency=None, eff_error=None, plot_dir=".", data_type="DATA", Npass=None, Nfail=None, convergence=None):
    plt.figure(figsize=(12, 8))
    hep.style.use("CMS")
    
    x = np.linspace(results["x_min"], results["x_max"], 1000)
    prefix, bg_type = results["type"].split('_')[:2]
    
    # Get components
    signal_func = MODELS[prefix]['signal']
    bg_func = MODELS[prefix]['backgrounds'][bg_type]
    
    # Determine the number of parameters for signal and background to correctly slice popt
    signal_param_count = signal_func.__code__.co_argcount - 1
    
    # Handle background function for parameter count
    if isinstance(bg_func, partial):
        original_bg_func = bg_func.func
        if original_bg_func.__name__ == "chebyshev_background":
            num_bg_params_to_pass = 1 + 3
        else:
            num_bg_params_to_pass = original_bg_func.__code__.co_argcount - 1 - len(bg_func.keywords)
    else:
        if bg_func.__name__ == "chebyshev_background":
            num_bg_params_to_pass = 1 + 3
        else:
            num_bg_params_to_pass = bg_func.__code__.co_argcount - 1
            
    signal_params = results["popt"][:signal_param_count]
    bg_params = results["popt"][signal_param_count : signal_param_count + num_bg_params_to_pass]
    
    signal = signal_func(x, *signal_params)
    background = bg_func(x, *bg_params)
    total = signal + background
    
    # Get model values for data points
    model_func = globals()[results["type"] + "_model"]
    model_values = model_func(results["centers"], *results["popt"])
    
    # Calculate parameter errors from pcov (from Hessian)
    param_errors = (results["full_perr"])
    print(f"Parameter errors: {param_errors}")
    
    # Plot main components
    plt.errorbar(results["centers"], results["values"], yerr=results["errors"], 
                 fmt="o", color="royalblue", markersize=6, capsize=3, capthick=1, 
                 label="Data", zorder=10)
    #plt.plot(x, total, color="black", linewidth=2, label="Total fit")
    plt.plot(x, signal, color="red", linestyle="--", linewidth=1.5, label="Signal")
    #plt.plot(x, background, color="green", linestyle="--", linewidth=1.5, label="Background")

    # Formatting
    plt.xlabel("$m_{ee}$ [GeV]", fontsize=12)
    plt.ylabel("Events / GeV", fontsize=12)
    
    # Legend and info box
    hist_name_parts = results['hist_name'].split('_')
    short_bin_id = hist_name_parts[0]
    pass_fail_str = hist_name_parts[1]
    full_pt_bin_name_suffix = BINS_INFO[short_bin_id][0]
    
    # Create parameter info text with errors
    param_text = []
    param_names = FIT_CONFIGS[results["type"]]["param_names"]
    for i, (name, value) in enumerate(zip(param_names, results["popt"])):
        if i < len(param_errors) and np.isclose(param_errors[i], 0.0): # Check if the parameter error is effectively zero
            param_text.append(f"{name} = {value:.3f} (fixed)")
        elif i < len(param_errors):
            error = param_errors[i]
            param_text.append(f"{name} = {value:.3f} ± {error:.3f}")
        else:
            param_text.append(f"{name} = {value:.3f} ± {error:.3f}: (N/A error)")
    
    # Create stats info text
    stats_text = [
        f"χ²/ndf = {results['chi_squared']:.1f}/{len(results['centers'])-len(results['popt'])}",
        f"Red. χ² = {results['reduced_chi_squared']:.2f}",
        f"Efficiency = {efficiency:.4f} ± {eff_error:.4f}\n"
        f"{convergence}"
    ]
    
    # Combine all text
    info_text = ["Fit Parameters:"] + param_text + [""] + stats_text
    
    # Place legend and info box
    plt.legend(loc="upper right", frameon=True, fontsize=10)
    plt.text(0.05, 0.95, "\n".join(info_text), 
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8), 
             fontsize=9, verticalalignment='top',
             fontfamily='monospace')
    
    # Title
    plt.title(f"{data_type.replace('_', ' ')}: {full_pt_bin_name_suffix} ({pass_fail_str})", 
              pad=10, fontsize=20)
    
    # Save plot
    os.makedirs(f"{plot_dir}", exist_ok=True)
    plt.savefig(f"{plot_dir}/{data_type}_{results['type']}_fit_{results['hist_name']}.png", 
                bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Plot saved for {results['hist_name']}\n")

BINS_INFO = {
    f"bin{i:02d}": (f"pt_{lo}p00To{hi}p00", f"{lo:.2f}-{hi:.2f}")
    for i, (lo, hi) in enumerate([
        (5,8), (8,10), (10,15), (15,20), (20,30), (30,35), 
        (35,40), (40,45), (45,50), (50,55), (55,60), 
        (60,80), (80,100), (100,150), (150,250), (250,400)
    ])
}

FIT_CONFIGS = {
    "dcb_ps": {
        "param_names": ["A", "mu", "sigma", "alphaL", "nL", "alphaR", "nR", "B", "a", "b"],
        "bounds": {
            "A": (0, 10000, np.inf),
            "mu": (89, 90, 91),
            "sigma": (1, 2.76, 5),
            "alphaL": (0.1, 1, 10),
            "nL": (0.1, 5, 30),
            "alphaR": (0.1, 1, 10),
            "nR": (0.1, 5, 30),
            "B": (0.000001, 1, np.inf),
            "a": (0.5, 1, 15),
            "b": (0.5, 1, 15),
        },
    },
    "dcb_lin": {
        "param_names": ["A", "mu", "sigma", "alphaL", "nL", "alphaR", "nR", "B", "b", "C"],
        "bounds": {
            "A": (0.1, 800000, np.inf),
            "mu": (89, 90, 91),
            "sigma": (1, 2.76, 4),
            "alphaL": (0.1, 2, 1000000),
            "nL": (0.1, 15, 1000000),
            "alphaR": (0.1, 2, 1000000),
            "nR": (0.1, 5, 1000000),
            "B": (0, 5, np.inf),
            "b": (-1, 0, 1),
            "C": (-2, -0.5, 0.1),
        },
    },
    "dcb_exp": {
        "param_names": ["A", "mu", "sigma", "alphaL", "nL", "alphaR", "nR", "B", "C"],
        "bounds": {
            "A": (10, 100000, np.inf),
            "mu": (89, 90, 91),
            "sigma": (2.75, 2.76, 2.77),
            "alphaL": (0.93, 0.94, 0.95),
            "nL": (0.1, 5, 30),
            "alphaR": (1.78, 1.79, 1.8),
            "nR": (0.1, 5, 30),
            "B": (0.00001, 1, np.inf),
            "C": (0.0001, 0.1, 10),
        },
    },
    "dcb_cheb": {
        "param_names": ["A", "mu", "sigma", "alphaL", "nL", "alphaR", "nR", "B"] + [f"c{i}" for i in range(3)],
        "bounds": {
            "A": (10, 10000, np.inf),
            "mu": (89, 90, 91),
            "sigma": (2, 2.76, 2.77),
            "alphaL": (0.9, 0.94, 0.95),
            "nL": (0.1, 5, 30),
            "alphaR": (1.78, 1.79, 1.9),
            "nR": (0.1, 5, 30),
            "B": (0.000001, 1, np.inf),
            "c0": (-30, 0, 30),
            "c1": (-30, 0, 30),
            "c2": (-30, -5, -2),
        },
    },
    "dv_ps": {
        "param_names": ["A", "mu", "sigma1", "gamma1", "sigma2", "gamma2", "B", "a", "b"],
        "bounds": {
            "A": (10, 10000, np.inf),
            "mu": (89, 90, 91),
            "sigma1": (3.23, 3.24, 3.25),
            "gamma1": (0.01, 0.02, 0.03),
            "sigma2": (2.08, 2.09, 2.10),
            "gamma2": (0.81, 0.82, 0.83),
            "B": (0.000001, 1, np.inf),
            "a": (0.1, 2, 3),
            "b": (0.1, 2, 4),
        },
    },
    "dv_lin": {
        "param_names": ["A", "mu", "sigma1", "gamma1", "sigma2", "gamma2", "B", "b", "C"],
        "bounds": {
            "A": (10, 10000, np.inf),
            "mu": (89, 90, 91),
            "sigma1": (3.23, 3.24, 3.9),
            "gamma1": (0.66, 0.67, 0.68),
            "sigma2": (2.11, 2.12, 2.13),
            "gamma2": (0.90, 0.91, 0.92),
            "B": (0.001, 1, np.inf),
            "b": (-1, 0, 1),
            "C": (-1, -0.5, 0),
        },
    },
    "dv_exp": {
        "param_names": ["A", "mu", "sigma1", "gamma1", "sigma2", "gamma2", "B", "C"],
        "bounds": {
            "A": (10, 10000, np.inf),
            "mu": (89, 90, 92),
            "sigma1": (3.23, 3.24, 3.9),
            "gamma1": (0.01, 0.2, 3),
            "sigma2": (2.11, 2.12, 2.13),
            "gamma2": (0.90, 0.91, 0.92),
            "B": (0.001, 1, np.inf),
            "C": (-10, 0.1, 10),
        },
    },
    "dv_cheb": {
        "param_names": ["A", "mu", "sigma1", "gamma1", "sigma2", "gamma2", "B"] + [f"c{i}" for i in range(3)],
        "bounds": {
            "A": (10, 10000, np.inf),
            "mu": (89, 90, 91),
            "sigma1": (2, 2.5, 3),
            "gamma1": (1, 2, 3),
            "sigma2": (2, 2.5, 3),
            "gamma2": (1, 2, 3),
            "B": (0.000001, 1, np.inf),
            "c0": (0, 1, 10),
            "c1": (-10, 0, 10),
            "c2": (-10, 0, 10),
        },
    },
    "dg_ps": {
        "param_names": ["A", "mu", "sigma", "B", "a", "b"], # Combined sigma for double gaussian
        "bounds": {
            "A": (10, 10000, np.inf),
            "mu": (89, 90, 91),
            "sigma": (0.5, 2, 7),
            "B": (0.000001, 100, np.inf),
            "a": (0.1, 1.44, 2.45),
            "b": (0.1, 2.09, 4),
        },
    },
    "dg_lin": {
        "param_names": ["A", "mu", "sigma", "B", "b", "C"],
        "bounds": {
            "A": (10, 10000, np.inf),
            "mu": (89, 90, 91),
            "sigma": (1, 2.76, 4),
            "B": (0.000001, 1, np.inf),
            "b": (-1, 0, 1),
            "C": (-0.5, 0, 0.5),
        },
    },
    "dg_exp": {
        "param_names": ["A", "mu", "sigma", "B", "C"], # Combined sigma for double gaussian
        "bounds": {
            "A": (10, 10000, np.inf),
            "mu": (89, 90, 91),
            "sigma": (2.75, 2.76, 2.77),
            "B": (0.00001, 1, np.inf),
            "C": (0.0001, 0.1, 10),
        },
    },
    "dg_cheb": {
        "param_names": ["A", "mu", "sigma", "B"] + [f"c{i}" for i in range(3)], # Combined sigma for double gaussian
        "bounds": {
            "A": (10, 10000, np.inf),
            "mu": (89, 90, 91),
            "sigma": (2, 2.76, 6),
            "B": (0.000001, 1, np.inf),
            "c0": (-30, 0, 30),
            "c1": (-30, 0, 30),
            "c2": (-30, -5, -2),
        },
    },
}

def main():
    parser = argparse.ArgumentParser(description="Fit ROOT histograms with different models.")
    parser.add_argument("--bin", required=True, choices=BINS_INFO.keys())
    parser.add_argument("--type", required=True, choices=FIT_CONFIGS.keys())
    parser.add_argument("--data", required=True, 
                        choices=["DATA_barrel_1", "DATA_barrel_2", "MC_barrel_1", "MC_barrel_2"])
    parser.add_argument("--fix", default="", 
                        help="Comma-separated list of parameters to fix in format param1=value1,param2=value2")
    
    args = parser.parse_args()
    
    # File paths
    file_paths = {
        "DATA_barrel_1": "/uscms/home/hortua/nobackup/egamma-tnp/examples/nanoaod_filters_custom/blp2/DATA_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_2_23D_histos_pt_barrel_1.root",
        "DATA_barrel_2": "/uscms/home/hortua/nobackup/egamma-tnp/examples/nanoaod_filters_custom/blp2/DATA_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_2_23D_histos_pt_barrel_2.root",
        "MC_barrel_1": "/uscms/home/hortua/nobackup/egamma-tnp/examples/nanoaod_filters_custom/blp2/MC_DY_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_2_23D_histos_pt_barrel_1.root",
        "MC_barrel_2": "/uscms/home/hortua/nobackup/egamma-tnp/examples/nanoaod_filters_custom/blp2/MC_DY_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_2_23D_histos_pt_barrel_2.root"
    }
    
    # Load data
    try:
        root_file = uproot.open(file_paths[args.data])
    except Exception as e:
        print(f"Error opening file: {e}")
        return
    
    # Prepare directories
    plot_dir = f"{args.bin}_fits/ml_splits/{'DATA' if args.data.startswith('DATA') else 'MC'}"
    os.makedirs(plot_dir, exist_ok=True)
    
    # Load histograms
    bin_suffix, bin_range = BINS_INFO[args.bin]
    hist_pass = load_histogram(root_file, f"{args.bin}_{bin_suffix}_Pass")
    hist_fail = load_histogram(root_file, f"{args.bin}_{bin_suffix}_Fail")
    
    if not hist_pass or not hist_fail:
        print("Error: Could not load pass or fail histograms.")
        root_file.close()
        return
    
    # Parse fixed parameters
    fixed_params = {}
    if args.fix:
        for item in args.fix.split(','):
            try:
                k, v = item.split('=')
                fixed_params[k.strip()] = float(v.strip())
            except ValueError:
                print(f"Warning: Ignoring malformed parameter '{item}'")
    
    # Fit pass histogram
    fit_pass = perform_fit(args.type, hist_pass, f"{args.bin}_Pass", fixed_params)
    fail_fixed = {k:v for k,v in zip(FIT_CONFIGS[args.type]["param_names"], fit_pass["popt"])}
    params_to_float_in_fail = ["A", "B", "nL", "nR"] 
    
    for param in params_to_float_in_fail:
        if param in fail_fixed:
            del fail_fixed[param]
            
    fit_fail = perform_fit(args.type, hist_fail, f"{args.bin}_Fail", fail_fixed)
    
    # Calculate efficiency
    Npass_sig, Npass_bg = fit_pass["A_value"], fit_pass["B_value"]
    Nfail_sig, Nfail_bg = fit_fail["A_value"], fit_fail["B_value"]

    # Efficiency calculation based on signal counts
    if (Npass_sig + Nfail_sig) > 0:
        eff = Npass_sig / (Npass_sig + Nfail_sig)
        eff_err = np.sqrt( (Nfail_sig * fit_pass["A_error"])**2 + (Npass_sig * fit_fail["A_error"])**2 ) / ((Npass_sig + Nfail_sig)**2)
    else:
        eff = 0.0
        eff_err = 0.0

    # Plot results
    plot_fit(fit_pass, eff, eff_err, plot_dir, args.data, Npass_sig, Nfail_sig, fit_pass["convergence"])
    plot_fit(fit_fail, eff, eff_err, plot_dir, args.data, Npass_sig, Nfail_sig, fit_fail["convergence"])
    
    root_file.close()
    print(f"\nEfficiency for pt bin {bin_range} GeV = {eff:.4f} ± {eff_err:.4f}")

if __name__ == "__main__":
    main()
