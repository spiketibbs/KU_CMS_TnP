import argparse
import os
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import uproot
from numpy.polynomial.chebyshev import Chebyshev
from scipy.optimize import curve_fit
from scipy.special import voigt_profile
from collections.abc import Iterable
from scipy.stats import norm
from scipy.interpolate import BPoly
from scipy.optimize import minimize
import numdifftools as nd

x_min, x_max = 65, 115


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
    
    # Use log-exp-sum trick for numerical stability
    log_expected = np.log(np.clip(expected, 1e-10, None))
    nll = -np.sum(values * log_expected - expected)
    
    # Fallback for invalid values
    if np.isnan(nll) or np.isinf(nll):
        return 1e20
    return nll

def compute_errors_from_hessian(nll_func, res_x, param_names, fixed_params, model, centers, values):
    # Wrap the NLL so it only depends on the free parameters
    wrapped_nll = lambda p: nll_func(p, model, param_names, fixed_params, centers, values)
    hessian_func = nd.Hessian(wrapped_nll)

    try:
        hessian = hessian_func(res_x)
        # Regularize Hessian
        hessian += np.eye(hessian.shape[0]) * np.max(np.abs(hessian)) * 1e-8
        cov_matrix = np.linalg.inv(hessian)

    except Exception as e:
        print(f"Error computing Hessian: {e}")
        cov_matrix = np.eye(len(res_x)) * np.nan

    # Compute errors
    diag = np.diag(cov_matrix)
    if np.any(diag < 0):
        print("Warning: Negative variances found in covariance matrix.")
    errors = np.sqrt(np.abs(np.diag(cov_matrix)))
    for name, err in zip(param_names, errors):    
        if err > 1e6:  # or whatever you consider "huge"
            print(f"Warning: Large uncertainty on parameter {name}: ±{err:.2e}")

    # Map errors back to full parameter set
    full_perr = {}
    free_idx = 0
    for name in param_names:
        if name in fixed_params:
            full_perr[name] = 0.0
        else:
            full_perr[name] = errors[free_idx]
            free_idx += 1

    return full_perr, cov_matrix

def double_crystal_ball(x, mu, sigma, alphaL, nL, alphaR, nR):
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
    if norm <= 0:  # Prevent division by zero
        norm = 1e-9
    return result / norm

def double_voigtian(x, mu, sigma1, gamma1, sigma2, gamma2):
    result = (voigt_profile(x-mu, sigma1, gamma1) + 
              voigt_profile(x-mu, sigma2, gamma2))
    return result / np.trapezoid(result, x)

def double_gaussian(x, mu, sigma):
    gauss = lambda x, s: np.exp(-0.5*((x-mu)/s)**2)/(s*np.sqrt(2*np.pi))
    return gauss(x, sigma) / np.trapezoid(gauss(x, sigma), x)

def CB_G(x, mu, sigma, alpha, n, sigma2):
    def crystal_ball_unnormalized(x, mu, sigma, alpha, n):
        z = (x - mu) / sigma
        result = np.zeros_like(z)
        abs_alpha = np.abs(alpha)
        
        # Core region (Gaussian)
        mask_core = (z > -abs_alpha) if (alpha < 0) else (z < abs_alpha)
        result[mask_core] = np.exp(-0.5 * z[mask_core]**2)
        
        # Tail region (Power law)
        mask_tail = (z <= -abs_alpha) if (alpha < 0) else (z >= abs_alpha)
        N = (n / abs_alpha)**n * np.exp(-0.5 * abs_alpha**2)
        
        if alpha < 0:  # Left tail
            result[mask_tail] = N * (n / abs_alpha - abs_alpha - z[mask_tail])**(-n)
        else:  # Right tail
            result[mask_tail] = N * (n / abs_alpha - abs_alpha + z[mask_tail])**(-n)
        return result
    
    y_cb = crystal_ball_unnormalized(x, mu, sigma, alpha, n)
    y_gauss = norm.pdf(x, loc=mu, scale=sigma2)

    y_total = y_cb + y_gauss
    normalization = np.trapezoid(y_total, x)
    return y_total / normalization

def phase_space(x, a, b, x_min=x_min, x_max=x_max):
    safe_x = np.clip(x, x_min, x_max)
    result = (safe_x-x_min)**a * (x_max-safe_x)**b
    result[(x<=x_min)|(x>=x_max)] = 0
    norm = np.trapezoid((safe_x-x_min)**a * (x_max-safe_x)**b, x)
    if norm <= 0:  # Prevent division by zero
        norm = 1e-9
    return result / norm

def linear(x, b, C):
    if b == 0 and C == 0:
        return np.ones_like(x)
    else:
        return (b + C*x) / np.trapezoid(b + C*x, x)

def exponential(x, C):
    return np.exp(C*x) / np.trapezoid(np.exp(C*x), x)

def chebyshev_background(x, *coeffs, x_min=x_min, x_max=x_max):
    x_norm = 2*(x-x_min)/(x_max-x_min) - 1
    return Chebyshev(coeffs)(x_norm) / np.trapezoid(Chebyshev(coeffs)(x_norm), x)

def bernstein_poly(x, *coeffs, x_min = x_min, x_max = x_max):
    c = np.array(coeffs).reshape(-1, 1)
    return BPoly(c, [x_min, x_max])(x)

def cms(x, *params):

    def check_iterable(obj):
        return isinstance(obj, Iterable)

    def err_exp(t):
        return np.exp(-t**2)

    def compute_erfc(x0_vals):
        if isinstance(x0_vals, Iterable):
            return np.array([1 - (2/np.sqrt(np.pi)) * 
                            np.trapezoid(err_exp(np.linspace(0, x0, 100)), np.linspace(0, x0, 100))
                            for x0 in x0_vals])
        else:
            return 1 - (2/np.sqrt(np.pi)) * np.trapezoid(err_exp(np.linspace(0, x0_vals, 100)), 
                                                np.linspace(0, x0_vals, 100))
        
    peak, alpha, beta, gamma = params
    x0 = (alpha - x) * beta
    u = (x - peak) * gamma

    erfc = compute_erfc(x0)
    exp_u = np.where(u < -70, 1e20, np.where(u > 70, 0, np.exp(-u)))

    unnorm_shape = erfc * exp_u
    norm = np.trapezoid(unnorm_shape, x)
    return unnorm_shape / norm

def create_combined_model(fit_type):
    if fit_type not in FIT_CONFIGS:
        raise ValueError(f"Unknown fit type: {fit_type}")
    
    config = FIT_CONFIGS[fit_type]
    signal_func = config["signal_func"]
    bg_func = config["background_func"]
    param_names = config["param_names"]
    
    def combined_model(centers, *params):
        n_pass = len(centers) // 2
        x = centers[:n_pass]   # Pass region centers
        y = centers[n_pass:]   # Fail region centers
        
        params_dict = dict(zip(param_names, params))
        
        # Shared signal parameters
        signal_params = [params_dict[p] for p in SIGNAL_MODELS[fit_type.split('_')[0]]["params"]]
        
        # Background parameters
        bg_pass_params = [params_dict[f"{p}_pass"] for p in BACKGROUND_MODELS[fit_type.split('_')[1]]["params"]]
        bg_fail_params = [params_dict[f"{p}_fail"] for p in BACKGROUND_MODELS[fit_type.split('_')[1]]["params"]]
        
        # Calculate components
        signal_pass = signal_func(x, *signal_params)
        signal_fail = signal_func(y, *signal_params)
        bg_pass = params_dict["B_p"] * bg_func(x, *bg_pass_params)
        bg_fail = params_dict["B_f"] * bg_func(y, *bg_fail_params)
        
        
        # Combine results
        # FORMULA: 
        # 
        # Model = (N[ε * Signal_func_pass + (1 - ε) * Signal_func_fail] + B_pass * Background_func_pass + B_fail * Backgorund_func_fail)
        #
        result_pass = params_dict["N"] * params_dict["epsilon"] * signal_pass + bg_pass
        result_fail = params_dict["N"] * (1 - params_dict["epsilon"]) * signal_fail + bg_fail
        
        return np.concatenate([result_pass, result_fail])
    return combined_model

def fit_function(fit_type, hist_pass, hist_fail, fixed_params=None, x_min=x_min, x_max=x_max):
    fixed_params = fixed_params or {}

    if fit_type not in FIT_CONFIGS:
        raise ValueError(f"Unknown fit type: {fit_type}")

    config = FIT_CONFIGS[fit_type]
    param_names = config["param_names"]

    # Prepare data: centers, values, errors for pass
    centers_pass = (hist_pass["edges"][:-1] + hist_pass["edges"][1:]) / 2
    values_pass = hist_pass["values"]
    values_pass[values_pass <= 0] = 0.000001
    errors_pass = hist_pass.get("errors", np.sqrt(values_pass))
    errors_pass[errors_pass <= 0] = 0.000001


    # Prepare data for fail
    centers_fail = (hist_fail["edges"][:-1] + hist_fail["edges"][1:]) / 2
    values_fail = hist_fail["values"]
    values_fail[values_fail <= 0] = 0.000001
    errors_fail = hist_fail.get("errors", np.sqrt(values_fail))
    errors_fail[errors_fail <= 0] = 0.000001


    # Crop window
    mask_pass = (centers_pass >= x_min) & (centers_pass <= x_max)
    mask_fail = (centers_fail >= x_min) & (centers_fail <= x_max)

    centers_pass = centers_pass[mask_pass]
    values_pass = values_pass[mask_pass]
    errors_pass = errors_pass[mask_pass]

    centers_fail = centers_fail[mask_fail]
    values_fail = values_fail[mask_fail]
    errors_fail = errors_fail[mask_fail]

    # Combine pass and fail data arrays
    all_centers = np.concatenate([centers_pass, centers_fail])
    all_values = np.concatenate([values_pass, values_fail])
    # errors not used in NLL but can be kept if needed elsewhere

    # Create model function from fit_type
    model = create_combined_model(fit_type)

    # Calculate data-based initial guesses
    N_p0 = (np.sum(values_pass) + np.sum(values_fail))
    B_p_p0 = (np.median(values_pass[-10:]) * len(values_pass))
    B_f_p0 = (np.median(values_fail[-10:]) * len(values_fail))

    # Scale fixed parameters if present
    for name in ['N', 'B_p', 'B_f']:
        if name in fixed_params:
            fixed_params[name]

    # Update bounds with data-based values
    bounds = config["bounds"].copy()
    bounds.update({
        "N": (0, N_p0, 2*N_p0),
        "B_p": (0, B_p_p0/2, 2*B_p_p0),
        "B_f": (0, B_f_p0/2, 2*B_f_p0)
    })

    # Prepare initial parameter guesses
    p0 = []
    bounds_low = []
    bounds_high = []
    initial_guesses = {}  # Store ALL initial guesses here

    for name in param_names:
        if name in fixed_params:
            initial_guesses[name] = fixed_params[name]
            continue
            
        # Special cases for data-based parameters
        if name == "N":
            initial_guesses[name] = N_p0
        elif name == "B_p":
            initial_guesses[name] = B_p_p0
        elif name == "B_f":
            initial_guesses[name] = B_f_p0
        elif name == "epsilon":
            initial_guesses[name] = 0.9
        else:
            # Use middle value from bounds
            b = bounds[name]
            initial_guesses[name] = b[1]
            
        # Set bounds and add to p0 for minimization
        b = bounds[name]
        p0.append(initial_guesses[name])
        bounds_low.append(b[0])
        bounds_high.append(b[2])

    nll_objective = lambda params_array: negative_log_likelihood(
        params_array, model, param_names, fixed_params, all_centers, all_values
    )

    # === Step 1: Coarse Optimization with Nelder-Mead ===
    print("\n[Step 1] Running coarse optimization with Nelder-Mead...\n")
    result1 = minimize(
        nll_objective,
        p0,
        method='Nelder-Mead',
        options={
            'maxiter': 20000,
            'xatol': 1e-4,
            'fatol': 1e-4,
            'disp': True
        }
    )

    full_popt1 = {}
    free_idx1 = 0
    for name in param_names:
        if name in fixed_params:
            full_popt1[name] = fixed_params[name]
        else:
            full_popt1[name] = result1.x[free_idx1]
            free_idx1 += 1

    # === Step 2: Fine Optimization with L-BFGS-B starting from Nelder-Mead result ===
    print("\n[Step 2] Refining with L-BFGS-B starting from Nelder-Mead solution...\n")
    result = minimize(
        nll_objective,
        result1.x,
        method='L-BFGS-B',
        bounds=list(zip(bounds_low, bounds_high)),
        options={
            'maxiter': 50000,
            'disp': True,
            'ftol': 1e-8,
            'gtol': 1e-8
        }
    )

    # Extract fit results (rebuild full parameter dict)
    full_popt = {}
    free_idx = 0
    for name in param_names:
        if name in fixed_params:
            full_popt[name] = fixed_params[name]
        else:
            full_popt[name] = result.x[free_idx]
            free_idx += 1

    if result.success:
        full_perr, full_cov = compute_errors_from_hessian(negative_log_likelihood, result.x, param_names, fixed_params, model, all_centers, all_values)
    else:
        full_perr = {name: 0.0 if name in fixed_params else np.nan for name in param_names}
        full_cov = np.eye(len(param_names)) * np.nan

    # FORMAT FOR TERMINAL VIEWING
    print("\nFit Results:")
    print("="*120)
    print(f"{'Parameter':<20} {'Initial Guesses':<20} {'First Run Value':<20} {'Final Value':<20} {'Error':<20} {'Fixed':<10}")
    print("-"*120)

    for name in param_names:
        init_guess = initial_guesses[name]  # Get from our stored dict
        value = full_popt[name]
        value0 = full_popt1[name]
        error = full_perr[name]
        is_fixed = "Yes" if name in fixed_params else "No"
        
        # Format scientific notation for very small/large numbers
        if abs(value) < 1e-3 or abs(value) > 1e5 or abs(value0) < 1e-3 or abs(value0) > 1e5:
            ig_str = f"{init_guess:.4e}"
            val_str = f"{value:.4e}"
            val0_str = f"{value0:.4e}"
            err_str = f"{error:.4e}"
        else:
            ig_str = f"{init_guess:.2f}"
            val_str = f"{value:.6f}"
            val0_str = f"{value0:.6f}"
            err_str = f"{error:.6f}"
            
        print(f"{name:<20} {ig_str:<20} {val0_str:<20} {val_str:<20} {err_str:<20} {is_fixed:<10}")
    print("="*120)

    if result.success == True:
        convergence = "Fit converged!"
    else:
        convergence = "Fit did NOT converge!"

    # Calculate reduced chi2-like metric
    expected = model(all_centers, *[full_popt[name] for name in param_names])
    ndf = len(all_values) - len([p for p in param_names if p not in fixed_params])
    mask = (all_values > 0) & (expected > 0)
    mask = (all_values > 0) & (expected > 0)
    O = all_values[mask]
    E = expected[mask]

    # Clip log to avoid invalid values
    log_term = np.log(O / E)
    log_term = np.where(np.isfinite(log_term), log_term, 0)

    chi2 = 2 * np.sum(E - O + O * log_term)

    ndf = len(O) - len([p for p in param_names if p not in fixed_params])
    reduced_chi2 = chi2 / ndf if ndf > 0 else float("inf")

    contributions = 2 * (E - O + O * np.log(O / E))
    top_idx = np.argsort(contributions)[::-1]
    for i in top_idx[:10]:
        print(f"Bin {i}: O={O[i]:.1f}, E={E[i]:.1f}, contrib={contributions[i]:.2f}")


    print(f"\nCHI² DEBUG:")
    print(f"O[:10] = {O[:10]}")
    print(f"E[:10] = {E[:10]}")
    print(f"chi² = {chi2:.2f}, dof = {ndf}, reduced = {reduced_chi2:.2f}")

    return {
        "popt": full_popt,
        "perr": full_perr,
        "cov": full_cov,
        "bin": fit_type.split('_')[0],
        "chi_squared": chi2,
        "reduced_chi_squared": reduced_chi2,
        "success": result.success,
        "message": result.message,
        "fun": result.fun,
        "param_names": param_names,
        "type": fit_type,
        "centers_pass": centers_pass,
        "values_pass": values_pass,
        "errors_pass": errors_pass,
        "centers_fail": centers_fail,
        "values_fail": values_fail,
        "errors_fail": errors_fail,
        "x_min": x_min,
        "x_max": x_max,
        "convergence": convergence
    }

def plot_combined_fit(results, plot_dir=".", data_type="DATA", fixed_params=None):
    if results is None:
        print("No results to plot")
        return
    
    fixed_params = fixed_params or {}
    fit_type = results["type"]
    config = FIT_CONFIGS[fit_type]
    signal_func = config["signal_func"]
    bg_func = config["background_func"]
    params = results["popt"]
    
    # Get signal and background model names for the legend
    signal_model_name = {
        "dcb": "Double Crystal Ball",
        "dv": "Double Voigtian",
        "dg": "Double Gaussian",
        "cb_g": "Crystal Ball + Gaussian"
    }.get(fit_type.split('_')[0], "Unknown Signal")
    
    background_model_name = {
        "ps": "Phase Space",
        "lin": "Linear",
        "exp": "Exponential",
        "cheb": "Chebyshev Polynomial",
        "cms": "CMS Shape",
        "bpoly": "Bernstein Polynomial"
    }.get(fit_type.split('_')[1], "Unknown Background")
    
    # Create x values for plotting
    x = np.linspace(results["x_min"], results["x_max"], 1000)
    
    # Get SHARED signal parameters
    signal_params = [params[p] for p in SIGNAL_MODELS[fit_type.split('_')[0]]["params"]]
    
    # Helper function to format parameters
    def format_param(name, value, error, fixed_params):
        if name in fixed_params:
            return f"{name} = {fixed_params[name]:.3f} (fixed)"
        elif np.isnan(value):
            return f"{name} = NaN"
        elif np.isinf(value):
            return f"{name} = Infinity"
        elif error == 0:
            return f"{name} = {value:.3f} (fixed)"
        else: 
            return f"{name} = {value:.3f} ± {error:.6f}"

    # Plot PASS components
    plt.figure(figsize=(12, 8))
    hep.style.use("CMS")
    
    # Get background parameters for pass
    bg_pass_params = []
    for p in BACKGROUND_MODELS[fit_type.split('_')[1]]["params"]:
        bg_pass_params.append(params[f"{p}_pass"])
    
    # Calculate components
    signal_pass = params["N"] * params["epsilon"] * signal_func(x, *signal_params)
    bg_pass = params["B_p"] * bg_func(x, *bg_pass_params)
    total_pass = signal_pass + bg_pass
    
    # Plot pass data and fit with updated legend labels
    plt.errorbar(results["centers_pass"], results["values_pass"], yerr=results["errors_pass"], 
                fmt="o", color="royalblue", markersize=6, capsize=3, label="Data (Pass)")
    plt.plot(x, total_pass, 'k-', label="Total fit")
    plt.plot(x, signal_pass, 'r--', label=f"Signal ({signal_model_name})")
    plt.plot(x, bg_pass, 'g--', label=f"Background ({background_model_name})")
    
    # Formatting
    plt.xlabel("$m_{ee}$ [GeV]", fontsize=12)
    plt.ylabel("Events / GeV", fontsize=12)
    plt.title(f"{data_type.replace('_', ' ')}: {BINS_INFO[results['bin']][1]} GeV (Pass)", pad=10)
    
    # Add fit info
    dof = len(results["centers_pass"]) + len(results["centers_fail"]) - len(results["popt"])
    chi2_red = results["chi_squared"] / dof

    
    # For PASS plot:
    signal_params_text = "\n".join([
        format_param(p, params[p], results["perr"][p], fixed_params)
        for p in SIGNAL_MODELS[fit_type.split('_')[0]]["params"]
    ])
    bg_params_text = "\n".join([
        format_param(f"{p}_pass", params[f"{p}_pass"], results["perr"][f"{p}_pass"], fixed_params)
        for p in BACKGROUND_MODELS[fit_type.split('_')[1]]["params"]
    ])

    info_text = [
        f"N = {params['N']:.1f} ± {results['perr']['N']:.1f}",
        f"ε = {params['epsilon']:.6f} ± {results['perr']['epsilon']:.6f}",
        f"B_p = {params['B_p']:.1f} ± {results['perr']['B_p']:.1f}",
        f"B_f = {params['B_f']:.1f} ± {results['perr']['B_f']:.1f}",
        f"Convergence: {results['convergence']}",
        f""
        f"Signal yield: {params['N']*params['epsilon']:.1f}",
        f"Bkg yield: {params['B_p']:.1f}",
        f"χ²/ndf = {results['chi_squared']:.1f}/{dof} = {chi2_red:.2f}",
        "",
        "Signal params:",
        signal_params_text,
        "",
        "Background params:",
        bg_params_text
    ]

    plt.legend(loc="upper right", fontsize=10)
    plt.gca().text(
        0.02, 0.98,
        "\n".join(info_text),
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(facecolor='white', edgecolor='black', alpha=0.8)
    )
    
    # Save pass plot
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(f"{plot_dir}/{data_type}_{results['type']}_fit_{results['bin']}_Pass.png", 
               bbox_inches="tight", dpi=300)
    plt.close()
    
    # Plot FAIL components
    plt.figure(figsize=(12, 8))
    hep.style.use("CMS")
    
    # Get background parameters for fail
    bg_fail_params = []
    for p in BACKGROUND_MODELS[fit_type.split('_')[1]]["params"]:
        bg_fail_params.append(params[f"{p}_fail"])
    
    # Calculate components
    signal_fail = signal_func(x, *signal_params)
    bg_fail = params["B_f"] * bg_func(x, *bg_fail_params)
    total_fail = params["N"] * (1-params["epsilon"]) * signal_fail + bg_fail
    
    # Plot fail data and fit
    plt.errorbar(results["centers_fail"], results["values_fail"], yerr=results["errors_fail"], 
                fmt="o", color="royalblue", markersize=6, capsize=3, label="Data (Fail)")
    plt.plot(x, total_fail, 'k-', label="Total fit")
    plt.plot(x, params["N"]*(1-params["epsilon"])*signal_fail, 'r--', label="Signal")
    plt.plot(x, bg_fail, 'g--', label="Background")
    
    # Formatting
    plt.xlabel("$m_{ee}$ [GeV]", fontsize=12)
    plt.ylabel("Events / GeV", fontsize=12)
    plt.title(f"{data_type.replace('_', ' ')}: {BINS_INFO[results['bin']][1]} GeV (Fail)", pad=10)
    
    # For FAIL plot:
    bg_params_text_fail = "\n".join([
        format_param(f"{p}_fail", params[f"{p}_fail"], results["perr"][f"{p}_fail"], fixed_params)
        for p in BACKGROUND_MODELS[fit_type.split('_')[1]]["params"]
    ])

    info_text = [
        f"N = {params['N']:.1f} ± {results['perr']['N']:.1f}",
        f"ε = {params['epsilon']:.6f} ± {results['perr']['epsilon']:.6f}",
        f"B_p = {params['B_p']:.1f} ± {results['perr']['B_p']:.1f}",
        f"B_f = {params['B_f']:.1f} ± {results['perr']['B_f']:.1f}",
        f"Convergence: {results['convergence']}",
        f""
        f"Signal yield: {params['N']*(1-params['epsilon']):.1f}",
        f"Bkg yield: {params['B_f']:.1f}",
        f"χ²/ndf = {results['chi_squared']:.1f}/{dof} = {chi2_red:.2f}",
        "",
        "Signal params:",
        signal_params_text,
        "",
        "Background params:",
        bg_params_text_fail
    ]
    
    plt.legend(loc="upper right", fontsize=10)
    plt.gca().text(
        0.02, 0.98,
        "\n".join(info_text),
        transform=plt.gca().transAxes,
        fontsize=9,
        verticalalignment='top',
        horizontalalignment='left',
        bbox=dict(facecolor='white', edgecolor='black', alpha=0.8)
    )
    
    # Save fail plot
    plt.savefig(f"{plot_dir}/{data_type}_{results['type']}_fit_{results['bin']}_Fail.png", 
               bbox_inches="tight", dpi=300)
    print(f"Plots saved to {plot_dir}")
    plt.close()

BINS_INFO = {
    f"bin{i}": (f"pt_{lo}p00To{hi}p00", f"{lo:.2f}-{hi:.2f}")
    for i, (lo, hi) in enumerate([
        (5,7), (7,10), (10,20), (20,45), (45,75), (75,100), (100,500)
    ])
}

SIGNAL_MODELS = {
    "dcb": {
        "func": double_crystal_ball,
        "params": ["mu", "sigma", "alphaL", "nL", "alphaR", "nR"],
        "bounds": {
            "mu": (89, 90, 93),
            "sigma": (0.1, 4, 10),
            "alphaL": (0, 1.0, np.inf),
            "nL": (0, 5.0, np.inf),
            "alphaR": (0, 1.0, np.inf),
            "nR": (0, 5.0, np.inf)
        }
    },
    "dv": {
        "func": double_voigtian,
        "params": ["mu", "sigma1", "gamma1", "sigma2", "gamma2"],
        "bounds": {
            "mu": (88, 90, 93),
            "sigma1": (2.0, 3.0, 4.0),
            "gamma1": (0.01, 0.5, 3.0),
            "sigma2": (1.0, 2.0, 3.0),
            "gamma2": (0.5, 1.0, 3.0)
        }
    },
    "dg": {
        "func": double_gaussian,
        "params": ["mu", "sigma"],
        "bounds": {
            "mu": (89, 90, 91),
            "sigma": (1.0, 2.5, 4.0)
        }
    },
    "cbg": {
        "func": CB_G,
        "params": ["mu", "sigma", "alpha", "n", "sigma2"],
        "bounds": {
            "mu": (88, 90, 93),
            "sigma": (1, 2.5, 5),
            "alpha": (-10, -1, 10),
            "n": (0.1, 5.0, 100),
            "sigma2": (1, 3, 10)
        }
    }
}

# Then define all background models
BACKGROUND_MODELS = {
    "ps": {
        "func": lambda x, a, b: phase_space(x, a, b, x_min=x_min, x_max=x_max),  # Wrap with lambda
        "params": ["a", "b"],
        "bounds": {
            "a": (0, 2, np.inf),
            "b": (0, 1, np.inf)
        }
    },
    "lin": {
        "func": linear,
        "params": ["b", "C"],
        "bounds": {
            "b": (-1, 0, 1),
            "C": (-1, -0.01, 0)
        }
    },
    "exp": {
        "func": exponential,
        "params": ["C"],
        "bounds": {
            "C": (-10, -0.1, 10)
        }
    },
    "cheb": {
        "func": chebyshev_background,
        "params": ["c0", "c1", "c2"],
        "bounds": {
            "c0": (-30, 0, 30),
            "c1": (-30, 0, 30),
            "c2": (-30, -5, -2)
        }
    },
    "bpoly": {
        "func": bernstein_poly,
        "params": ["c0", "c1", "c2", "c3"],
        "bounds": {
            "c0": (-3, 0, 3),
            "c1": (-3, 0, 3),
            "c2": (-3, 0, 3),
            "c3": (-3, 0, 3)
        }
    },
    "cms": {
        "func": cms,
        "params": ["peak", "alpha", "beta", "gamma"],
        "bounds": {
            "peak": (80, 90, 100),
            "alpha": (75, 100, 125),
            "beta": (0.001, 0.1, 1),
            "gamma": (0.001, 0.1, 1)
        }
    }

}

# Now combine them into all possible combinations
FIT_CONFIGS = {}
for sig_name, sig_config in SIGNAL_MODELS.items():
    for bg_name, bg_config in BACKGROUND_MODELS.items():
        fit_type = f"{sig_name}_{bg_name}"
        
        # Build parameter names list
        param_names = ["N", "epsilon", "B_p", "B_f"]
        
        # Add SHARED signal parameters (not pass/fail)
        param_names.extend(sig_config["params"])
        
        # Add pass/fail versions of background parameters
        for p in bg_config["params"]:
            param_names.extend([f"{p}_pass", f"{p}_fail"])
        
        # Build bounds dictionary
        bounds = {
            "N": (0, 100000, np.inf),
            "epsilon": (0.75, 0.9, 0.9999),
            "B_p": (0, 10000, np.inf),
            "B_f": (0, 10000, np.inf)
        }
        
        # Add signal bounds (shared)
        for p, b in sig_config["bounds"].items():
            bounds[p] = b
        
        # Add background bounds (pass/fail)
        for p, b in bg_config["bounds"].items():
            bounds[f"{p}_pass"] = b
            bounds[f"{p}_fail"] = b
        
        FIT_CONFIGS[fit_type] = {
            "param_names": param_names,
            "bounds": bounds,
            "signal_func": sig_config["func"],
            "background_func": bg_config["func"]
        }

def main():
    parser = argparse.ArgumentParser(description="Fit ROOT histograms with different models.")
    parser.add_argument("--bin", required=True, choices=BINS_INFO.keys())
    parser.add_argument("--type", required=True, choices=FIT_CONFIGS.keys())
    parser.add_argument("--data", required=True, 
                       choices=["DATA_barrel_1_tag",         "DATA_barrel_1",         "DATA_barrel_2_tag",         "DATA_barrel_2", 
                                "DATA_NEW_barrel_1_tag",     "DATA_NEW_barrel_1",     "DATA_NEW_barrel_2_tag",     "DATA_NEW_barrel_2",
                                "MC_DY_barrel_1_tag",        "MC_DY_barrel_1",        "MC_DY_barrel_2_tag",        "MC_DY_barrel_2",
                                "MC_DY2_2L_2J_barrel_1_tag", "MC_DY2_2L_2J_barrel_1", "MC_DY2_2L_2J_barrel_2_tag", "MC_DY2_2L_2J_barrel_2",
                                "MC_DY2_2L_4J_barrel_1_tag", "MC_DY2_2L_4J_barrel_1", "MC_DY2_2L_4J_barrel_2_tag", "MC_DY2_2L_4J_barrel_2"])
    parser.add_argument("--fix", default="", 
                       help="Comma-separated list of parameters to fix in format param1=value1,param2=value2")
    
    args = parser.parse_args()
    
# NEW FILE PATH
    file_paths = {
        "DATA_barrel_1_tag": "/uscms/home/hortua/nobackup/egamma-tnp/examples/nanoaod_filters_custom/blp_3gt/DATA_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_23D_histos_pt_barrel_1.root",
        "DATA_barrel_1": "/uscms/home/hortua/nobackup/egamma-tnp/examples/nanoaod_filters_custom/blp_3/DATA_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_23D_histos_pt_barrel_1.root",

        "DATA_barrel_2_tag": "/uscms/home/hortua/nobackup/egamma-tnp/examples/nanoaod_filters_custom/blp_3gt/DATA_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_23D_histos_pt_barrel_2.root",
        "DATA_barrel_2": "/uscms/home/hortua/nobackup/egamma-tnp/examples/nanoaod_filters_custom/blp_3/DATA_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_23D_histos_pt_barrel_2.root",

        "DATA_endcap_tag": "/uscms/home/hortua/nobackup/egamma-tnp/examples/nanoaod_filters_custom/blp_3gt/DATA_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_23D_histos_pt_endcap.root",
        "DATA_endcap": "/uscms/home/hortua/nobackup/egamma-tnp/examples/nanoaod_filters_custom/blp_3/DATA_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_23D_histos_pt_endcap.root",

        "DATA_NEW_barrel_1_tag": "/uscms/home/hortua/nobackup/egamma-tnp/examples/nanoaod_filters_custom/blp_3gt/DATA_NEW_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_NEW_23D_histos_pt_barrel_1.root",
        "DATA_NEW_barrel_1": "/uscms/home/hortua/nobackup/egamma-tnp/examples/nanoaod_filters_custom/blp_3/DATA_NEW_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_NEW_23D_histos_pt_barrel_1.root",

        "DATA_NEW_barrel_2_tag": "/uscms/home/hortua/nobackup/egamma-tnp/examples/nanoaod_filters_custom/blp_3gt/DATA_NEW_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_NEW_23D_histos_pt_barrel_2.root",
        "DATA_NEW_barrel_2": "/uscms/home/hortua/nobackup/egamma-tnp/examples/nanoaod_filters_custom/blp_3/DATA_NEW_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_NEW_23D_histos_pt_barrel_2.root",
        
        "DATA_NEW_endcap_tag": "/uscms/home/hortua/nobackup/egamma-tnp/examples/nanoaod_filters_custom/blp_3gt/DATA_NEW_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_NEW_23D_histos_pt_endcap.root",
        "DATA_NEW_endcap": "/uscms/home/hortua/nobackup/egamma-tnp/examples/nanoaod_filters_custom/blp_3/DATA_NEW_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_NEW_23D_histos_pt_endcap.root",

        "MC_DY_barrel_1_tag": "/uscms/home/hortua/nobackup/egamma-tnp/examples/nanoaod_filters_custom/blp_3gt/MC_DY_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY_23D_histos_pt_barrel_1.root",
        "MC_DY_barrel_1": "/uscms/home/hortua/nobackup/egamma-tnp/examples/nanoaod_filters_custom/blp_3/MC_DY_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY_23D_histos_pt_barrel_1.root",

        "MC_DY_barrel_2_tag": "/uscms/home/hortua/nobackup/egamma-tnp/examples/nanoaod_filters_custom/blp_3gt/MC_DY_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY_23D_histos_pt_barrel_2.root",
        "MC_DY_barrel_2": "/uscms/home/hortua/nobackup/egamma-tnp/examples/nanoaod_filters_custom/blp_3/MC_DY_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY_23D_histos_pt_barrel_2.root",

        "MC_DY_endcap_tag": "/uscms/home/hortua/nobackup/egamma-tnp/examples/nanoaod_filters_custom/blp_3gt/MC_DY_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY_23D_histos_pt_endcap.root",
        "MC_DY_endcap": "/uscms/home/hortua/nobackup/egamma-tnp/examples/nanoaod_filters_custom/blp_3/MC_DY_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY_23D_histos_pt_endcap.root",

        "MC_DY2_2L_2J_barrel_1_tag": "/uscms/home/hortua/nobackup/egamma-tnp/examples/nanoaod_filters_custom/blp_3gt/MC_DY2_2L_2J_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY2_2L_2J_23D_histos_pt_barrel_1.root",
        "MC_DY2_2L_2J_barrel_1": "/uscms/home/hortua/nobackup/egamma-tnp/examples/nanoaod_filters_custom/blp_3/MC_DY2_2L_2J_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY2_2L_2J_23D_histos_pt_barrel_1.root",

        "MC_DY2_2L_2J_barrel_2_tag": "/uscms/home/hortua/nobackup/egamma-tnp/examples/nanoaod_filters_custom/blp_3gt/MC_DY2_2L_2J_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY2_2L_2J_23D_histos_pt_barrel_2.root",
        "MC_DY2_2L_2J_barrel_2": "/uscms/home/hortua/nobackup/egamma-tnp/examples/nanoaod_filters_custom/blp_3/MC_DY2_2L_2J_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY2_2L_2J_23D_histos_pt_barrel_2.root",

        "MC_DY2_2L_2J_endcap_tag": "/uscms/home/hortua/nobackup/egamma-tnp/examples/nanoaod_filters_custom/blp_3gt/MC_DY2_2L_2J_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY2_2L_2J_23D_histos_pt_endcap.root",
        "MC_DY2_2L_2J_endcap": "/uscms/home/hortua/nobackup/egamma-tnp/examples/nanoaod_filters_custom/blp_3/MC_DY2_2L_2J_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY2_2L_2J_23D_histos_pt_endcap.root",

        "MC_DY2_2L_4J_barrel_1_tag": "/uscms/home/hortua/nobackup/egamma-tnp/examples/nanoaod_filters_custom/blp_3gt/MC_DY2_2L_4J_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY2_2L_4J_23D_histos_pt_barrel_1.root",
        "MC_DY2_2L_4J_barrel_1": "/uscms/home/hortua/nobackup/egamma-tnp/examples/nanoaod_filters_custom/blp_3/MC_DY2_2L_4J_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY2_2L_4J_23D_histos_pt_barrel_1.root",

        "MC_DY2_2L_4J_barrel_2_tag": "/uscms/home/hortua/nobackup/egamma-tnp/examples/nanoaod_filters_custom/blp_3gt/MC_DY2_2L_4J_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY2_2L_4J_23D_histos_pt_barrel_2.root",
        "MC_DY2_2L_4J_barrel_2": "/uscms/home/hortua/nobackup/egamma-tnp/examples/nanoaod_filters_custom/blp_3/MC_DY2_2L_4J_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY2_2L_4J_23D_histos_pt_barrel_2.root",

        "MC_DY2_2L_4J_endcap_tag": "/uscms/home/hortua/nobackup/egamma-tnp/examples/nanoaod_filters_custom/blp_3gt/MC_DY2_2L_4J_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY2_2L_4J_23D_histos_pt_endcap.root",
        "MC_DY2_2L_4J_endcap": "/uscms/home/hortua/nobackup/egamma-tnp/examples/nanoaod_filters_custom/blp_3/MC_DY2_2L_4J_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY2_2L_4J_23D_histos_pt_endcap.root"

    }

    # Load data
    try:
        root_file = uproot.open(file_paths[args.data])
    except Exception as e:
        print(f"Error opening file: {e}")
        return

    # Prepare directories
    plot_dir = f"{args.bin}_fits/{'DATA' if args.data.startswith('DATA') else 'MC'}"
    os.makedirs(plot_dir, exist_ok=True)
    
    # Load histograms
    bin_suffix, bin_range = BINS_INFO[args.bin]
    hist_pass = load_histogram(root_file, f"{args.bin}_{bin_suffix}_Pass")
    hist_fail = load_histogram(root_file, f"{args.bin}_{bin_suffix}_Fail") 

    print(f"Looking for histograms:")
    print(f"Pass: {args.bin}_{bin_suffix}_Pass")
    print(f"Fail: {args.bin}_{bin_suffix}_Fail")

    if not hist_pass or not hist_fail:
        root_file.close()
        return

    # Parse fixed parameters
    fixed_params = {}
    if args.fix:
        for item in args.fix.split(','):
            try:
                k, v = item.split('=')
                fixed_params[k.strip()] = float(v.strip())
            except:
                print(f"Warning: Ignoring malformed parameter '{item}'")
    
    # Perform combined fit
    results = fit_function(args.type, hist_pass, hist_fail, fixed_params)
    if results is None:
        print("Fit failed, no results to plot")
        root_file.close()
        return
    
    results["bin"] = args.bin  # Add bin info for plotting
    
    # Plot results
    plot_combined_fit(results, plot_dir, args.data)
    
    root_file.close()
    print(f"\nEfficiency for pt bin {bin_range} GeV = {results['popt']['epsilon']:.4f} ± {results['perr']['epsilon']:.4f}")

if __name__ == "__main__":
    main()