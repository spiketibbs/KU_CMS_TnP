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
from iminuit import cost, Minuit


x_min, x_max = 70, 110


def load_histogram(root_file, hist_name):
    keys = {key.split(";")[0]: key for key in root_file.keys()}
    if hist_name in keys:
        obj = root_file[keys[hist_name]]
        if isinstance(obj, uproot.behaviors.TH1.Histogram):
            values, edges = obj.to_numpy()
            print(f"Histogram: {hist_name}")
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

def double_crystal_ball(x, mu, sigma, alphaL, nL, alphaR, nR):
    nL = np.clip(nL, 1, 50)
    nR = np.clip(nR, 1, 50)

    z = (x - mu) / sigma
    result = np.zeros_like(z)

    # avoid division by zero
    abs_aL = max(np.abs(alphaL), 1e-8)
    abs_aR = max(np.abs(alphaR), 1e-8)

    # core
    core = np.exp(-0.5 * z**2)
    mask_core = (z > -abs_aL) & (z < abs_aR)
    result[mask_core] = core[mask_core]

    # left tail
    mask_L = z <= -abs_aL
    # log of normalization constant
    logNL = nL * np.log(nL/abs_aL) - 0.5 * abs_aL**2
    tL = (nL/abs_aL - abs_aL - z[mask_L])
    tL = np.maximum(tL, 1e-8)
    result[mask_L] = np.exp(logNL - nL * np.log(tL))

    # right tail
    mask_R = z >= abs_aR
    logNR = nR * np.log(nR/abs_aR) - 0.5 * abs_aR**2
    tR = (nR/abs_aR - abs_aR + z[mask_R])
    tR = np.maximum(tR, 1e-8)
    result[mask_R] = np.exp(logNR - nR * np.log(tR))

    # final normalization
    norm = np.trapezoid(result, x)
    if norm <= 0 or not np.isfinite(norm):
        norm = 1e-8
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
    # 1) Clip exponents into a safe range
    a_clamped = np.clip(a, 0, 20)
    b_clamped = np.clip(b, 0, 20)

    # 2) Work in log‐space
    #    t1 = x - x_min, t2 = x_max - x
    t1 = np.clip(x - x_min, 1e-8, None)
    t2 = np.clip(x_max - x, 1e-8, None)

    log_pdf = a_clamped * np.log(t1) + b_clamped * np.log(t2)
    pdf = np.exp(log_pdf - np.max(log_pdf))   # subtract max for stability

    # zero outside
    pdf[(x <= x_min) | (x >= x_max)] = 0

    # 3) Normalize numerically
    norm = np.trapezoid(pdf, x)
    return pdf / (norm if norm>0 else 1e-8)

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

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def create_combined_model_integral(fit_type, n_pass_edges):
    if fit_type not in FIT_CONFIGS:
        raise ValueError(f"Unknown fit type: {fit_type}")
    
    config = FIT_CONFIGS[fit_type]
    signal_func = config["signal_func"]
    bg_func = config["background_func"]
    param_names = config["param_names"]
    
    def model(xe, *all_pars):
        P = dict(zip(param_names, all_pars))

        # split raw array
        edges_pass    = xe[:n_pass_edges]
        edges_fail = xe[n_pass_edges:]      

        signal_params = [P[p] for p in SIGNAL_MODELS[fit_type.split('_')[0]]["params"]]

        bg_pass_params = [P[f"{p}_pass"] for p in BACKGROUND_MODELS[fit_type.split('_')[1]]["params"]]
        bg_fail_params = [P[f"{p}_fail"] for p in BACKGROUND_MODELS[fit_type.split('_')[1]]["params"]]

        # build integrals
        signal_pass = signal_func(edges_pass, *signal_params)
        signal_fail = signal_func(edges_fail, *signal_params)
        bg_pass = P["B_p"] * bg_func(edges_pass, *bg_pass_params)
        bg_fail = P["B_f"] * bg_func(edges_fail, *bg_fail_params)

        N   = P["N"]
        ε   = P["epsilon"]
        Bp  = P["B_p"]
        Bf  = P["B_f"]

        model_pass = N * ε * signal_pass + bg_pass
        model_fail = N * (1 - ε) * signal_fail + bg_fail

        return np.concatenate([model_pass, model_fail])

    return model

def print_minuit_params_table(minuit_obj):
    # Header
    header = f"{'idx':>3} | {'name':^10} | {'value':^12} | {'error':^12} | {'fixed':^5} | {'lower':^10} | {'upper':^10}"
    sep    = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    # Rows
    for i, p in enumerate(minuit_obj.params):
        low  = p.lower_limit if p.lower_limit is not None else ""
        high = p.upper_limit if p.upper_limit is not None else ""
        print(f"{i:3d} | {p.name:10s} | {p.value:12.6f} | {p.error:12.6f} | "
              f"{str(p.is_fixed):^5s} | {str(low):10s} | {str(high):10s}")
    print(sep)

def print_fit_progress(m):
    print(f"Iteration: N={m.values['N']:.1f}, ε={m.values['epsilon']:.3f}, "
          f"B_p={m.values['B_p']:.1f}, B_f={m.values['B_f']:.1f}, "
          f"fval={m.fval:.1f}")

def fit_function(fit_type, hist_pass, hist_fail, fixed_params=None, x_min=x_min, x_max=x_max):
    fixed_params = fixed_params or {}

    if fit_type not in FIT_CONFIGS:
        raise ValueError(f"Unknown fit type: {fit_type}")

    config = FIT_CONFIGS[fit_type]
    param_names = config["param_names"]

    # Prepare data
    edges_pass = hist_pass["edges"]
    values_pass = hist_pass["values"]
    values_pass[values_pass <= 0] = 0.000001
    errors_pass = hist_pass.get("errors", np.sqrt(values_pass))
    errors_pass[errors_pass <= 0] = 0.000001

    edges_fail = hist_fail["edges"]
    values_fail = hist_fail["values"]
    values_fail[values_fail <= 0] = 0.000001
    errors_fail = hist_fail.get("errors", np.sqrt(values_fail))
    errors_fail[errors_fail <= 0] = 0.000001

    bin_mask_pass = (edges_pass[:-1] >= x_min) & (edges_pass[1:] <= x_max)
    edges_pass = edges_pass[np.r_[bin_mask_pass, False] | np.r_[False, bin_mask_pass]]
    centers_pass = 0.5 * (edges_pass[:-1] + edges_pass[1:])
    values_pass = values_pass[bin_mask_pass]
    errors_pass = errors_pass[bin_mask_pass]

    bin_mask_fail = (edges_fail[:-1] >= x_min) & (edges_fail[1:] <= x_max)
    edges_fail = edges_fail[np.r_[bin_mask_fail, False] | np.r_[False, bin_mask_fail]]
    centers_fail = 0.5 * (edges_fail[:-1] + edges_fail[1:])
    values_fail = values_fail[bin_mask_fail]
    errors_fail = errors_fail[bin_mask_fail]

    # Calculate data-based initial guesses
    N_p0 = (np.sum(values_pass) + np.sum(values_fail)) / 2
    B_p_p0 = np.median(values_pass[:5] + values_pass[-5:]) * len(values_pass)/10
    B_f_p0 = np.median(values_fail[:5] + values_fail[-5:]) * len(values_fail)/10
    total_pass = np.sum(values_pass)
    total_fail = np.sum(values_fail)
    initial_epsilon = total_pass / (total_pass + total_fail)

    # Scale fixed parameters if present
    for name in ['N', 'epsilon', 'B_p', 'B_f']:
        if name in fixed_params:
            fixed_params[name]

    # Update bounds with data-based values
    bounds = config["bounds"].copy()
    bounds.update({
        "N":   (0, N_p0*0.7, N_p0*5),
        "B_p": (0, B_p_p0/2, np.inf),
        "B_f": (0, B_f_p0/2, np.inf),
        "epsilon": (0, initial_epsilon, 1)
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
            initial_guesses[name] = N_p0*0.7
        elif name == "B_p":
            initial_guesses[name] = B_p_p0
        elif name == "B_f":
            initial_guesses[name] = B_f_p0
        elif name == "epsilon":
            initial_guesses[name] = 0.95
        else:
            # Use middle value from bounds
            b = bounds[name]
            initial_guesses[name] = b[1]
            
        # Set bounds and add to p0 for minimization
        b = bounds[name]
        p0.append(initial_guesses[name])
        bounds_low.append(b[0])
        bounds_high.append(b[2])

    # Combine edges and values for the fit
    combined_edges = np.concatenate([edges_pass, edges_fail[1:]])
    combined_values = np.concatenate([values_pass, values_fail])

    model_integral = create_combined_model_integral(fit_type, len(edges_pass))
        
    # Initialize cost function
    c = cost.ExtendedBinnedNLL(combined_values, combined_edges, model_integral, use_pdf = 'approximate')
    c.errdef = Minuit.LIKELIHOOD

    # Prepare initial values using the initial_guesses we calculated
    init = initial_guesses

    # Initialize Minuit with positional arguments and parameter names
    m = Minuit(c, *[init[name] for name in param_names], name=param_names)
    
    # Set limits and fixed parameters
    for name in param_names:
        if name in fixed_params:
            m.fixed[name] = True
        elif name in bounds:  # Now using our updated bounds
            m.limits[name] = bounds[name][0], bounds[name][2]
    m.strategy = 2
    print(f"Initial Epsilon: {m.values['epsilon']:.3f}")

    m.interactive()

    m.simplex()
    m.migrad()
    print(f"M2 Epsilon: {m.values['epsilon']:.3f}")

    m.print_level = 2
    m.hesse()
    print_minuit_params_table(m)
    print_fit_progress(m)

    print(f"𝜒²/ndof = {m.fval:.2f} / {m.ndof} = {m.fmin.reduced_chi2:.5f}")
    print(f"Fit valid: {m.valid}")

    popt = m.values.to_dict()
    perr = m.errors.to_dict()
    cov = m.covariance
    print(f"Covariance matrix:\n{cov}")

    chi2 = m.fval
    dof = m.ndof
    reduced_chi2 = m.fmin.reduced_chi2

    results = {
        "type": fit_type,
        "popt": popt,
        "perr": perr,
        "cov": cov,
        "chi_squared": chi2,
        "reduced_chi_squared": reduced_chi2,
        "dof": dof,
        "success": m.valid,
        "message": m.fmin,
        "param_names": param_names,
        "centers_pass": centers_pass,
        "edges_pass": edges_pass,
        "values_pass": values_pass,
        "errors_pass": errors_pass,
        "centers_fail": centers_fail,
        "edges_fail": edges_fail,
        "values_fail": values_fail,
        "errors_fail": errors_fail,
        "x_min": x_min,
        "x_max": x_max,
    }
    return results

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
    chi2_red = results["reduced_chi_squared"]

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
        f""
        f"Signal yield: {params['N']*params['epsilon']:.1f}",
        f"Bkg yield: {params['B_p']:.1f}",
        f"χ²/ndf = {results['chi_squared']:.1f}/{results['dof']} = {chi2_red:.2f}",
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
        f""
        f"Signal yield: {params['N']*(1-params['epsilon']):.1f}",
        f"Bkg yield: {params['B_f']:.1f}",
        f"χ²/ndf = {results['chi_squared']:.1f}/{results['dof']} = {chi2_red:.2f}",
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
            "mu": (87, 89, 92),
            "sigma": (0, 2, 6),
            "alphaL": (0, 2.0, 50),
            "nL": (0, 10.0, 100),
            "alphaR": (0, 1.0, 50),
            "nR": (0, 5.0, 100)
        }
    },
    "dv": {
        "func": double_voigtian,
        "params": ["mu", "sigma1", "gamma1", "sigma2", "gamma2"],
        "bounds": {
            "mu": (87, 89, 92),
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
            "mu": (88, 90, 93),
            "sigma": (1.0, 2.5, 6)
        }
    },
    "cbg": {
        "func": CB_G,
        "params": ["mu", "sigma", "alpha", "n", "sigma2"],
        "bounds": {
            "mu": (87, 89, 92),
            "sigma": (0, 2, 5),
            "alpha": (-10, -1, 10),
            "n": (0, 5.0, 100),
            "sigma2": (0, 2, 10)
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
            "b": (-2, 0, 2),
            "C": (-1, -0.1, 1)
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
            "N": (6000, 7000, np.inf),
            "epsilon": (0.8, 0.95, 1),
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
