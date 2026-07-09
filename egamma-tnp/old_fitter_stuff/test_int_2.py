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

def print_minuit_params_table(minuit_obj):
    # Header
    header = f"{'idx':>3} | {'name':^11} | {'value':^12} | {'error':^12} | {'fixed':^5} | {'lower':^10} | {'upper':^10}"
    sep    = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    # Rows
    for i, p in enumerate(minuit_obj.params):
        low  = p.lower_limit if p.lower_limit is not None else ""
        high = p.upper_limit if p.upper_limit is not None else ""
        print(f"{i:3d} | {p.name:11s} | {p.value:12.6f} | {p.error:12.6f} | "
              f"{str(p.is_fixed):^5s} | {str(low):10s} | {str(high):10s}")
    print(sep)

def print_fit_progress(m):
    print(f"Iteration: N={m.values['N']:.1f}, ε={m.values['epsilon']:.3f}, "
          f"B_p={m.values['B_p']:.1f}, B_f={m.values['B_f']:.1f}, "
          f"fval={m.fval:.1f}")
    
class PassFailPlotter:
    def __init__(self, cost_func, n_bins_pass, edges_pass, edges_fail, fit_type):
        self.cost = cost_func
        self.n_bins_pass = n_bins_pass
        self.edges_pass = edges_pass
        self.edges_fail = edges_fail
        self.param_names = FIT_CONFIGS[fit_type]["param_names"]
        self.signal_func = FIT_CONFIGS[fit_type]["signal_func"]
        self.bg_func = FIT_CONFIGS[fit_type]["background_func"]
        self.signal_param_names = SIGNAL_MODELS[fit_type.split('_')[0]]["params"]
        self.bg_param_names = BACKGROUND_MODELS[fit_type.split('_')[1]]["params"]

    def __call__(self, args):
        # Unpack param vector into a dict
        param_dict = dict(zip(self.param_names, args))

        # Split the data
        data = self.cost.data
        data_pass = data[:self.n_bins_pass]
        data_fail = data[self.n_bins_pass:]

        cx_pass = 0.5 * (self.edges_pass[:-1] + self.edges_pass[1:])
        cx_fail = 0.5 * (self.edges_fail[:-1] + self.edges_fail[1:])

        # Rebuild signal and background
        signal_params = [param_dict[p] for p in self.signal_param_names]
        bg_pass_params = [param_dict[f"{p}_pass"] for p in self.bg_param_names]
        bg_fail_params = [param_dict[f"{p}_fail"] for p in self.bg_param_names]

        signal_pass = self.signal_func(cx_pass, *signal_params)
        signal_fail = self.signal_func(cx_fail, *signal_params)

        bg_pass = self.bg_func(cx_pass, *bg_pass_params)
        bg_fail = self.bg_func(cx_fail, *bg_fail_params)

        N = param_dict["N"]
        epsilon = param_dict["epsilon"]
        B_p = param_dict["B_p"]
        B_f = param_dict["B_f"]

        signal_y_pass = N - (B_p - B_f) * epsilon * signal_pass
        signal_y_fail = N - (B_p - B_f) * (1 - epsilon) * signal_fail
        bg_y_pass = B_p * bg_pass
        bg_y_fail = B_f * bg_fail
        total_pass = signal_y_pass + bg_y_pass
        total_fail = signal_y_fail + bg_y_fail

        # Plot pass
        plt.subplot(2, 1, 1)
        plt.cla()
        plt.title("Pass")
        plt.errorbar(cx_pass, data_pass, yerr=np.sqrt(data_pass), fmt='o', color='black', label='Data')
        plt.stairs(bg_y_pass, self.edges_pass, fill=True, color='orange', label='Background')
        plt.stairs(total_pass, self.edges_pass, baseline=bg_y_pass, fill=True, color='skyblue', label='Signal')
        plt.stairs(total_pass, self.edges_pass, color='navy', label='Total Fit')
        plt.legend()

        # Plot fail
        plt.subplot(2, 1, 2)
        plt.cla()
        plt.title("Fail")
        plt.errorbar(cx_fail, data_fail, yerr=np.sqrt(data_fail), fmt='o', color='black', label='Data')
        plt.stairs(bg_y_fail, self.edges_fail, fill=True, color='orange', label='Background')
        plt.stairs(total_fail, self.edges_fail, baseline=bg_y_fail, fill=True, color='skyblue', label='Signal')
        plt.stairs(total_fail, self.edges_fail, color='navy', label='Total Fit')
        plt.legend()

        plt.tight_layout()
    
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
    safe_x = np.clip(x, x_min, x_max)
    result = (safe_x-x_min)**a * (x_max-safe_x)**b
    result[(x<=x_min)|(x>=x_max)] = 0
    norm = np.trapezoid((safe_x-x_min)**a * (x_max-safe_x)**b, x)
    if norm <= 0:  # Prevent division by zero
        norm = 1e-9
    return result / norm

def linear(x, b, C):
    threshold=0.01

    if b == 0 and abs(C) <= threshold:
        return np.ones_like(x)
    else:
        # Clamp C if it's very close to zero
        if abs(C) <= threshold:
            C = 0
        
        y = b + C * x
        integral = np.trapezoid(y, x)
        if integral == 0:
            raise ValueError("Integral is zero, cannot normalize")
        
        return y / integral

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

def create_combined_model(fit_type, all_edges):
    if fit_type not in FIT_CONFIGS:
        raise ValueError(f"Unknown fit type: {fit_type}")
    
    config = FIT_CONFIGS[fit_type]
    signal_func = config["signal_func"]
    bg_func = config["background_func"]
    param_names = config["param_names"]


    # Define edges_pass and edges_fail based on all_edges
    edges_pass = all_edges[:40]  # first 40 values
    edges_fail = np.concatenate(([all_edges[0]], all_edges[-39:]))  # first value + last 39 values

    def combined_model(all_edges, *params):
        # Use the newly defined edges_pass and edges_fail arrays
        x = edges_pass
        y = edges_fail

        params_dict = dict(zip(param_names, params))
        
        # Shared signal parameters
        signal_params = [params_dict[p] for p in SIGNAL_MODELS[fit_type.split('_')[0]]["params"]]
        
        # Background parameters
        bg_pass_params = [params_dict[f"{p}_pass"] for p in BACKGROUND_MODELS[fit_type.split('_')[1]]["params"]]
        bg_fail_params = [params_dict[f"{p}_fail"] for p in BACKGROUND_MODELS[fit_type.split('_')[1]]["params"]]
        
        # Calculate components
        signal_pass = signal_func(x, *signal_params)
        signal_fail = signal_func(y, *signal_params)
        bg_pass = bg_func(x, *bg_pass_params)
        bg_fail = bg_func(y, *bg_fail_params)
        
        # Combine results according to formula
        N = params_dict["N"]
        epsilon = params_dict["epsilon"]
        B_p = params_dict["B_p"]
        B_f = params_dict["B_f"]
        

        results = (N - (B_p - B_f)) * ( epsilon * signal_pass + (1 - epsilon) * signal_fail) + B_p * bg_pass + B_f * bg_fail
        
       # result_pass = params_dict["N"] * params_dict["epsilon"] * signal_pass + params_dict["B_p"] * bg_pass
       # result_fail = params_dict["N"] * (1 - params_dict["epsilon"]) * signal_fail + params_dict["B_f"] * bg_fail
        
        return results
    
    return combined_model

def fit_function(fit_type, hist_pass, hist_fail, fixed_params=None, x_min=x_min, x_max=x_max):
    fixed_params = fixed_params or {}

    if fit_type not in FIT_CONFIGS:
        raise ValueError(f"Unknown fit type: {fit_type}")

    config = FIT_CONFIGS[fit_type]
    param_names = config["param_names"]

    # Prepare data: centers, values, errors for pass
    centers_pass = (hist_pass["edges"][:-1] + hist_pass["edges"][1:]) / 2
    edges_pass = hist_pass["edges"]
    values_pass = hist_pass["values"]
    values_pass[values_pass <= 0] = 0.000001
    errors_pass = hist_pass.get("errors", np.sqrt(values_pass))
    errors_pass[errors_pass <= 0] = 0.000001


    # Prepare data for fail
    centers_fail = (hist_fail["edges"][:-1] + hist_fail["edges"][1:]) / 2
    edges_fail = hist_fail["edges"]
    values_fail = hist_fail["values"]
    values_fail[values_fail <= 0] = 0.000001
    errors_fail = hist_fail.get("errors", np.sqrt(values_fail))
    errors_fail[errors_fail <= 0] = 0.000001


    # Crop window
    mask_pass = (centers_pass >= x_min) & (centers_pass <= x_max)
    mask_fail = (centers_fail >= x_min) & (centers_fail <= x_max)

    bin_mask_pass = (edges_pass[:-1] >= x_min) & (edges_pass[1:] <= x_max)
    centers_pass = centers_pass[mask_pass]
    edges_pass = edges_pass[np.r_[bin_mask_pass, False] | np.r_[False, bin_mask_pass]]
    values_pass = values_pass[mask_pass]
    errors_pass = errors_pass[mask_pass]

    bin_mask_fail = (edges_fail[:-1] >= x_min) & (edges_fail[1:] <= x_max)
    centers_fail = centers_fail[mask_fail]
    edges_fail = edges_fail[np.r_[bin_mask_fail, False] | np.r_[False, bin_mask_fail]]
    values_fail = values_fail[mask_fail]
    errors_fail = errors_fail[mask_fail]

    # Combine pass and fail data arrays
    print(f"Length of edges pass: {len(edges_pass)}")
    print(f"Edges: {(edges_pass)}")
    print(f"Length of edges fail: {len(edges_fail)}")
    print(f"Edges: {(edges_fail)}")
    all_edges = np.concatenate([edges_pass, edges_fail[1:]])
    all_centers = np.concatenate([centers_pass, centers_fail])
    all_values = np.concatenate([values_pass, values_fail])
    print(f"Length of all values: {len(all_values)}")
    # errors not used in NLL but can be kept if needed elsewhere

    # Calculate data-based initial guesses
    N_p0 = (np.sum(values_pass) + np.sum(values_fail))
    B_p_p0 = max(1, np.median(values_pass[-10:]) * len(values_pass))
    B_f_p0 = max(1, np.median(values_fail[-10:]) * len(values_fail))

    # Scale fixed parameters if present
    for name in ['N', 'epsilon', 'B_p', 'B_f']:
        if name in fixed_params:
            fixed_params[name]

    # Update bounds with data-based values
    bounds = config["bounds"].copy()
    bounds.update({
        "N":   (0, N_p0, N_p0*4),
        "B_p": (0, B_p_p0/4, B_p_p0),
        "B_f": (0, 1, B_f_p0*5),
        "epsilon": (0.8, 0.95, 1)
    })

    print(f"Initial N value: {N_p0}")
    print(f"Initial B_p value: {B_p_p0}")
    print(f"Initial B_f value: {B_f_p0}")

    # Prepare initial parameter guesses
    p0 = []
    bounds_low = []
    bounds_high = []
    initial_guesses = {}  # Store ALL initial guesses here

    for name in param_names:
        if name in fixed_params:
            initial_guesses[name] = fixed_params[name]
            continue
        else:
            # Use middle value from bounds
            b = bounds[name]
            initial_guesses[name] = b[1]
            
        # Set bounds and add to p0 for minimization
        b = bounds[name]
        p0.append(initial_guesses[name])
        bounds_low.append(b[0])
        bounds_high.append(b[2])

    print(f"Length edges_pass: {len(edges_pass)}")
    print(f"Length edges_fail: {len(edges_fail)}")
    print(f"Length all_edges: {len(all_edges)}")

    print(f"Length values_pass: {len(values_pass)}")
    print(f"Length values_fail: {len(values_fail)}")
    print(f"Length all_values: {len(all_values)}")

    # Create model function from fit_type
    model = create_combined_model(fit_type, all_edges)
    c = cost.ExtendedBinnedNLL(all_values, all_edges, model, use_pdf = 'approximate')
    c.errdef = Minuit.LIKELIHOOD
    
    init = initial_guesses
    m = Minuit(c, *[init[name] for name in param_names], name=param_names)
    for name in param_names:
        if name in fixed_params:
            # Set parameter value and fix it
            init[name] = fixed_params[name]
            m.fixed[name] = True
        elif name in bounds:
            m.limits[name] = (bounds[name][0], bounds[name][2])
    plotter = PassFailPlotter(c, len(edges_pass)-1, edges_pass, edges_fail , fit_type)
    m.interactive(plotter)
    #m.scipy()
    m.simplex()

    # 5. Run the fit:
    m.migrad()
    m.print_level = 2
    m.hesse()
    print_minuit_params_table(m)
    print_fit_progress(m)

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
        "values_pass": values_pass,
        "errors_pass": errors_pass,
        "centers_fail": centers_fail,
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
            "mu": (88, 90.5, 92),
            "sigma": (1, 3, 10),
            "alphaL": (0.1, 1.0, 100),
            "nL": (0.11, 5.0, 200),
            "alphaR": (0.1, 1.0, 100),
            "nR": (0.11, 5.0, 200)
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
            "C": (-0.25, 0.1, 0.25)
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