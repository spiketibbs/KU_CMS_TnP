import argparse
import os
from functools import partial

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import uproot
from numpy.polynomial.chebyshev import Chebyshev
from scipy.optimize import curve_fit
from scipy.special import voigt_profile

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
    return result / norm

def double_voigtian(x, mu, sigma1, gamma1, sigma2, gamma2):
    result = (voigt_profile(x-mu, sigma1, gamma1) + 
              voigt_profile(x-mu, sigma2, gamma2))
    return result / np.trapezoid(result, x)

def double_gaussian(x, mu, sigma):
    gauss = lambda x, s: np.exp(-0.5*((x-mu)/s)**2)/(s*np.sqrt(2*np.pi))
    return gauss(x, sigma) / np.trapezoid(gauss(x, sigma), x)

def phase_space(x, a, b, x_min=70, x_max=110):
    safe_x = np.clip(x, x_min, x_max)
    result = (safe_x-x_min)**a * (x_max-safe_x)**b
    result[(x<=x_min)|(x>=x_max)] = 0
    return result / np.trapezoid((safe_x-x_min)**a * (x_max-safe_x)**b, x)

def linear(x, b, C):
    if b == 0 and C == 0:
        return np.ones_like(x)
    else:
        return (b + C*x) / np.trapezoid(b + C*x, x)

def exponential(x, C):
    return np.exp(C*x) / np.trapezoid(np.exp(C*x), x)

def chebyshev_background(x, *coeffs, x_min=70, x_max=110):
    x_norm = 2*(x-x_min)/(x_max-x_min) - 1
    return Chebyshev(coeffs)(x_norm) / np.trapezoid(Chebyshev(coeffs)(x_norm), x)

def create_combined_model(fit_type):
    if fit_type not in FIT_CONFIGS:
        raise ValueError(f"Unknown fit type: {fit_type}")
    
    config = FIT_CONFIGS[fit_type]
    signal_func = config["signal_func"]
    bg_func = config["background_func"]
    
    # Get parameter names and split into components
    param_names = config["param_names"]
    
    def combined_model(x, y, *params):
        params_dict = dict(zip(param_names, params))
        
        # Extract signal parameters for pass/fail
        signal_pass_params = []
        signal_fail_params = []
        for p in SIGNAL_MODELS[fit_type.split('_')[0]]["params"]:
            signal_pass_params.append(params_dict[f"{p}_pass"])
            signal_fail_params.append(params_dict[f"{p}_fail"])
        
        # Extract background parameters for pass/fail
        bg_pass_params = []
        bg_fail_params = []
        for p in BACKGROUND_MODELS[fit_type.split('_')[1]]["params"]:
            bg_pass_params.append(params_dict[f"{p}_pass"])
            bg_fail_params.append(params_dict[f"{p}_fail"])
        
        # Calculate components
        signal_pass = signal_func(x, *signal_pass_params)
        signal_fail = signal_func(y, *signal_fail_params)
        bg_pass = params_dict["B_p"] * bg_func(x, *bg_pass_params)
        bg_fail = params_dict["B_f"] * bg_func(y, *bg_fail_params)
        
        # Combine with efficiency
        return (params_dict["N"] * ( params_dict["epsilon"] * signal_pass + 
                (1 - params_dict["epsilon"]) * signal_fail) + 
                bg_pass + bg_fail)
    
    return combined_model

def fit_function(fit_type, hist_pass, hist_fail, fixed_params=None):
    fixed_params = fixed_params or {}
    
    if fit_type not in FIT_CONFIGS:
        raise ValueError(f"Unknown fit type: {fit_type}")
    
    # Prepare data
    centers_pass = (hist_pass["edges"][:-1] + hist_pass["edges"][1:]) / 2
    values_pass = hist_pass["values"]
    errors_pass = hist_pass["errors"]
    errors_pass[errors_pass == 0] = 1.0
    
    centers_fail = (hist_fail["edges"][:-1] + hist_fail["edges"][1:]) / 2
    values_fail = hist_fail["values"]
    errors_fail = hist_fail["errors"]
    errors_fail[errors_fail == 0] = 1.0
    
    # Mass window
    x_min, x_max = 70, 110

    # Mask and crop pass data
    mask_pass = (centers_pass >= x_min) & (centers_pass <= x_max)
    centers_pass = centers_pass[mask_pass]
    values_pass = values_pass[mask_pass]
    errors_pass = errors_pass[mask_pass]

    # Mask and crop fail data
    mask_fail = (centers_fail >= x_min) & (centers_fail <= x_max)
    centers_fail = centers_fail[mask_fail]
    values_fail = values_fail[mask_fail]
    errors_fail = errors_fail[mask_fail]

    # Combine only filtered data
    all_centers = np.concatenate([centers_pass, centers_fail])
    all_values = np.concatenate([values_pass, values_fail])
    all_errors = np.concatenate([errors_pass, errors_fail])
    
    # Get model configuration
    config = FIT_CONFIGS[fit_type]
    param_names = config["param_names"]
    
    # Create model
    model = create_combined_model(fit_type)
    
    # Prepare initial parameters and bounds
    p0 = []
    bounds_low = []
    bounds_high = []
    
    for name in param_names:
        if name in fixed_params:
            continue
        bounds = config["bounds"][name]
        p0.append(bounds[1])  # Use middle value as initial guess
        bounds_low.append(bounds[0])
        bounds_high.append(bounds[2])
    
    bounds = (bounds_low, bounds_high)
    
    # Handle fixed parameters
    if fixed_params:
        fixed_indices = {i: fixed_params[name] for i, name in enumerate(param_names) if name in fixed_params}
        model = create_fixed_param_wrapper(model, fixed_indices)
    
    # Perform fit
    try:
        popt, pcov = curve_fit(model, all_centers, all_values, p0=p0, sigma=all_errors, absolute_sigma=True, 
                             bounds=bounds, maxfev=20000)
    except Exception as e:
        print(f"Fit failed: {e}")
        return None
    
    # Reconstruct full parameters
    full_popt = []
    free_idx = 0
    for i, name in enumerate(param_names):
        if name in fixed_params:
            full_popt.append(fixed_params[name])
        else:
            full_popt.append(popt[free_idx])
            free_idx += 1
    
    # Calculate fit quality
    expected = model(all_centers, *popt)
    chi2 = np.sum(((all_values - expected)/all_errors)**2)
    ndf = len(all_values) - len(popt)
    reduced_chi2 = chi2 / ndf if ndf > 0 else float('inf')
    
    return {
        "centers_pass": centers_pass,
        "values_pass": values_pass,
        "errors_pass": errors_pass,
        "centers_fail": centers_fail,
        "values_fail": values_fail,
        "errors_fail": errors_fail,
        "x_min": x_min,
        "x_max": x_max,
        "popt": dict(zip(param_names, full_popt)),
        "pcov": pcov,
        "chi_squared": chi2,
        "reduced_chi_squared": reduced_chi2,
        "type": fit_type,
        "param_names": param_names
    }

def plot_combined_fit(results, efficiency=None, eff_error=None, plot_dir=".", data_type="DATA"):
    """Plot pass and fail components dynamically based on fit type"""
    if results is None:
        print("No results to plot")
        return
    
    fit_type = results["type"]
    config = FIT_CONFIGS[fit_type]
    signal_func = config["signal_func"]
    bg_func = config["background_func"]
    params = results["popt"]
    
    # Create x values for plotting
    x = np.linspace(results["x_min"], results["x_max"], 1000)
    
    # Plot PASS components
    plt.figure(figsize=(12, 8))
    hep.style.use("CMS")
    
    # Get signal parameters for pass
    signal_pass_params = []
    for p in SIGNAL_MODELS[fit_type.split('_')[0]]["params"]:
        signal_pass_params.append(params[f"{p}_pass"])
    
    # Get background parameters for pass
    bg_pass_params = []
    for p in BACKGROUND_MODELS[fit_type.split('_')[1]]["params"]:
        bg_pass_params.append(params[f"{p}_pass"])
    
    # Calculate components
    signal_pass = params["N"] * params["epsilon"] * signal_func(x, *signal_pass_params)
    bg_pass = params["B_p"] * bg_func(x, *bg_pass_params)
    total_pass = signal_pass + bg_pass
    
    # Plot pass data and fit
    plt.errorbar(results["centers_pass"], results["values_pass"], yerr=results["errors_pass"], 
                fmt="o", color="royalblue", markersize=6, capsize=3, label="Data (Pass)")
    plt.plot(x, total_pass, 'k-', label="Total fit")
    plt.plot(x, signal_pass, 'r--', label="Signal")
    plt.plot(x, bg_pass, 'g--', label="Background")
    
    # Formatting
    plt.xlabel("$m_{ee}$ [GeV]", fontsize=12)
    plt.ylabel("Events / GeV", fontsize=12)
    plt.title(f"{data_type.replace('_', ' ')}: {BINS_INFO[results['bin']][1]} GeV (Pass)", pad=10)
    
    # Add fit info
    dof = len(results["centers_pass"]) + len(results["centers_fail"]) - len(results["popt"])
    chi2_red = results["chi_squared"] / dof

    signal_params_text = "\n".join([f"{p}_pass = {params[f'{p}_pass']:.3f}" 
                                    for p in SIGNAL_MODELS[fit_type.split('_')[0]]["params"]])
    bg_params_text = "\n".join([f"{p}_pass = {params[f'{p}_pass']:.3f}" 
                                for p in BACKGROUND_MODELS[fit_type.split('_')[1]]["params"]])

    info_text = [
        f"Signal yield: {params['N']*params['epsilon']:.1f}",
        f"Bkg yield: {params['B_p']:.1f}",
        f"ε = {efficiency:.3f} ± {eff_error:.3f}",
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
    
    # Get signal parameters for fail
    signal_fail_params = []
    for p in SIGNAL_MODELS[fit_type.split('_')[0]]["params"]:
        signal_fail_params.append(params[f"{p}_fail"])
    
    # Get background parameters for fail
    bg_fail_params = []
    for p in BACKGROUND_MODELS[fit_type.split('_')[1]]["params"]:
        bg_fail_params.append(params[f"{p}_fail"])
    
    # Calculate components
    signal_fail = signal_func(x, *signal_pass_params)
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
    
    dof = len(results["centers_pass"]) + len(results["centers_fail"]) - len(results["popt"])
    chi2_red = results["chi_squared"] / dof

    signal_params_text = "\n".join([f"{p}_fail = {params[f'{p}_fail']:.3f}" 
                                    for p in SIGNAL_MODELS[fit_type.split('_')[0]]["params"]])
    bg_params_text = "\n".join([f"{p}_fail = {params[f'{p}_fail']:.3f}" 
                                for p in BACKGROUND_MODELS[fit_type.split('_')[1]]["params"]])

    info_text = [
        f"Signal yield: {params['N']*(1-params['epsilon']):.1f}",
        f"Bkg yield: {params['B_f']:.1f}",
        f"ε = {efficiency:.3f} ± {eff_error:.3f}",
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
    # Save fail plot
    plt.savefig(f"{plot_dir}/{data_type}_{results['type']}_fit_{results['bin']}_Fail.png", 
               bbox_inches="tight", dpi=300)
    plt.close()
    
    print(f"Plots saved for {results['bin']} (Pass/Fail)")

# Define BINS_INFO globally so plot_fit and main can access it
BINS_INFO = {
    f"bin{i:02d}": (f"pt_{lo}p00To{hi}p00", f"{lo:.2f}-{hi:.2f}")
    for i, (lo, hi) in enumerate([
        (5,8), (8,10), (10,15), (15,20), (20,30), (30,35), 
        (35,40), (40,45), (45,50), (50,55), (55,60), 
        (60,80), (80,100), (100,150), (150,250), (250,400)
    ])
}

SIGNAL_MODELS = {
    "dcb": {
        "func": double_crystal_ball,
        "params": ["mu", "sigma", "alphaL", "nL", "alphaR", "nR"],
        "bounds": {
            "mu": (89, 90, 91),
            "sigma": (1.0, 2.5, 5),
            "alphaL": (0.1, 1.0, 100),
            "nL": (0.1, 5.0, 100),
            "alphaR": (0.1, 1.0, 100),
            "nR": (0.1, 5.0, 100)
        }
    },
    "dv": {
        "func": double_voigtian,
        "params": ["mu", "sigma1", "gamma1", "sigma2", "gamma2"],
        "bounds": {
            "mu": (89, 90, 91),
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
    }
}

# Then define all background models
BACKGROUND_MODELS = {
    "ps": {
        "func": phase_space,
        "params": ["a", "b"],
        "bounds": {
            "a": (0.1, 1.44, 10),
            "b": (0.1, 2.09, 10)
        }
    },
    "lin": {
        "func": linear,
        "params": ["b", "C"],
        "bounds": {
            "b": (-1, 0, 1),
            "C": (-2, -0.5, 0.1)
        }
    },
    "exp": {
        "func": exponential,
        "params": ["C"],
        "bounds": {
            "C": (0.0001, 0.1, 10)
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
    }
}

# Now combine them into all possible combinations
FIT_CONFIGS = {}
for sig_name, sig_config in SIGNAL_MODELS.items():
    for bg_name, bg_config in BACKGROUND_MODELS.items():
        fit_type = f"{sig_name}_{bg_name}"
        
        # Build parameter names list
        param_names = ["N", "epsilon", "B_p", "B_f"]
        
        # Add pass/fail versions of signal parameters
        for p in sig_config["params"]:
            param_names.extend([f"{p}_pass", f"{p}_fail"])
        
        # Add pass/fail versions of background parameters
        for p in bg_config["params"]:
            param_names.extend([f"{p}_pass", f"{p}_fail"])
        
        # Build bounds dictionary
        bounds = {
            "N": (0, 7000, np.inf),
            "epsilon": (0, 0.5, 1),
            "B_p": (0, 10000, np.inf),
            "B_f": (0, 10000, np.inf)
        }
        
        # Add signal bounds (same for pass/fail)
        for p, b in sig_config["bounds"].items():
            bounds[f"{p}_pass"] = b
            bounds[f"{p}_fail"] = b
        
        # Add background bounds (same for pass/fail)
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
    plot_dir = f"{args.bin}_fits/{'DATA' if args.data.startswith('DATA') else 'MC'}"
    os.makedirs(plot_dir, exist_ok=True)
    
    # Load histograms
    bin_suffix, bin_range = BINS_INFO[args.bin]
    hist_pass = load_histogram(root_file, f"{args.bin}_{bin_suffix}_Pass")
    hist_fail = load_histogram(root_file, f"{args.bin}_{bin_suffix}_Fail")
    
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
    
    # Calculate efficiency
    N = results["popt"]["N"]
    epsilon = results["popt"]["epsilon"]
    eff = epsilon
    # Get the index of epsilon in the parameter list for covariance matrix
    param_names = list(results["popt"].keys())
    epsilon_index = param_names.index("epsilon")
    eff_err = np.sqrt(results["pcov"][epsilon_index, epsilon_index]) if results["pcov"].size > 0 else 0
    
    # Plot results
    plot_combined_fit(results, eff, eff_err, plot_dir, args.data)
    
    root_file.close()
    print(f"\nEfficiency for pt bin {bin_range} GeV = {eff:.4f} ± {eff_err:.4f}")

if __name__ == "__main__":
    main()