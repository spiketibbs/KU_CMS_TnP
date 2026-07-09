import uproot
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import mplhep as hep
import os
import argparse
from scipy.special import wofz

def load_histogram(root_file, hist_name):
    keys = {key.split(";")[0]: key for key in root_file.keys()}
    if hist_name in keys:
        obj = root_file[keys[hist_name]]
        if isinstance(obj, uproot.behaviors.TH1.Histogram):
            values, edges = obj.to_numpy()
            errors = obj.errors()  #  use stored errors, not sqrt(N)
            return {"values": values, "edges": edges, "errors": errors}
    return None

def double_crystal_ball(x, A, mu, sigma, alphaL, nL, alphaR, nR):
    z = (x - mu) / sigma
    result = np.zeros_like(z)

    mask_core = (z > -alphaL) & (z < alphaR)
    result[mask_core] = A * np.exp(-0.5 * z[mask_core]**2)

    mask_left = z <= -alphaL
    abs_alphaL = np.abs(alphaL)
    NL = (nL / abs_alphaL)**nL * np.exp(-0.5 * abs_alphaL**2)
    result[mask_left] = A * NL * (nL / abs_alphaL - abs_alphaL - z[mask_left])**(-nL)

    mask_right = z >= alphaR
    abs_alphaR = np.abs(alphaR)
    NR = (nR / abs_alphaR)**nR * np.exp(-0.5 * abs_alphaR**2)
    result[mask_right] = A * NR * (nR / abs_alphaR - abs_alphaR + z[mask_right])**(-nR)

    return result

def double_voigtian(x, A, mu, eta, sigma_L, gamma_L, sigma_R, gamma_R):
    def voigt_profile(x, mu, sigma, gamma):
        z = ((x - mu) + 1j * gamma) / (sigma * np.sqrt(2))
        return np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))                            
    V_left = voigt_profile(x, mu, sigma_L, gamma_L)
    V_right = voigt_profile(x, mu, sigma_R, gamma_R)
    shape = A * (eta * V_left + (1 - eta) * V_right)
    return shape

def phase_space(x, B, a, b, x_min, x_max):
    delta = 1e-5  # or 1e-3 depending on bin resolution
    safe_x = np.clip(x, x_min + delta, x_max - delta)
    shape = (safe_x - x_min)**a * (x_max - safe_x)**b
    shape[(x <= x_min) | (x >= x_max)] = 0
    return B * shape

def linear(x, B, C):
    shape = B + C * x
    return shape

def exponential(x, B, C):
    shape = B * np.e**(C * x)
    return shape

def dcb_plus_phase(x, A, mu, sigma, alphaL, nL, alphaR, nR, B, a, b, x_min, x_max):
    return double_crystal_ball(x, A, mu, sigma, alphaL, nL, alphaR, nR) + phase_space(x, B, a, b, x_min, x_max)
def dcb_plus_linear(x, A, mu, sigma, alphaL, nL, alphaR, nR, B, C):
    return double_crystal_ball(x, A, mu, sigma, alphaL, nL, alphaR, nR) + linear(x, B, C)
def dcb_plus_exponential(x, A, mu, sigma, alphaL, nL, alphaR, nR, B, C):
    return double_crystal_ball(x, A, mu, sigma, alphaL, nL, alphaR, nR) + exponential(x, B, C)
def dv_plus_phase(x, A, mu, eta, sigma_L, gamma_L, sigma_R, gamma_R, B, a, b, x_min, x_max):
    return double_voigtian(x, A, mu, eta, sigma_L, gamma_L, sigma_R, gamma_R) + phase_space(x, B, a, b, x_min, x_max)
def dv_plus_linear(x, A, eta, mu, sigma_L, gamma_L, sigma_R, gamma_R, B, C):
    return double_voigtian(x, A, mu, eta, sigma_L, gamma_L, sigma_R, gamma_R) + linear(x, B, C)
def dv_plus_exponential(x, A, eta, mu, sigma_L, gamma_L, sigma_R, gamma_R, B, C):
    return double_voigtian(x, A, mu, eta, sigma_L, gamma_L, sigma_R, gamma_R) + exponential(x, B, C)
def compute_signal_background_events_dcb_ps(x, popt, x_min, x_max):
    signal = double_crystal_ball(x, *popt[:7])
    background = phase_space(x, popt[7], popt[8], popt[9], x_min, x_max)
    return np.trapezoid(signal, x), np.trapezoid(background, x)
def compute_signal_background_events_dcb_lin(x, popt):
    signal = double_crystal_ball(x, *popt[:7])
    background = linear(x, popt[7], popt[8])
    return np.trapezoid(signal, x), np.trapezoid(background, x)
def compute_signal_background_events_dcb_exp(x, popt):
    signal = double_crystal_ball(x, *popt[:7])
    background = exponential(x, popt[7], popt[8])
    return np.trapezoid(signal, x), np.trapezoid(background, x)
def compute_signal_background_events_dv_ps(x, popt, x_min, x_max):
    signal = double_voigtian(x, *popt[:7])
    background = phase_space(x, popt[7], popt[8], popt[9], x_min, x_max)
    return np.trapezoid(signal, x), np.trapezoid(background, x)
def compute_signal_background_events_dv_lin(x, popt):
    signal = double_voigtian(x, *popt[:7])
    background = linear(x, popt[7], popt[8], popt[9])
    return np.trapezoid(signal, x), np.trapezoid(background, x)
def compute_signal_background_events_dv_exp(x, popt):
    signal = double_voigtian(x, *popt[:7])
    background = exponential(x, popt[7], popt[8], popt[9])
    return np.trapezoid(signal, x), np.trapezoid(background, x)

def fit_hist(type, hist, hist_name, plot_dir, data_type):
    print(f"Fitting histogram '{hist_name}' ...")
    os.makedirs(plot_dir, exist_ok=True)
    plt.figure(figsize=(12, 8))
    hep.style.use("CMS")

    centers = (hist["edges"][:-1] + hist["edges"][1:]) / 2
    values = hist["values"]
    errors = hist["errors"]  # use real bin errors
    errors[errors == 0] = 1.0  # avoid zero division

    x_min, x_max = 70, 110
    mask = (centers >= x_min) & (centers <= x_max)
    centers = centers[mask]
    values = values[mask]
    errors = errors[mask]

    plt.errorbar(centers, values, yerr=errors, fmt='o', color='royalblue', capsize=3)
    
    # Define parameter bounds and initial guesses for each fit type
    FIT_CONFIGS = {
        "dcb_ps": {
            "param_names": ["A", "mu", "sigma", "alphaL", "nL", "alphaR", "nR", "B", "a", "b"],
            "bounds": {
                "A": (max(values)*0.5, max(values)*0.8, max(values)*3),  # (lower, initial, upper)
                "mu": (89.74, 89.75, 89.76),
                "sigma": (2.75, 2.76, 2.77),
                "alphaL": (0.90, 0.91, 0.92),
                "nL": (2, 10, 40),
                "alphaR": (1.74, 1.75, 1.76),
                "nR": (3.01, 3.02, 3.03),
                "B": (0.0001, 100, np.inf),
                "a": (1.43, 1.44, 1.45),
                "b": (2.08, 2.09, 2.10)
            }
        },
        "dcb_lin": {
            "param_names": ["A", "mu", "sigma", "alphaL", "nL", "alphaR", "nR", "B", "C"],
            "bounds": {
                "A": (max(values)*0.5, max(values)*0.8, max(values)*2),
                "mu": (89, 90, 92),
                "sigma": (2.61, 2.62, 2.63),
                "alphaL": (0.65, 0.66, 0.67),
                "nL": (0.1, 5, 10),
                "alphaR": (1.69, 1.70, 1.71),
                "nR": (1, 5, 10),
                "B": (0.001, 1, np.inf),
                "C": (0.1, 0.1, np.inf)
            }
        },
        "dcb_exp": {
            "param_names": ["A", "mu", "sigma", "alphaL", "nL", "alphaR", "nR", "B", "C"],
            "bounds": {
                "A": (max(values)*0.5, max(values)*0.8, max(values)*2),
                "mu": (89, 90, 92),
                "sigma": (2.61, 2.62, 2.63),
                "alphaL": (0.65, 0.66, 0.67),
                "nL": (0.1, 5, 10),
                "alphaR": (1.69, 1.70, 1.71),
                "nR": (1, 5, 10),
                "B": (0.001, 1, np.inf),
                "C": (0.1, 0.1, np.inf)
            }
        },
        "dv_ps": {
            "param_names": ["A", "mu", "eta", "sigma_L", "gamma_L", "sigma_R", "gamma_R", "B", "a", "b"],
            "bounds": {
                "A": (max(values)*0.1, max(values)*5, np.inf),
                "mu": (88, 90, 92),
                "eta": (0.1, 0.5, 0.9),
                "sigma_L": (0.5, 2, 4),
                "gamma_L": (0.1, 1, 2),
                "sigma_R": (0.5, 2, 4),
                "gamma_R": (0.1, 1, 2),
                "B": (0.001, 1, np.inf),
                "a": (0.6, 0.65, 0.7),
                "b": (0.65, 0.7, 0.75)
            }
        },
        "dv_lin": {
            "param_names": ["A", "mu", "eta", "sigma_L", "gamma_L", "sigma_R", "gamma_R", "B", "C"],
            "bounds": {
                "A": (max(values)*0.5, max(values)*0.8, max(values)*2),
                "mu": (89, 90, 92),
                "eta": (0.49, 0.5, 0.51),
                "sigma_L": (0.1, 5, 15),
                "gamma_L": (0.1, 5, 15),
                "sigma_R": (0.1, 5, 15),
                "gamma_R": (0.1, 5, 15),
                "B": (0.001, 1, np.inf),
                "C": (0.1, 0.1, np.inf)
            }
        },
        "dv_exp": {
            "param_names": ["A", "mu", "eta", "sigma_L", "gamma_L", "sigma_R", "gamma_R", "B", "C"],
            "bounds": {
                "A": (max(values)*0.5, max(values)*0.8, max(values)*2),
                "mu": (89, 90, 92),
                "eta": (0.49, 0.5, 0.51),
                "sigma_L": (0.1, 5, 15),
                "gamma_L": (0.1, 5, 15),
                "sigma_R": (0.1, 5, 15),
                "gamma_R": (0.1, 5, 15),
                "B": (0.001, 1, np.inf),
                "C": (0.1, 0.1, np.inf)
            }
        }
    }

    def get_fit_parameters(fit_type):
        """Get initial parameters and bounds for the specified fit type"""
        config = FIT_CONFIGS[fit_type]
        
        lower_dict = {}
        p0_dict = {}
        upper_dict = {}
        
        for param, (lower, p0, upper) in config["bounds"].items():
            lower_val = lower
            p0_val = p0
            upper_val = upper
                
            lower_dict[param] = lower_val
            p0_dict[param] = p0_val
            upper_dict[param] = upper_val
        
        return config["param_names"], lower_dict, p0_dict, upper_dict


# DOUBLE CRYSTAL BALL PLUS PHASE SPACE
    if type == "dcb_ps":
        def model(x, A, mu, sigma, alphaL, nL, alphaR, nR, B, a, b):
            return dcb_plus_phase(x, A, mu, sigma, alphaL, nL, alphaR, nR, B, a, b, x_min, x_max)
# DOUBLE CRYSTAL BALL PLUS LINEAR/EXPONENTIAL
    elif type == "dcb_lin" or type == "dcb_exp":
        if type == "dcb_lin":
            def model(x, A, mu, sigma, alphaL, nL, alphaR, nR, B, C):
                return dcb_plus_linear(x, A, mu, sigma, alphaL, nL, alphaR, nR, B, C)
        elif type == "dcb_exp":
            def model(x, A, mu, sigma, alphaL, nL, alphaR, nR, B, C):
                return dcb_plus_exponential(x, A, mu, sigma, alphaL, nL, alphaR, nR, B, C)
# DOUBLE VOIGTIAN PLUS PHASE SPACE
    elif type == "dv_ps":
        def model(x, A, mu, eta, sigma_L, gamma_L, sigma_R, gamma_R, B, a, b):
            return dv_plus_phase(x, A, mu, eta, sigma_L, gamma_L, sigma_R, gamma_R, B, a, b, x_min, x_max)
# DOUBLE VOIGTIAN PLUS LINEAR/EXPONENTIAL
    elif type == "dv_lin" or type == "dv_exp":
        if type == "dv_lin":
            def model(x, A, mu, eta, sigma_L, gamma_L, sigma_R, gamma_R, B, C):
                return dv_plus_linear(x, A, mu, eta, sigma_L, gamma_L, sigma_R, gamma_R, B, C)
        elif type == "dv_exp":
            def model(x, A, mu, eta, sigma_L, gamma_L, sigma_R, gamma_R, B, C):
                return dv_plus_exponential(x, A, mu, eta, sigma_L, gamma_L, sigma_R, gamma_R, B, C)

    param_names, lower_dict, p0_dict, upper_dict = get_fit_parameters(type)

    p0 = [p0_dict[name] for name in param_names]
    lower = [lower_dict[name] for name in param_names]
    upper = [upper_dict[name] for name in param_names]

    popt, pcov, infodict, errmsg, ier = curve_fit(
        model, centers, values, p0=p0, sigma=errors,
        absolute_sigma=True, bounds= (lower, upper), full_output = True, maxfev=20000)
    
    #print(infodict)
    # Check for convergence
    if ier == 1 or ier == 2:
        print(f"Curve fit converged successfully, ier = {ier}")
    elif ier == 3 or ier == 4 or ier == 5:
        print(f"Curve fit did not converge, ier = {ier} ")
        print("Error message:", errmsg)

    #print("Fitted parameters:", popt)
    #print("Covariance matrix:", pcov)

    perr = np.sqrt(np.diag(pcov))

    x = np.linspace(x_min, x_max, 1000)
    if type == "dcb_ps":
        signal = double_crystal_ball(x, *popt[:7])
        background = phase_space(x, popt[7], popt[8], popt[9], x_min, x_max)
        signal_events, background_events = compute_signal_background_events_dcb_ps(x, popt, x_min, x_max)
        signal_label = "Double Crystal Ball"
        background_label = "Phase-space"
        signal_params = f"DCB (Signal): A={popt[0]:.2f} ± {perr[0]:.2f}, μ={popt[1]:.2f} ± {perr[1]:.2f}, σ={popt[2]:.2f} ± {perr[2]:.2f}\n αL={popt[3]:.2f} ± {perr[3]:.2f}, nL={popt[4]:.2f} ± {perr[4]:.2f}, αR={popt[5]:.2f} ± {perr[5]:.2f}, nR={popt[6]:.2f} ± {perr[6]:.2f}\n"
        background_params = f"Phase Background: B={popt[7]:.2f} ± {perr[7]:.2f}, a={popt[8]:.2f} ± {perr[8]:.2f}, b={popt[9]:.2f} ± {perr[9]:.2f}"
    elif type == "dcb_lin":
        signal = double_crystal_ball(x, *popt[:7])
        background = linear(x, popt[7], popt[8])
        signal_events, background_events = compute_signal_background_events_dcb_lin(x, popt)
        signal_label = "Double Crystal Ball"
        background_label = "Linear"
        signal_params = f"DCB (Signal): A={popt[0]:.2f} ± {perr[0]:.2f}, μ={popt[1]:.2f} ± {perr[1]:.2f}, σ={popt[2]:.2f} ± {perr[2]:.2f}\n αL={popt[3]:.2f} ± {perr[3]:.2f}, nL={popt[4]:.2f} ± {perr[4]:.2f}, αR={popt[5]:.2f} ± {perr[5]:.2f}, nR={popt[6]:.2f} ± {perr[6]:.2f}\n"
        background_params = f"Linear Background: B={popt[7]:.2f} ± {perr[7]:.2f}, C={popt[8]:.2f} ± {perr[8]:.2f}"
    elif type == "dcb_exp":
        signal = double_crystal_ball(x, *popt[:7])
        background = exponential(x, popt[7], popt[8])
        signal_events, background_events = compute_signal_background_events_dcb_exp(x, popt)
        signal_label = "Double Crystal ball"
        background_label = "Exponential"
        signal_params = f"DCB (Signal): A={popt[0]:.2f} ± {perr[0]:.2f}, μ={popt[1]:.2f} ± {perr[1]:.2f}, σ={popt[2]:.2f} ± {perr[2]:.2f}\n αL={popt[3]:.2f} ± {perr[3]:.2f}, nL={popt[4]:.2f} ± {perr[4]:.2f}, αR={popt[5]:.2f} ± {perr[5]:.2f}, nR={popt[6]:.2f} ± {perr[6]:.2f}\n"
        background_params = f"Exp Background: B={popt[7]:.2f} ± {perr[7]:.2f}, C={popt[8]:.2f} ± {perr[8]:.2f}"
    elif type == "dv_ps":
        signal = double_voigtian(x, *popt[:7])
        background = phase_space(x, popt[7], popt[8], popt[9], x_min, x_max)
        signal_events, background_events = compute_signal_background_events_dv_ps(x, popt, x_min, x_max)
        signal_label = "Double Voigtian"
        background_label = "Phase-space"
        signal_params = f"DV (Signal): A={popt[0]:.2f} ± {perr[0]:.2f}, μ={popt[1]:.2f} ± {perr[1]:.2f}, η={popt[2]:.2f} ± {perr[2]:.2f}\n σL={popt[3]:.2f} ± {perr[3]:.2f}, γL={popt[4]:.2f} ± {perr[4]:.2f}, σR={popt[5]:.2f} ± {perr[5]:.2f}, γR={popt[6]:.2f} ± {perr[6]:.2f}\n"
        background_params = f"Phase Background: B={popt[7]:.2f} ± {perr[7]:.2f}, a={popt[8]:.2f} ± {perr[8]:.2f}, b={popt[9]:.2f} ± {perr[9]:.2f}"
    elif type == "dv_lin":
        signal = double_voigtian(x, *popt[:7])
        background = linear(x, popt[7], popt[8])
        signal_events, background_events = compute_signal_background_events_dv_lin(x, popt)
        signal_label = "Double Voigtian"
        background_label = "Linear"
        signal_params = f"DV (Signal): A={popt[0]:.2f} ± {perr[0]:.2f}, μ={popt[1]:.2f} ± {perr[1]:.2f}, η={popt[2]:.2f} ± {perr[2]:.2f}\n σL={popt[3]:.2f} ± {perr[3]:.2f}, γL={popt[4]:.2f} ± {perr[4]:.2f}, σR={popt[5]:.2f} ± {perr[5]:.2f}, γR={popt[6]:.2f} ± {perr[6]:.2f}\n"
        background_params = f"Linear Background: B={popt[7]:.2f} ± {perr[7]:.2f}, C={popt[8]:.2f} ± {perr[8]:.2f}"
    elif type == "dv_exp":
        signal = double_voigtian(x, *popt[:7])
        background = exponential(x, popt[7], popt[8])
        signal_events, background_events = compute_signal_background_events_dv_exp(x, popt)
        signal_label = "Double Voigtian"
        background_label = "Exponential"
        signal_params = f"DV (Signal): A={popt[0]:.2f} ± {perr[0]:.2f}, μ={popt[1]:.2f} ± {perr[1]:.2f}, η={popt[2]:.2f} ± {perr[2]:.2f}\n σL={popt[3]:.2f} ± {perr[3]:.2f}, γL={popt[4]:.2f} ± {perr[4]:.2f}, σR={popt[5]:.2f} ± {perr[5]:.2f}, γR={popt[6]:.2f} ± {perr[6]:.2f}\n"
        background_params = f"Exponential Background: B={popt[7]:.2f} ± {perr[7]:.2f}, C={popt[8]:.2f} ± {perr[8]:.2f}"

        
    combined = signal + background

    dA, dsigma = perr[0], perr[2]
    signal_error = np.sqrt((popt[2] * np.sqrt(2 * np.pi) * dA)**2 + (popt[0] * np.sqrt(2 * np.pi) * dsigma)**2)

    if type == "dcb_ps" or type == "dv_ps":
        dB = perr[7]
        background_shape = (x - x_min)**popt[8] * (x_max - x)**popt[9]
        background_error = np.trapezoid(background_shape, x) * dB
    else:
        background_error = np.sqrt(perr[7]**2 + perr[8]**2)

    expected = model(centers, *popt)
    residuals = values - expected
    standardized_residuals = residuals / errors

    #for i, s_r in enumerate(standardized_residuals):
        #if abs(s_r) > 2:
            #print(f"High residual at x = {centers[i]:.2f}: standardized residual - {s_r:.2f}")

    chi_squared = np.sum(((values - expected) / errors) ** 2)
    dof = len(values) - len(popt)
    reduced_chi_squared = chi_squared / dof
    print(f"B = {popt[7]:.6f} ± {perr[7]:.6f}")

    print(f" - Estimated number of signal events: {signal_events:.2f} ± {signal_error:.2f}")
    print(f" - Estimated number of background events: {background_events:.2f} ± {background_error:.2f}")
    print(f" - Chi-squared: {chi_squared:.2f}, Reduced Chi-squared: {reduced_chi_squared:.2f}")

    plt.plot(x, signal, color='orange', label=signal_label)
    plt.plot(x, background, color='red', label=background_label)
    plt.plot(x, combined, color='black', label="Total Fit")

    legend_text = (
        f"{signal_params}"
        f"{background_params}\n"
        f"Signal Events: {signal_events:.2f} ± {signal_error:.2f}\n"
        f"Background Events: {background_events:.2f} ± {background_error:.2f}\n"
        f"χ² = {chi_squared:.2f}, Reduced χ² = {reduced_chi_squared:.2f}"
    )

    plt.text(0.05, 0.95, legend_text, transform=plt.gca().transAxes,
             fontsize=9, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

    plt.title(f"Fit to {hist_name} ({background_label} background)")
    plt.xlabel(r"$m_{ee}$ [GeV]")
    plt.ylabel("Number of events")
    plt.legend(loc='upper right', fontsize=8)
    plt.savefig(f"{plot_dir}/{data_type}_{type}_fit_{hist_name}.svg")
    plt.close()
    print("Finished fit!\n")

    return signal_events, signal_error

def main():
    # Define all available bins
    bins = {
        "bin00": ("pt_5p00To8p00", "5.00-8.00"),
        "bin01": ("pt_8p00To10p00", "8.00-10.00"),
        "bin02": ("pt_10p00To15p00", "10.00-15.00"),
        "bin03": ("pt_15p00To20p00", "15.00-20.00"),
        "bin04": ("pt_20p00To30p00", "20.00-30.00"),
        "bin05": ("pt_30p00To35p00", "30.00-35.00"),
        "bin06": ("pt_35p00To40p00", "35.00-40.00"),
        "bin07": ("pt_40p00To45p00", "40.00-45.00"),
        "bin08": ("pt_45p00To50p00", "45.00-50.00"),
        "bin09": ("pt_50p00To55p00", "50.00-55.00"),
        "bin10": ("pt_55p00To60p00", "55.00-60.00"),
        "bin11": ("pt_60p00To80p00", "60.00-80.00"),
        "bin12": ("pt_80p00To100p00", "80.00-100.00"),
        "bin13": ("pt_100p00To150p00", "100.00-150.00"),
        "bin14": ("pt_150p00To250p00", "150.00-250.00"),
        "bin15": ("pt_250p00To400p00", "250.00-400.00")
    }
    
    parser = argparse.ArgumentParser(description="Fit ROOT histograms with different models.")
    parser.add_argument("--bin", type=str, required=True, choices=bins.keys(), 
                       help="Which bin to fit (e.g., bin00, bin01, etc.)")
    parser.add_argument("--type", type=str, required=True, 
                       choices=["dcb_ps", "dcb_lin", "dcb_exp", "dv_ps", "dv_lin", "dv_exp"])
    parser.add_argument("--data", type=str, required=True, choices=["DATA", "MC"])
    args = parser.parse_args()

    if args.data == "DATA":
        root_file = uproot.open("/uscms/home/hortua/nobackup/egamma-tnp/examples/nanoaod_filters_custom/blp1/DATA_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_23D_histos_pt_barrel.root")
    elif args.data == "MC":
        root_file = uproot.open("/uscms/home/hortua/nobackup/egamma-tnp/examples/nanoaod_filters_custom/blp1/MC_DY_2023/get_1d_pt_eta_phi_tnp_histograms_1/MC_DY_23D_histos_pt_barrel.root")
    plot_dir = f"{args.bin}_fits"

    # Get the bin information
    bin_suffix, bin_range = bins[args.bin]
    hist_name_pass = f"{args.bin}_{bin_suffix}_Pass"
    hist_name_fail = f"{args.bin}_{bin_suffix}_Fail"

    # Load the histograms
    hist_pass = load_histogram(root_file, hist_name_pass)
    hist_fail = load_histogram(root_file, hist_name_fail)

    # Perform the fits
    Npass, Npass_err = fit_hist(args.type, hist_pass, hist_name_pass, plot_dir, args.data)
    Nfail, Nfail_err = fit_hist(args.type, hist_fail, hist_name_fail, plot_dir, args.data)

    # Calculate efficiency
    Ntotal = Npass + Nfail
    efficiency = Npass / Ntotal
    efficiency_error = np.sqrt(efficiency * (1 - efficiency) / Ntotal)

    print(f"Efficiency for pt bin {bin_range} GeV = {efficiency:.4f} ± {efficiency_error:.4f}")

if __name__ == "__main__":
    main()
