import numpy as np
import os
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares
import matplotlib.gridspec as gridspec

# ==========================================
# --- 1. DICTIONARY SETUP ---
# ==========================================
DATA_DICT = {
    "bar1": 
    [
        {"DATA": "0.9567 ± 0.0039", "MC": "0.9756 ± 0.0019", "SF": "0.98063 ± 0.00446"},
        {"DATA": "0.9551 ± 0.0025", "MC": "0.9721 ± 0.0014", "SF": "0.98247 ± 0.00296"},
        {"DATA": "0.9452 ± 0.0013", "MC": "0.9697 ± 0.0010", "SF": "0.97476 ± 0.00170"},
        {"DATA": "0.9441 ± 0.0013", "MC": "0.9662 ± 0.0007", "SF": "0.97709 ± 0.00148"},
        {"DATA": "0.9428 ± 0.0008", "MC": "0.9660 ± 0.0005", "SF": "0.97601 ± 0.00099"},
        {"DATA": "0.9430 ± 0.0006", "MC": "0.9659 ± 0.0004", "SF": "0.97631 ± 0.00077"},
        {"DATA": "0.9445 ± 0.0005", "MC": "0.9657 ± 0.0004", "SF": "0.97808 ± 0.00067"},
        {"DATA": "0.9410 ± 0.0006", "MC": "0.9645 ± 0.0005", "SF": "0.97561 ± 0.00077"}
    ],
    "bar2": 
    [
        {"DATA": "0.9636 ± 0.0056", "MC": "0.9683 ± 0.0027", "SF": "0.99519 ± 0.00636"},
        {"DATA": "0.9531 ± 0.0029", "MC": "0.9640 ± 0.0021", "SF": "0.98868 ± 0.00372"},
        {"DATA": "0.9378 ± 0.0021", "MC": "0.9508 ± 0.0013", "SF": "0.98639 ± 0.00258"},
        {"DATA": "0.9364 ± 0.0017", "MC": "0.9487 ± 0.0011", "SF": "0.98703 ± 0.00208"},
        {"DATA": "0.9325 ± 0.0013", "MC": "0.9454 ± 0.0011", "SF": "0.98635 ± 0.00180"},
        {"DATA": "0.9291 ± 0.0009", "MC": "0.9485 ± 0.0012", "SF": "0.97946 ± 0.00154"},
        {"DATA": "0.9288 ± 0.0001", "MC": "0.9443 ± 0.0006", "SF": "0.98361 ± 0.00066"},
        {"DATA": "0.9249 ± 0.0009", "MC": "0.9413 ± 0.0007", "SF": "0.98267 ± 0.00120"}
    ],
}

# Use raw strings (r"") for LaTeX formatting
ETA_RANGES = {
    "bar1": r"0.0 \leq |\eta| < 0.8",
    "bar2": r"0.8 \leq |\eta| < 1.4"
}

# ==========================================
# --- 2. HELPER FUNCTIONS ---
# ==========================================

def parse_dataset(data_list):
    """Parses 'val ± err' strings into numpy arrays."""
    data_effs, data_errs = [], []
    mc_effs, mc_errs = [], []
    sf_orig = []
    
    for row in data_list:
        d_val, d_err = map(float, row["DATA"].split("±"))
        m_val, m_err = map(float, row["MC"].split("±"))
        s_val, _     = map(float, row["SF"].split("±"))
        
        data_effs.append(d_val)
        data_errs.append(d_err)
        mc_effs.append(m_val)
        mc_errs.append(m_err)
        sf_orig.append(s_val)
        
    return (np.array(data_effs), np.array(data_errs), 
            np.array(mc_effs), np.array(mc_errs), np.array(sf_orig))

def fitter(effs, errs, x_vals, func):
    """Standard Minuit LeastSquares fitter."""
    mask = np.isfinite(effs) & np.isfinite(errs)
    x_fit = x_vals[mask]
    y_fit = np.array(effs)[mask]
    y_errs = np.array(errs)[mask]

    least_squares = LeastSquares(x_fit, y_fit, y_errs, func)
    
    m = Minuit(least_squares, a=0, b=np.mean(y_fit))
    m.limits["a"] = (-0.1, 0.1)  
    m.limits["b"] = (0.0, 1.5)   
    m.migrad()
    
    return m, x_fit, y_fit, y_errs

# ==========================================
# --- 3. MAIN EXTRAPOLATION ROUTINE ---
# ==========================================

def extrapolate_and_plot(dataset_key, fit_bin_ranges, extrap_bins):
    print(f"\nProcessing {dataset_key}...")
    
    # Extract data dynamically
    data_list = DATA_DICT[dataset_key]
    data_effs, data_errs, mc_effs, mc_errs, sf_orig = parse_dataset(data_list)
    
    bin_centers = np.array([(low + high)/2 for low, high in fit_bin_ranges])
    x_center = np.mean(bin_centers) 
    
    def poly3_centered(x, a, b):
        return a * (x - x_center) + b

    # --- Toy samples ---
    N_toys = 100000
    rng = np.random.default_rng(seed=9990)
    data_toys = [rng.normal(mu, sigma, N_toys) for mu, sigma in zip(data_effs, data_errs)]
    mc_toys   = [rng.normal(mu, sigma, N_toys) for mu, sigma in zip(mc_effs, mc_errs)]

    # --- Scale factors ---
    sf_toys = [d/m for d, m in zip(data_toys, mc_toys)]
    sf_means = np.array([np.mean(sf) for sf in sf_toys])

    # --- PROCESS DATA & MC (Stats) ---
    data_means = np.array([np.mean(d) for d in data_toys])
    data_mins = np.array([np.min(d) for d in data_toys])
    data_maxs = np.array([np.max(d) for d in data_toys])
    data_fit_errs = (data_maxs - data_mins) / 2.0

    mc_means = np.array([np.mean(m) for m in mc_toys])
    mc_mins = np.array([np.min(m) for m in mc_toys])
    mc_maxs = np.array([np.max(m) for m in mc_toys])
    mc_fit_errs = (mc_maxs - mc_mins) / 2.0

    # Create range errors for plotting data and mc
    data_range_errors = [data_means - data_mins, data_maxs - data_means]
    mc_range_errors = [mc_means - mc_mins, mc_maxs - mc_means]

    data_m, _, _, _ = fitter(data_means, data_fit_errs, bin_centers, poly3_centered)
    mc_m, _, _, _ = fitter(mc_means, mc_fit_errs, bin_centers, poly3_centered)

    # --- PROCESS SCALE FACTORS (Stats) ---
    sf_mins = np.array([np.min(sf) for sf in sf_toys])
    sf_maxs = np.array([np.max(sf) for sf in sf_toys])

    yerr_lower = sf_means - sf_mins
    yerr_upper = sf_maxs - sf_means
    sf_range_errors = [yerr_lower, yerr_upper]
    sf_fit_errs = (sf_maxs - sf_mins) / 2.0

    sf_m, sf_x_fit, sf_y_fit, sf_y_err = fitter(sf_means, sf_fit_errs, bin_centers, poly3_centered)

    # ==========================================
    # --- PLOTTING SETUP ---
    # ==========================================
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, width_ratios=[3, 1]) 

    ax_eff = fig.add_subplot(gs[0, 0]) 
    ax_sf = fig.add_subplot(gs[1, 0], sharex=ax_eff) 
    ax_tables = fig.add_subplot(gs[:, 1]) 
    ax_tables.axis('off') 

    # --- TOP PLOT: Efficiency & SF Toys ---
# --- TOP PLOT: Efficiency & SF Toys ---
    for i, ((low, hi), toys, orig) in enumerate(zip(fit_bin_ranges, data_toys, data_effs)):
        ax_eff.hlines(orig, low, hi, color="blue", linewidth=2, label="Original DATA" if i==0 else "")
        ax_eff.scatter([ (low+hi)/2 ]*N_toys, toys, marker='x', color="blue", 
                       rasterized=True, alpha=0.02, label="DATA toys" if i==0 else "")

    for i, ((low, hi), toys, orig) in enumerate(zip(fit_bin_ranges, mc_toys, mc_effs)):
        ax_eff.hlines(orig, low, hi, color="green", linewidth=2, label="Original MC" if i==0 else "")
        ax_eff.scatter([ (low+hi)/2 ]*N_toys, toys, marker='x', color="green", 
                       rasterized=True, alpha=0.02, label="MC toys" if i==0 else "")

    for i, ((low, hi), toys, orig) in enumerate(zip(fit_bin_ranges, sf_toys, sf_orig)):
        ax_eff.hlines(orig, low, hi, color="red", linewidth=2, label="Original SF" if i==0 else "")
        ax_eff.scatter([ (low+hi)/2 ]*N_toys, toys, marker='x', color="red", 
                       rasterized=True, alpha=0.02, label="SF toys" if i==0 else "")

    ax_eff.set_ylabel("Efficiency / Scale Factor")
    ax_eff.legend(loc='lower right')
    ax_eff.grid(True)

    # --- BOTTOM PLOT: All Fits and Extrapolations ---
    x_min_plot = extrap_bins[0][0]
    x_max_plot = fit_bin_ranges[-1][1]
    
    # 1. Setup Fit Lines (Full Range)
    xx = np.linspace(x_min_plot, x_max_plot + 5, 500)
    yy_data_fit = poly3_centered(xx, data_m.values["a"], data_m.values["b"])
    yy_mc_fit   = poly3_centered(xx, mc_m.values["a"], mc_m.values["b"])
    yy_sf_fit   = poly3_centered(xx, sf_m.values["a"], sf_m.values["b"])

    # 2. Setup Error Bands 
    xx_band = np.linspace(x_min_plot, x_max_plot, 200)

    data_res = np.abs(data_means - poly3_centered(bin_centers, data_m.values["a"], data_m.values["b"]))
    width_data = np.interp(xx_band, bin_centers, data_res)

    mc_res = np.abs(mc_means - poly3_centered(bin_centers, mc_m.values["a"], mc_m.values["b"]))
    width_mc = np.interp(xx_band, bin_centers, mc_res)

    sf_res = np.abs(sf_means - poly3_centered(bin_centers, sf_m.values["a"], sf_m.values["b"]))
    width_sf = np.interp(xx_band, bin_centers, sf_res)

    # -------------------------------------------------------------
    # NEW LOGIC: Adjust error band widths for plots
    # -------------------------------------------------------------
    mask_3_6 = (xx_band >= 2) & (xx_band < 4)
    width_data[mask_3_6] = np.where(width_data[mask_3_6] < 0.02, 0.02, width_data[mask_3_6])
    width_mc[mask_3_6]   = np.where(width_mc[mask_3_6]   < 0.02, 0.02, width_mc[mask_3_6])
    width_sf[mask_3_6]   = np.where(width_sf[mask_3_6]   < 0.02, 0.02, width_sf[mask_3_6])

    mask_6_10 = (xx_band >= 4) & (xx_band <= 7)
    width_data[mask_6_10] = np.where(width_data[mask_6_10] < 0.01, 0.01, width_data[mask_6_10])
    width_mc[mask_6_10]   = np.where(width_mc[mask_6_10]   < 0.01, 0.01, width_mc[mask_6_10])
    width_sf[mask_6_10]   = np.where(width_sf[mask_6_10]   < 0.01, 0.01, width_sf[mask_6_10])
    # -------------------------------------------------------------

    yy_data_band = poly3_centered(xx_band, data_m.values["a"], data_m.values["b"])
    yy_mc_band   = poly3_centered(xx_band, mc_m.values["a"], mc_m.values["b"])
    yy_sf_band   = poly3_centered(xx_band, sf_m.values["a"], sf_m.values["b"])

    # --- Plotting Fits and Error Bands ---
    # DATA
    ax_sf.scatter(bin_centers, data_effs, marker='x', color="blue", s=80, zorder=10, label="Data original")
    ax_sf.errorbar(bin_centers, data_means, yerr=data_range_errors, fmt='o', color="darkblue", zorder=10, label="Data mean")
    ax_sf.plot(xx, yy_data_fit, color="blue", lw=2, linestyle='-', label="Data Fit")
    ax_sf.fill_between(xx_band, yy_data_band - width_data, yy_data_band + width_data, color='blue', alpha=0.2, label="Data Band")

    # MC
    ax_sf.scatter(bin_centers, mc_effs, marker='x', color="green", s=80, zorder=10, label="MC original")
    ax_sf.errorbar(bin_centers, mc_means, yerr=mc_range_errors, fmt='o', color="darkgreen", zorder=10, label="MC mean")
    ax_sf.plot(xx, yy_mc_fit, color="green", lw=2, linestyle='-', label="MC Fit")
    ax_sf.fill_between(xx_band, yy_mc_band - width_mc, yy_mc_band + width_mc, color='green', alpha=0.2, label="MC Band")

    # SF
    ax_sf.scatter(bin_centers, sf_orig, marker='x', color="red", s=80, zorder=10, label="SF original")
    ax_sf.errorbar(bin_centers, sf_means, yerr=sf_range_errors, fmt='o', color="black", zorder=10, label="SF mean")
    ax_sf.plot(xx, yy_sf_fit, color="red", lw=2, linestyle='-', label="SF Fit")
    ax_sf.fill_between(xx_band, yy_sf_band - width_sf, yy_sf_band + width_sf, color='gray', alpha=0.4, label="SF Band")

    # --- Extrapolations Loop ---
    extrap_results_for_table = [] 

    print(f"\n--- EXTRAPOLATION RESULTS: {dataset_key} ---")
    print(f"{'Bin':<10} | {'DATA (eff ± err)':<18} | {'MC (eff ± err)':<18} | {'SF (val ± err)':<18}")
    print("-" * 75)

    for i, (x1, x2) in enumerate(extrap_bins):
        x_mid = (x1 + x2) / 2.0
        
        # Calculate raw interpolations first
        y_d = poly3_centered(x_mid, data_m.values['a'], data_m.values['b'])
        err_d = np.interp(x_mid, bin_centers, data_res)
        
        y_m = poly3_centered(x_mid, mc_m.values['a'], mc_m.values['b'])
        err_m = np.interp(x_mid, bin_centers, mc_res)
        
        y_s = poly3_centered(x_mid, sf_m.values['a'], sf_m.values['b'])
        err_s = np.interp(x_mid, bin_centers, sf_res)

        # -------------------------------------------------------------
        # NEW LOGIC: Adjust discrete errors for error bars and tables
        # -------------------------------------------------------------
        if x1 == 2 and x2 == 4:
            err_d = 0.02 if err_d < 0.02 else err_d
            err_m = 0.02 if err_m < 0.02 else err_m
            err_s = 0.02 if err_s < 0.02 else err_s
        elif x1 == 4 and x2 == 7:
            err_d = 0.01 if err_d < 0.01 else err_d
            err_m = 0.01 if err_m < 0.01 else err_m
            err_s = 0.01 if err_s < 0.01 else err_s
        # -------------------------------------------------------------

        # Plot adjusted error bars
        ax_sf.hlines(y_d, x1, x2, color='blue', linestyle='--', linewidth=2)
        ax_sf.errorbar(x_mid, y_d, yerr=err_d, fmt='o', color='blue', capsize=5)

        ax_sf.hlines(y_m, x1, x2, color='green', linestyle='--', linewidth=2)
        ax_sf.errorbar(x_mid, y_m, yerr=err_m, fmt='o', color='green', capsize=5)

        ax_sf.hlines(y_s, x1, x2, color='purple', linestyle='--', linewidth=2)
        ax_sf.errorbar(x_mid, y_s, yerr=err_s, fmt='o', color='purple', capsize=5)
        
        # Terminal Print
        print(f"{x1}-{x2:<5} | {y_d:.4f} ± {err_d:.4f}   | {y_m:.4f} ± {err_m:.4f}   | {y_s:.4f} ± {err_s:.4f}")
        
        # Save ALL values for the matplotlib table
        extrap_results_for_table.append(((x1, x2), (y_d, err_d), (y_m, err_m), (y_s, err_s)))

    print("=" * 75 + "\n")

    # --- Tables ---
    chi2 = sf_m.fval
    ndof = len(sf_x_fit) - sf_m.nfit
    chi2_ndof = chi2 / ndof if ndof > 0 else np.nan
    
    # 1. Fit Parameter Table (SF only, to save space)
    fit_table_data = []
    for par in sf_m.parameters:
        val, err = sf_m.values[par], sf_m.errors[par]
        fit_table_data.append([par, f"{val:.4f} ± {err:.4f}"])
    fit_table_data.append([r"$\chi^2$/ndof", f"{chi2:.2f} / {ndof} = {chi2_ndof:.2f}"])

    fit_table = ax_tables.table(cellText=fit_table_data, colLabels=["Parameter (SF)", "Value"], 
                                cellLoc='center', colLoc='center', bbox=[0.1, 0.75, 0.8, 0.20])
    fit_table.auto_set_font_size(False); fit_table.set_fontsize(9)

    # 2. Max Diff Table
    diff_table_data = [[f"{int(low)}-{int(high)}", f"{diff:.4f}"] for (low, high), diff in zip(fit_bin_ranges[:len(sf_res)], sf_res)]
    diff_table = ax_tables.table(cellText=diff_table_data, colLabels=["Bin [GeV]", "Max diff (SF)"], 
                                 cellLoc='center', colLoc='center', bbox=[0.1, 0.45, 0.8, 0.25])
    diff_table.auto_set_font_size(False); diff_table.set_fontsize(9)

    # 3. Extrapolation Table (Transposed: Pt range on top, Types on side)
    extrap_col_labels = ["Type"] + [f"{x1}-{x2}" for ((x1, x2), _, _, _) in extrap_results_for_table]
    
    data_row = ["DATA"] + [f"{yd:.4f} ± {ed:.4f}" for (_, (yd, ed), _, _) in extrap_results_for_table]
    mc_row   = ["MC"]   + [f"{ym:.4f} ± {em:.4f}" for (_, _, (ym, em), _) in extrap_results_for_table]
    sf_row   = ["SF"]   + [f"{ys:.4f} ± {es:.4f}" for (_, _, _, (ys, es)) in extrap_results_for_table]
    
    transposed_table_data = [data_row, mc_row, sf_row]

    extrap_table = ax_tables.table(cellText=transposed_table_data, colLabels=extrap_col_labels, 
                                   cellLoc='center', colLoc='center', bbox=[0.0, 0.05, 1.0, 0.30])
    extrap_table.auto_set_font_size(False); extrap_table.set_fontsize(9) 

    # --- Final Layout ---
    ax_sf.set_xlabel("pT bin center [GeV]")
    ax_sf.set_ylabel("Values (Eff or SF)")
    
    ax_sf.legend(loc='upper right', fontsize='small', ncol=3) 
    ax_sf.grid(True)

    # Dynamic Ticks based on extrap bins + fit bins
    extrap_ticks = [x1 for x1, _ in extrap_bins]
    fit_ticks = [low for low, _ in fit_bin_ranges] + [fit_bin_ranges[-1][1]]
    tick_positions = sorted(list(set(extrap_ticks + fit_ticks)))
    tick_labels    = [str(x) for x in tick_positions]
    
    plt.setp(ax_eff.get_xticklabels(), visible=False)
    ax_sf.set_xticks(tick_positions)
    ax_sf.set_xticklabels(tick_labels)

    # --- UPDATED TITLE LOGIC HERE ---
    eta_label = ETA_RANGES.get(dataset_key, dataset_key)
    plt.suptitle(f"PROMPT Data/MC Efficiency Scale Factor Fits (${eta_label}$)", fontsize=16, fontweight="bold")
    
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # --- FILE PATH REMAINS UNCHANGED ---
    output_file = f"MUONS/EXTRAP_PLOTS_1/23/2023_PROMPT_postBPix_extrap_{dataset_key}.pdf"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    #plt.savefig(output_file, dpi=300)
    print(f"Plot generation finished for {dataset_key}")
    #plt.show()

# ==========================================
# --- 4. EXECUTION ---
# ==========================================

if __name__ == "__main__":
    my_fit_bins = [(10, 15), (15, 20), (20, 25), (25, 30), (30, 35),(35, 40), (40, 45), (45, 50)] 
    my_extrap_bins = [(2, 4), (4,7)]
    
    for key in DATA_DICT.keys():
        extrapolate_and_plot(key, my_fit_bins, my_extrap_bins)