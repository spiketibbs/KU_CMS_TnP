###################################################################################
#                            CONFIG.JSON FORMAT
###################################################################################
#
#   {
#     "info_level": "INFO_2"       < ----- INFO, DEBUG, or INFO_2 (more verbose) (leave blank for no output)
#     "mass": "Z",                 < ----- Determines what mass you are fitting (Z, Z_muon, JPsi, JPsi_muon)
#     "input": {
#       "root_files_DATA": [                                  < ----- The name will be the name of the plot file that is saved in plot_dir
#           "NAME DATA 1":   ".root DATA file path 1 ..."          < ----- The name will be the name of the plot file that is saved in plot_dir
#           "NAME DATA 2":   ".root DATA file path 2 ..."          < ----- The name will be the name of the plot file that is saved in plot_dir
#           "NAME DATA 3":   ".root DATA file path 3 ..."          < ----- The name will be the name of the plot file that is saved in plot_dir
#       ],
#       "root_files_MC": [
#           "NAME MC 1":     ".root MC file path 1 ..."            < ----- The name will be the name of the plot file that is saved in plot_dir
#           "NAME MC 2":     ".root MC file path 2 ..."            < ----- The name will be the name of the plot file that is saved in plot_dir
#           "NAME MC 3":     ".root MC file path 3 ..."            < ----- The name will be the name of the plot file that is saved in plot_dir
#       ]
#     },
#     "fit": {
#       "bin_ranges": [[5,7], [7,10], [10,20], [20,45], [45,75], [75,500]],    < ----- Specify which pT range(s) you are fitting (in example, bin0 (5-7), bin1 (7-10), bin2 (10-20), bin3 (20-45), bin4 (45-75), bin5 (75-500))
#       "bin": ["bin0", "bin1, etc"],    < ----- Specify which pT range(s) you are fitting (in example, bin0 (5-7), bin1 (7-10), bin2 (10-20), bin3 (20-45), bin4 (45-75), bin5 (75-500))
#       "fit_type": "dcb_cms"    < ----- Format is: (signal shape)_(background shape). Signal shapes: (dcb, g, dv, cbg), Background shapes: (lin, exp, cms, bpoly, cheb, ps)
#       "use_cdf": false,        < ----- If a shape does not have a cdf version, defaults back to pdf
#       "sigmoid_eff": false,    < ----- Switches to an unbounded efficiency that is transformed back between 0 and 1
#       "interactive": true,     < ----- Turns on interactive window for fitting (very useful for difficult fits)
#       "x_min": 70,             < ----- x range minimum for plotting
#       "x_max": 110,            < ----- x range maximum for plotting
#       "abseta": 1,             < ----- ***Only impacts muon .root files. Defines absolute eta ranges
#       "numerator": "gold",     < ----- ***Only impacts muon .root files. Defines numerator for efficiencies
#       "denominator": "blp"     < ----- ***Only impacts muon .root files. Defines denominator for efficiencies
#     },
#     "output": {
#       "plot_dir": "",          < ----- Sets location to save plots to (if left blank, it won't save)
#       "results_file": ""       < ----- Sets location to save results to (if left blank, it won't save)
#    },
#    "scale_factors": {
#        "data_mc_pair": {                                      < ----- Creates explicit scale factors for pairs of data and MC files (useful for comparing one file to multiple others)
#            "Scale Factor 1": ["NAME DATA 1", "NAME MC 1"],    < ----- Outputs scale factor of two file specified. DATA must be put before MC
#            "Scale Factor 2": ["NAME DATA 2", "NAME MC 2"],    < ----- Outputs scale factor of two file specified. DATA must be put before MC
#            "Scale Factor 3": ["NAME DATA 3", "NAME MC 3"]     < ----- Outputs scale factor of two file specified. DATA must be put before MC
#     }
#    }
#  }
#
###################################################################################
# RUN EXAMPLE CONFIG FILE: run-fitter --config src/egamma_tnp/config/config_example.json
###################################################################################
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import numpy as np
import uproot
import matplotlib.pyplot as plt
import mplhep as hep
from collections import defaultdict
from rich import box
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TaskProgressColumn, ProgressColumn, TimeRemainingColumn
from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.console import Group
from rich.live import Live


from fit_function import fit_function, logging, print_efficiency_summary, CustomTimeElapsedColumn
from fitter_plot_model import load_histogram, plot_combined_fit, fig_to_array, BINS_INFO, save_fits_and_plots

import warnings
import pandas as pd

# Suppress specific Minuit warnings about fixed parameters
warnings.filterwarnings("ignore", message="Cannot scan over fixed parameter")
console = Console()

COLOR_BORDER = "#00B4D8"
COLOR_PRIMARY = "#E6EDF3"
COLOR_SECONDARY = "#AAB3BF"
COLOR_SUCCESS = "#06D6A0"
COLOR_ERROR = "#E63946"
COLOR_HIGHLIGHT = "#FFB703"
COLOR_WARNING = "#F77F00"
COLOR_BG_DARK = "#0D1117"


def _coerce_excel_value(value):
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return value
        try:
            return int(stripped)
        except ValueError:
            try:
                return float(stripped)
            except ValueError:
                return value
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    mass = config["mass"]
    root_files_DATA = config["input"].get("root_files_DATA", {})
    root_files_MC = config["input"].get("root_files_MC", {})
    separate_signal_shape = config["fit"].get("separate_signal_shape", False)

    # Infer year from DATA file paths, or let config override it explicitly.
    def normalize_year(year_value):
        if year_value is None:
            return None
        y = str(year_value).strip()
        if y.isdigit() and len(y) == 4:
            return y[-2:]
        if y.isdigit() and len(y) == 2:
            return y
        return y

    def infer_year(root_files):
        for path in root_files.values():
            p = str(path).lower()
            if "run2016" in p or "2016" in p:
                return "16"
            if "run2017" in p or "2017" in p:
                return "17"
            if "run2018" in p or "2018" in p:
                return "18"
            if "run2022" in p or "2022" in p:
                return "22"
            if "run2023" in p or "2023" in p:
                return "23"
            if "run2024" in p or "2024" in p:
                return "24"
        return "unknown"

    year = normalize_year(config["fit"].get("year")) or infer_year(root_files_DATA)

    fit_type = config["fit"]["fit_type"]
    use_cdf = config["fit"].get("use_cdf", False)
    sigmoid_eff = config["fit"].get("sigmoid_eff", False)
    interactive = config["fit"].get("interactive", False)
    args_bin = config["fit"].get("bin", "bin0")
    if isinstance(args_bin, str):
        args_bin = [args_bin]
    x_min = config["fit"].get("x_min", None)
    x_max = config["fit"].get("x_max", None)
    info = config["info_level"]

    all_results = []
    data_msg_per_bin = []
    mc_msg_per_bin = []
    data_msg = []
    mc_msg = []
    data_eff_per_bin, data_err_per_bin, mc_eff_per_bin, mc_err_per_bin, sf_per_bin, sf_err_per_bin = [], [], [], [], [], []
    output_tables_data = defaultdict(list)
    output_progs_data = defaultdict(list)
    output_texts_data = defaultdict(list)
    output_tables_mc = defaultdict(list)
    output_progs_mc = defaultdict(list)
    output_texts_mc = defaultdict(list)
    all_pt_bins = []
    console = Console()


    bins_progress = Progress(
        TextColumn("[bold magenta]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),  # <-- Only here
        console=console,
    )

    # Sub progress bars (DATA/MC per bin)
    sub_progress = Progress(
        TextColumn("[cyan]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        CustomTimeElapsedColumn(),
        console=console,
    )
        
    progress_group = Group(bins_progress, sub_progress)


    with Live(progress_group, refresh_per_second=5, console=console):
        total_fits = len(args_bin) * (len(root_files_DATA) + len(root_files_MC))
        task_bins = bins_progress.add_task("Fitting bins", total=total_fits)

        for pt in args_bin:

            data_eff_list = []
            data_err_list = []
            mc_eff_list = []
            mc_err_list = []
            sf_list = []
            sf_err_list = []
            data_msg_parts = []
            mc_msg_parts = []

            has_data = len(root_files_DATA) > 0
            has_mc = len(root_files_MC) > 0

            task_data = None
            task_mc = None

            if has_data:
                task_data = sub_progress.add_task(f"    DATA ({pt})", total=len(root_files_DATA))
            else:
                sub_progress.console.print(f"[yellow]No DATA files to fit for {pt}")

            if has_mc:
                task_mc = sub_progress.add_task(f"    MC   ({pt})", total=len(root_files_MC))
            else:
                sub_progress.console.print(f"[yellow]No MC files to fit for {pt}")

            hist_pass_name = config["fit"].get("hist_pass_name")
            hist_fail_name = config["fit"].get("hist_fail_name")
            if not hist_pass_name or not hist_fail_name:
                bin_suffix, bin_range = BINS_INFO[pt]
                if mass in ["Z", "JPsi"]:
                    hist_pass_name = f"{pt}_{bin_suffix}_Pass"
                    hist_fail_name = f"{pt}_{bin_suffix}_Fail"
                else:
                    abseta = config["fit"].get("abseta", 1)
                    pt_hist = int(pt.split("bin")[1]) + 1  # bin0->1, bin1->2, etc to match ROOT file naming
                    num = config["fit"].get("numerator", "gold")
                    den = config["fit"].get("denominator", "baselineplus")
                    hist_pass_name = f"NUM_{num}_DEN_{den}_abseta_{abseta}_pt_{pt_hist}_Pass"
                    hist_fail_name = f"NUM_{num}_DEN_{den}_abseta_{abseta}_pt_{pt_hist}_Fail"

            data_eff_dict = {}
            data_err_dict = {}
            mc_eff_dict = {}
            mc_err_dict = {}

            data_items = list(root_files_DATA.items())
            mc_items = list(root_files_MC.items())
            if len(mc_items) > 1:
                mc_items = mc_items[:1]

            data_eff_all = {}
            data_err_all = {}
            mc_eff_all = {}
            mc_err_all = {}

            # Initialize data variables
            data_fit_results = None
            data_eff = None
            data_err = None
            data_key = None
            style_data = "white"

            # Fit DATA once (assuming single data file)
            data_fit_results = None
            if has_data and len(data_items) == 1:
                data_key, data_path = data_items[0]
                sub_progress.update(task_data, description=f"    [cyan]DATA ({pt}): [yellow]{data_key}")

                with uproot.open(data_path) as f:
                    h_pass = load_histogram(f, hist_pass_name, "DATA")
                    h_fail = load_histogram(f, hist_fail_name, "DATA")

                data_fit_results = fit_function(fit_type, h_pass, h_fail, use_cdf=use_cdf, args_bin=pt, args_data="DATA", args_mass=mass,
                    sigmoid_eff=sigmoid_eff, interactive=interactive, x_min=x_min, x_max=x_max, 
                    data_name=data_key, mc_name=None, separate_signal_shape=separate_signal_shape)

                data_fit_results['data_key'] = data_key
                data_fit_results['data_type'] = 'DATA'

                data_eff_dict[data_key] = data_fit_results["popt"]["epsilon"]
                data_err_dict[data_key] = data_fit_results["perr"]["epsilon"]

                plot_path = Path(config["output"]["plot_dir"]) / "DATA" / pt
                plot_path.mkdir(parents=True, exist_ok=True)

                if config["output"].get("plot_dir"):
                    abseta = config["fit"].get("abseta", 1)
                    barrel_name = {1: "bar1", 2: "bar2", 3: "end", 4: "end2"}.get(abseta, f"bar{abseta}")
                    plot_path = Path(config["output"]["plot_dir"]) / "DATA" / pt / barrel_name
                    plot_path.mkdir(parents=True, exist_ok=True)
                    fig_pass, fig_fail = plot_combined_fit(data_fit_results, plot_path, data_type="DATA", sigmoid_eff=sigmoid_eff, 
                                                           args_mass=mass, separate_signal_shape=separate_signal_shape)
                    fig_pass.savefig(plot_path / f"{data_key}_Pass.png", bbox_inches="tight", dpi=300)
                    fig_fail.savefig(plot_path / f"{data_key}_Fail.png", bbox_inches="tight", dpi=300)
                    plt.close(fig_pass)
                    plt.close(fig_fail)
                    plt.close('all')

                output_tables_data[pt].append(data_fit_results.get("sum_table", ""))
                output_progs_data[pt].append(data_fit_results.get("sum_prog", ""))
                output_texts_data[pt].append(data_fit_results.get("sum_text", ""))

                all_results.append(data_fit_results)

                eps, err = data_fit_results["popt"]["epsilon"], data_fit_results["perr"]["epsilon"]
                data_eff, data_err = eps, err
                data_msg = f"{data_key} fit {'passed' if data_fit_results['message'].is_valid else 'failed'}"
                data_msg_parts.append(data_msg)
                
                style_data = "green" if data_fit_results["message"].is_valid else "red"
                sub_progress.update(task_data, advance=1, style=style_data)
                bins_progress.update(task_bins, advance=1)

                data_eff_list.append(data_eff)
                data_err_list.append(data_err)

            # Process each MC file and create combined plots with the data fit
            for mc_key, mc_path in mc_items:
                if mc_key is not None and has_mc:
                    sub_progress.update(task_mc, description=f"    [cyan]MC   ({pt}): [yellow]{mc_key}")

                    with uproot.open(mc_path) as f:
                        h_pass = load_histogram(f, hist_pass_name, "MC")
                        h_fail = load_histogram(f, hist_fail_name, "MC")

                    res_mc = fit_function(fit_type, h_pass, h_fail, use_cdf=use_cdf, args_bin=pt, args_data="MC", args_mass=mass,
                        sigmoid_eff=sigmoid_eff, interactive=interactive, x_min=x_min, x_max=x_max, 
                        data_name=None, mc_name=mc_key, separate_signal_shape=separate_signal_shape)

                    res_mc['mc_key'] = mc_key
                    res_mc['data_key'] = data_key if data_fit_results else None
                    res_mc['data_type'] = 'MC'

                    mc_eff_dict[mc_key] = res_mc["popt"]["epsilon"]
                    mc_err_dict[mc_key] = res_mc["perr"]["epsilon"]

                    plot_path = Path(config["output"]["plot_dir"]) / "MC" / pt
                    plot_path.mkdir(parents=True, exist_ok=True)

                    if config["output"].get("plot_dir"):
                        abseta = config["fit"].get("abseta", 1)
                        barrel_name = {1: "bar1", 2: "bar2", 3: "end", 4: "end2"}.get(abseta, f"bar{abseta}")
                        plot_path = Path(config["output"]["plot_dir"]) / "MC" / pt / barrel_name
                        plot_path.mkdir(parents=True, exist_ok=True)
                        fig_pass, fig_fail = plot_combined_fit(res_mc, plot_path, data_type="MC", sigmoid_eff=sigmoid_eff, 
                                                               args_mass=mass, separate_signal_shape=separate_signal_shape)
                        fig_pass.savefig(plot_path / f"{mc_key}_Pass.png", bbox_inches="tight", dpi=300)
                        fig_fail.savefig(plot_path / f"{mc_key}_Fail.png", bbox_inches="tight", dpi=300)
                        plt.close(fig_pass)
                        plt.close(fig_fail)
                        plt.close('all')

                    # Create combined 4-panel plot with data and MC
                    if data_fit_results and config["output"].get("plot_dir"):
                        numerator = config["fit"].get("numerator", "unknown").lower()
                        # Map numerator values to their shorthand versions
                        numerator_map = {
                            "baselineplus": "blp",
                            "blp": "blp",
                            "goldid": "goldID",
                            "iso": "iso",
                            "prompt": "prompt",
                            "trackermuons": "blp",  # default shorthand
                        }
                        numerator = numerator_map.get(numerator, numerator)
                        # Shorten mass name for filename
                        mass_short = mass.replace("_muon", "").replace("_", "")
                        save_fits_and_plots(pt, 
                                          data_fit_results["popt"]["epsilon"], data_fit_results["perr"]["epsilon"],
                                          res_mc["popt"]["epsilon"], res_mc["perr"]["epsilon"],
                                          data_key, mc_key, 
                                          outdir=config["output"]["plot_dir"],
                                          abseta=config["fit"].get("abseta", 1),
                                          mass=mass_short,
                                          year=year,
                                          numerator=numerator)

                    output_tables_mc[pt].append(res_mc.get("sum_table", ""))
                    output_progs_mc[pt].append(res_mc.get("sum_prog", ""))
                    output_texts_mc[pt].append(res_mc.get("sum_text", ""))

                    all_results.append(res_mc)

                    eps, err = res_mc["popt"]["epsilon"], res_mc["perr"]["epsilon"]
                    mc_eff, mc_err = eps, err
                    mc_msg = f"{mc_key} fit {'passed' if res_mc['message'].is_valid else 'failed'}"
                    mc_msg_parts.append(mc_msg)
                    
                    style_mc = "green" if res_mc["message"].is_valid else "red"
                    sub_progress.update(task_mc, advance=1, style=style_mc)
                    bins_progress.update(task_bins, advance=1)
                
                mc_eff_list.append(mc_eff)
                mc_err_list.append(mc_err)
                
                # Calculate scale factor for this data-MC pair
                if data_eff is not None and mc_eff is not None and mc_eff != 0:
                    sf_val = data_eff / mc_eff
                    sf_err_val = np.sqrt((data_err / data_eff)**2 + (mc_err / mc_eff)**2) * sf_val
                else:
                    sf_val = None
                    sf_err_val = None
                
                sf_list.append(sf_val)
                sf_err_list.append(sf_err_val)

                if task_data is not None and task_data in sub_progress.task_ids:
                    sub_progress.update(task_data, description=f"    [bold]DATA ({pt}): [{style_data}]{data_key}")
                if task_mc is not None and task_mc in sub_progress.task_ids:
                    sub_progress.update(task_mc, description=f"    [bold]MC   ({pt}): [{style_mc}]{mc_key}")

            # Auto-call save_to_excel for one DATA + two MC
            if data_fit_results and len(mc_items) >= 1:
                epsilon_data = data_eff
                epsilon_err_data = data_err
                epsilon_mc = mc_eff_list[0] if mc_eff_list else None
                epsilon_err_mc = mc_err_list[0] if mc_err_list else None
                epsilon_data_2 = mc_eff_list[1] if len(mc_eff_list) > 1 else None
                epsilon_err_data_2 = mc_err_list[1] if len(mc_err_list) > 1 else None
                scale_factor = sf_list[0] if sf_list else None
                scale_factor_err = sf_err_list[0] if sf_err_list else None
                scale_factor2 = sf_list[1] if len(sf_list) > 1 else None
                scale_factor2_err = sf_err_list[1] if len(sf_err_list) > 1 else None
                args_bin = pt
                barrel = _coerce_excel_value(config["fit"].get("abseta", 1))
                args_type = data_key
                args_type2 = list(root_files_MC.keys())[0] if root_files_MC else None
                type_param = "DATA"
                num_den = f"{config['fit'].get('numerator', 'baselineplus')}_{config['fit'].get('denominator', 'TrackerMuons')}"
                mass_param = mass
                save_to_excel(data_fit_results, [res for res in all_results if res.get('data_type') == 'MC'], epsilon_data, epsilon_err_data, scale_factor, scale_factor_err, args_bin, barrel, args_type, args_type2, type_param, num_den, mass_param, epsilon_mc, epsilon_err_mc)

            data_msg_per_bin.append(data_msg_parts)
            mc_msg_per_bin.append(mc_msg_parts)
            
            # Repeat data efficiencies to match MC length for summary
            repeated_data_eff = data_eff_list * len(mc_eff_list) if data_eff_list else [None] * len(mc_eff_list)
            repeated_data_err = data_err_list * len(mc_err_list) if data_err_list else [None] * len(mc_err_list)
            
            data_eff_per_bin.append(repeated_data_eff)
            data_err_per_bin.append(repeated_data_err)
            mc_eff_per_bin.append(mc_eff_list)
            mc_err_per_bin.append(mc_err_list)
            sf_per_bin.append(sf_list)
            sf_err_per_bin.append(sf_err_list)
            
            all_pt_bins.append(pt)


    # Summary logging
    if info == "INFO_2":
        logging.info("\nFit Summary:")
        for pt in all_pt_bins:
            for text_data, text_mc in zip(output_texts_data[pt], output_texts_mc[pt]):
                console.print(text_data)
                console.print(text_mc)

    # Final SF output summary
    print_efficiency_summary(all_pt_bins, data_msg_per_bin, mc_msg_per_bin,
                             data_eff_per_bin, data_err_per_bin,
                             mc_eff_per_bin, mc_err_per_bin,
                             sf_per_bin, sf_err_per_bin)

    # If explicit scale_factors are in config, use the per-bin last-seen values (falls back to None)
    if config.get("scale_factors", {}).get("data_mc_pair"):
        table = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        table.add_column("Pair Name", style="cyan", justify="right", no_wrap=True)
        table.add_column("Bin", style="green", justify="center", no_wrap=True)
        table.add_column("Efficiency Summary", style="white", justify="left")

        for pair_name, (data_key, mc_key) in config["scale_factors"]["data_mc_pair"].items():
            for pt in all_pt_bins:
        
                data_eff = data_eff_dict.get(data_key)
                data_err = data_err_dict.get(data_key)
                mc_eff = mc_eff_dict.get(mc_key)
                mc_err = mc_err_dict.get(mc_key)

                if None in (data_eff, data_err, mc_eff, mc_err):
                    eff_line = f"DATA: N/A | MC: N/A | SF: N/A"
                    table.add_row(pair_name, str(pt), eff_line)
                    continue

                sf = data_eff / mc_eff if mc_eff != 0 else None
                sf_err = (
                    np.sqrt((data_err / data_eff) ** 2 + (mc_err / mc_eff) ** 2) * sf
                    if (sf and data_eff and mc_eff) else None
                )

                if sf is not None:
                    eff_line = (
                        f"DATA: {data_eff:.5f} ± {data_err:.5f} | "
                        f"MC: {mc_eff:.5f} ± {mc_err:.5f} | "
                        f"SF: {sf:.5f} ± {sf_err:.5f}"
                    )
                else:
                    eff_line = (
                        f"DATA: {data_eff:.5f} ± {data_err:.5f} | "
                        f"MC: {mc_eff:.5f} ± {mc_err:.5f} | "
                        f"SF: N/A (MC=0)"
                    )

                table.add_row(pair_name, str(pt), eff_line)

        console.print(
            Panel.fit(
                table,
                title="[bold yellow]Explicit Scale Factors (Per Bin)[/bold yellow]",
                border_style="bright_blue",
                padding=(1, 2),
            )
        )

    # Save fit results to JSON if configured (original behavior)
    results_file_name = str(config["output"].get("results_file", "")).strip()
    if results_file_name:
        if not os.path.isabs(results_file_name):
            results_file_name = os.path.join(config["output"].get("plot_dir", "."), results_file_name)

        # Convert results to plain python types for serialization
        def convert_types(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_types(v) for v in obj]
            return obj

        serializable_results = []
        keys_to_remove = ['m', 'message', 'cov', 'sum_text']
        for res in all_results:
            res_copy = res.copy()
            for key in keys_to_remove:
                res_copy.pop(key, None)
            serializable_results.append(convert_types(res_copy))

        try:
            Path(results_file_name).parent.mkdir(parents=True, exist_ok=True)
            with open(results_file_name, "w") as f:
                json.dump(serializable_results, f, indent=2)
            print(f"Saved combined fit results to {results_file_name}")
        except Exception as e:
            print(f"Error saving results to JSON: {e}")
    else:
        print("No results_file configured; skipping writing JSON results")


def save_to_excel(results_data, results_mc, epsilon_data, epsilon_err_data,
                 scale_factor, scale_factor_err,
                 args_bin, barrel, args_type, args_type2, type, num_den, mass, epsilon_mc, epsilon_err_mc,
                 filename="/Users/neilj/Desktop/python1/Physics/ISO_plots/sfs_ecol.xlsx"):
    """
    Save fitting results with all DATA entries first, then a single MC entry.
    Updates existing entries in their original positions if they match the same bin, barrel, and type.
    """
    # Create the new entries
    new_data_row = {
        "Type": "DATA",
        "bin": args_bin,
        "barrel": barrel,
        "num_den": num_den,
        "fit_type": args_type,
        "fit type, MC": args_type2,
        "mass": mass,
        "epsilon": epsilon_data,
        "epsilon_err": epsilon_err_data,
        "SF": None,
        "SF_err": None
    }
    
    new_mc_row = {
        "Type": "MC",
        "bin": args_bin,
        "barrel": barrel,
        "num_den": num_den,
        "fit_type Data": args_type,
        "fit type, MC": args_type2,
        "mass": mass,
        "epsilon": epsilon_mc,
        "epsilon_err": epsilon_err_mc,
        "SF": scale_factor,
        "SF_err": scale_factor_err
    }

    key_cols = ['Type', 'bin', 'barrel', 'num_den', 'mass']

    if os.path.exists(filename):
        existing_df = pd.read_excel(filename, sheet_name='ScaleFactors')

        # Keep existing barrel values numeric so older entries don't stay as text.
        if 'barrel' in existing_df.columns:
            existing_df['barrel'] = existing_df['barrel'].apply(_coerce_excel_value)

        # Drop any previous entries with the same identifying parameters so only the latest remains
        match_mask = (
            (existing_df['bin'] == args_bin) &
            (existing_df['barrel'] == barrel) &
            (existing_df['num_den'] == num_den) &
            (existing_df['mass'] == mass)
        )
        existing_df = existing_df[~match_mask]

        updated_df = pd.concat([existing_df, pd.DataFrame([new_data_row, new_mc_row])], ignore_index=True)
    else:
        updated_df = pd.DataFrame([new_data_row, new_mc_row])

    updated_df.to_excel(filename, sheet_name='ScaleFactors', index=False)
    print(f"Results saved/updated in {filename}")


if __name__ == "__main__":
    main()