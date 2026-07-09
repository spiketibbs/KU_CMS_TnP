from __future__ import annotations

import os
import argparse

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def create_subplots_for_bin(bin_name, fit_type):
    # Define the exact directory structure
    base_dir = f"{bin_name}_fits"
    data_dirs = {
        "MC_DY": os.path.join(base_dir, "MC"),
        "MC_DY2_2L_2J": os.path.join(base_dir, "MC"),
        "MC_DY2_2L_4J": os.path.join(base_dir, "MC"),
    }

    # Create output directory for combined plots
    output_dir = os.path.join(base_dir, "combined_plots")
    os.makedirs(output_dir, exist_ok=True)

    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(2, 3, figure=fig)

    # Track if we found any files
    files_found = False

    # Define plot positions and titles
    plot_config = [
        {"data_type": "MC_DY", "category": "pass", "position": (0, 0), "title": "MC OLD Pass"},
        {"data_type": "MC_DY", "category": "fail", "position": (1, 0), "title": "MC OLD Fail"},
        {"data_type": "MC_DY2_2L_2J", "category": "pass", "position": (0, 1), "title": "MC_DY_2L_2J Pass"},
        {"data_type": "MC_DY2_2L_2J", "category": "fail", "position": (1, 1), "title": "MC_DY_2L_2J Fail"},
        {"data_type": "MC_DY2_2L_4J", "category": "pass", "position": (0, 2), "title": "MC_DY_2L_4J Pass"},
        {"data_type": "MC_DY2_2L_4J", "category": "fail", "position": (1, 2), "title": "MC_DY_2L_4J Fail"},
    ]

    for config in plot_config:
        data_type = config["data_type"]
        category = config["category"]
        row, col = config["position"]
        title = config["title"]

        hist_suffix = category.capitalize()  # "Pass" or "Fail"
        filename = f"{data_type}_barrel_1_tag_{fit_type}_fit_{bin_name}_{hist_suffix}.png"
        filepath = os.path.join(data_dirs[data_type], filename)

        ax = fig.add_subplot(gs[row, col])

        if os.path.exists(filepath):
            files_found = True
            try:
                img = mpimg.imread(filepath)
                ax.imshow(img)
                ax.set_title(title, fontsize=25, pad=10)
            except Exception as e:
                ax.text(0.5, 0.5, f"Error loading:\n{os.path.basename(filepath)}", ha="center", va="center", fontsize=10)
                ax.set_title(f"{title} (corrupted)", fontsize=14, pad=10)
                print(f"Warning: Could not read {filepath} - {e!s}")
        else:
            ax.text(0.5, 0.5, f"File not found:\n{filename}", ha="center", va="center", fontsize=10)
            ax.set_title(f"{title} (missing)", fontsize=14, pad=10)
            print(f"Warning: File not found - {filepath}")

        ax.axis("off")

    if files_found:
        bin_number = bin_name.replace("bin", "")
        pt_range = get_pt_range(bin_name)
        fig.suptitle(f"Bin {bin_number} ({pt_range})\nFit Type: {fit_type.replace('_', ' ').title()}\nBARREL 1", fontsize=20, y=0.98)

        # === Layout first ===
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)

        ef1, ef1_err = 0.963091, 0.000010
        ef2, ef2_err = 0.964445, 0.000006
        ef3, ef3_err = 0.964492, 0.000007


        # === Then add labeled blank text boxes ===
        top_labels = ["MC OLD", "MC_DY_2L_2J", "MC_DY_2L_4J"]
        box_text = [f"EFF = {ef1} ± {ef1_err}", f"EFF = {ef2} ± {ef2_err}", f"EFF = {ef3} ± {ef3_err}"]
        x_positions = [0.17, 0.5, 0.83]  # Centered above each column

        for label, xpos, bt in zip(top_labels, x_positions, box_text):
            fig.text(xpos, 0.91, f"{label}:\n{bt}", ha="center", va="top", fontsize=14,
                    bbox=dict(boxstyle="round", edgecolor="black", facecolor="white"))
            
        output_filename = os.path.join(output_dir, f"{bin_name}_{fit_type}_combined.png")
        plt.savefig(output_filename, bbox_inches="tight", dpi=150)
        print(f"Successfully created combined plot:\n{output_filename}")
    else:
        print(f"No valid plot files found for {bin_name} with fit type {fit_type}")

    plt.close()


def get_pt_range(bin_name):
    pt_ranges = {
        "bin0": "5.00-7.00 GeV",
        "bin1": "7.00-10.00 GeV",
        "bin2": "10.00-20.00 GeV",
        "bin3": "20.00-45.00 GeV",
        "bin4": "45.00-75.00 GeV",
        "bin5": "75.00-100.00 GeV",
        "bin6": "100.00-500.00 GeV",
    }
    return pt_ranges.get(bin_name, "Unknown Range")


if __name__ == "__main__":
    available_bins = [f"bin{i}" for i in range(7)]
    available_fit_types = [
        "dcb_ps", "dcb_lin", "dcb_exp", "dcb_cheb",
        "dv_ps", "dv_lin", "dv_exp", "dv_cheb",
        "dg_ps", "dg_lin", "dg_exp", "dg_cheb"
    ]

    parser = argparse.ArgumentParser(
        description="Combine Z mass fit plots into 2x3 grid (MC × Pass/Fail)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--bin", type=str, required=True, choices=available_bins, help="Which pt bin to process (e.g., bin0, bin1)")
    parser.add_argument("--type", type=str, required=True, choices=available_fit_types, help="Which fit type to combine")
    parser.add_argument("--output-dir", type=str, default="combined_plots", help="Subdirectory name for output plots")

    args = parser.parse_args()

    print(f"\nProcessing {args.bin} with fit type {args.type}...")
    create_subplots_for_bin(args.bin, args.type)
