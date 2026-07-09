import uproot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

# Define all file paths
file_paths = {
    "DATA_barrel_1": "/uscms/home/hortua/nobackup/egamma-tnp/examples/nanoaod_filters_custom/blp_3gt/DATA_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_23D_histos_pt_barrel_1.root",
    "DATA_barrel_2": "/uscms/home/hortua/nobackup/egamma-tnp/examples/nanoaod_filters_custom/blp_3gt/DATA_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_23D_histos_pt_barrel_2.root",
    "DATA_endcap": "/uscms/home/hortua/nobackup/egamma-tnp/examples/nanoaod_filters_custom/blp_3gt/DATA_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_23D_histos_pt_endcap.root",
}

# Open all files
files = {key: uproot.open(path) for key, path in file_paths.items()}

def find_hist_name(file, bin_id, pass_fail):
    """Find histogram name matching bin ID and pass/fail status."""
    for key in file.keys():
        if key.startswith(bin_id) and f"_{pass_fail};1" in key:
            return key
    raise ValueError(f"No '{pass_fail}' histogram found for bin ID '{bin_id}'")

def plot_comparison(bin_id, output_dir="bin_plots"):
    """Create a 2x3 grid of plots for a given bin ID."""
    plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, hspace=0.3, wspace=0.3)
    
    regions = [
        ("barrel_1", "Barrel 1"),
        ("barrel_2", "Barrel 2"), 
        ("endcap", "Endcap")
    ]
    
    # Plot Pass histograms (top row)
    for col, (region, title) in enumerate(regions):
        ax = plt.subplot(gs[0, col])
        plot_histogram(ax, bin_id, region, "Pass")
        ax.set_title(f"{title} - Pass", pad=20)
    
    # Plot Fail histograms (bottom row)
    for col, (region, title) in enumerate(regions):
        ax = plt.subplot(gs[1, col])
        plot_histogram(ax, bin_id, region, "Fail")
        ax.set_title(f"{title} - Fail", pad=20)
    
    # Add overall title with bin range
    bin_ranges = {
        "bin0": "5-7 GeV",
        "bin1": "7-10 GeV",
        "bin2": "10-20 GeV",
        "bin3": "20-45 GeV",
        "bin4": "45-75 GeV",
        "bin5": "75-500 GeV",
    }
    bin_range = bin_ranges.get(bin_id, bin_id)
    plt.suptitle(f"DATA Histograms for {bin_id} ({bin_range})", y=0.98, fontsize=25)
    
    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/DATA_{bin_id}_plot.png", bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_dir}/DATA_{bin_id}_plot.png")

def plot_histogram(ax, bin_id, region, pass_fail):
    """Plot histogram for a specific region and pass/fail status."""
    file_key = f"DATA_{region}"
    
    try:
        # Get histogram name
        hist_name = find_hist_name(files[file_key], bin_id, pass_fail)
        
        # Load histogram
        hist = files[file_key][hist_name]
        
        # Get plot data
        edges = hist.axis().edges()
        centers = 0.5 * (edges[:-1] + edges[1:])
        values = hist.values()
        
        # Plot histogram
        ax.step(centers, values, where="mid", linewidth=2)
        
        # Format plot
        ax.set_xlabel("pT [GeV]")
        ax.set_ylabel("Counts")
        ax.grid(True)
        
    except Exception as e:
        ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
        ax.set_title(f"Error in {region} {pass_fail}")

# Create plots for all bins (bin0 to bin6)
for bin_id in [f"bin{i}" for i in range(7)]:
    plot_comparison(bin_id)