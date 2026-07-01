import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import mplhep
from matplotlib.colors import LinearSegmentedColormap

mplhep.style.use(mplhep.style.CMS)

# ==============================================================================
# CONFIGURATION SWITCH
# Toggle options below to automatically swap all tracking items.
# ==============================================================================
LEPTON_TYPE = 'Electron'  # Options: 'Electron' or 'Muon'
HEATMAP_TYPE = "DATA"  # Options: 'DATA' or 'MC' (affects which rows are read from the Excel file)
NUM = 'Not Prompt'  # Options: 'GoldID', 'ISO', 'Prompt', 'BLP', 'Not Prompt', or 'Not ID nor ISO'
ERA = "2023 PreBPix" # Electron Options: '2016', '2017', '2018', '2022 PreEE', '2022 PostEE', '2023 PreBPix', '2023 PostBPix'"
                     # Muon Options: '2016', '2017', '2018', '2022 PreEE', '2022 PostEE', '2023 PreBPix', '2023 PostBPix'"
# Clean up ERA and NUM strings by replacing spaces with underscores for filenames
ERA_clean = ERA.replace(" ", "_")
NUM_clean = NUM.replace(" ", "_")

LEPTON_CONFIGS = {
    'Electron': {
        # Modify the excel path to match the correct file for electrons
        'excel_path': f"Electrons_{NUM_clean}/2023_pre_NOT_Prompt.xlsx",
        # Change the pt_labels to '2-4' and '4-7' if 2016-2018, or '2-5' and '5-7' if 2022-2023
        'pt_labels': ['2-4', '4-7', '7-10', '10-20', '20-45', '45-75', '75-500']
        if ERA in ["2016 preVFP", "2016 post VFP", "2017", "2018"]
        else ['2-5', '5-7', '7-10', '10-20', '20-45', '45-75', '75-500'],
        'eta_labels': [1, 2, 3],
        'eta_pretty_labels': ['|η| ≤ 0.8', '0.8 < |η| ≤ 1.442', '1.556 < |η| ≤ 2.5'],
        'title': f"EFF vs. $p_T$ and η Region ({HEATMAP_TYPE} {LEPTON_TYPE} {NUM} {ERA})",
        'save_path': f"Electrons_{NUM_clean}/{HEATMAP_TYPE}_EFF_{LEPTON_TYPE}_{NUM_clean}_{ERA_clean}_HEATMAP.pdf"
    },
    'Muon': {
        # Modify the excel path to match the correct file for muons
        'excel_path': f"Muons_{NUM_clean}/2023_pre_NOT_Prompt_Muon.xlsx", 
        'pt_labels': ['3-6', '6-10', '10-20', '20-45', '45-500'], 
        'eta_labels': [1, 2, 3, 4],
        'eta_pretty_labels': ['|η| ≤ 0.9', '0.9 < |η| ≤ 1.2', '1.2 < |η| ≤ 2.1', '2.1 < |η| ≤ 2.4'], 
        'title': f"EFF vs. $p_T$ and η Region ({HEATMAP_TYPE} {LEPTON_TYPE} {NUM} {ERA})",
        'save_path': f"Muons_{NUM_clean}/{HEATMAP_TYPE}_EFF_{LEPTON_TYPE}_{NUM_clean}_{ERA_clean}_HEATMAP.pdf"
    }
}

# Pull current active dictionary selection
config = LEPTON_CONFIGS[LEPTON_TYPE]
pt_labels = config['pt_labels']
eta_labels = config['eta_labels']
eta_pretty_labels = config['eta_pretty_labels']

# ==============================================================================
# DATA LOADING & GRID CONSTRUCTION
# ==============================================================================
df = pd.read_excel(config['excel_path'], sheet_name='ScaleFactors')

# Filter only DATA or only MC rows for efficiencies
df_mc = df[df['Type'] == f'{HEATMAP_TYPE}'].copy()

# Parse integer from bin naming syntax (e.g., 'bin3' -> 3)
df_mc['bin_num'] = df_mc['bin'].str.extract(r'bin(\d+)').astype(int)

# Dynamic allocation size based on lepton types
heatmap_values = np.full((len(pt_labels), len(eta_labels)), np.nan)
heatmap_errors = np.full((len(pt_labels), len(eta_labels)), np.nan)

# Map raw series data into array grids
for _, row in df_mc.iterrows():
    bin_num = row['bin_num'] 
    eta_region = row['barrel']
    
    try:
        col_idx = eta_labels.index(eta_region)
    except ValueError:
        continue
    
    row_idx = bin_num
    if row_idx < 0 or row_idx >= len(pt_labels):
        continue
        
    heatmap_values[row_idx, col_idx] = row['epsilon']
    heatmap_errors[row_idx, col_idx] = row['epsilon_err']

# ==============================================================================
# MASKING LOGIC 
# Drops unmeasured placeholder dummy data bins (Value = 1.0, Error = 0.0)
# ==============================================================================
mask = np.isclose(heatmap_values, 1.0) & np.isclose(heatmap_errors, 0.0)
heatmap_values[mask] = np.nan

# ==============================================================================
# PLOTTING & STYLING
# ==============================================================================
colors_list = [(0, "red"), (1, "yellow")]
cmap = LinearSegmentedColormap.from_list("spike_at_one", colors_list)
cmap.set_bad(color='white')

norm = colors.Normalize(vmin=np.nanmin(heatmap_values), vmax=np.nanmax(heatmap_values))

fig, ax = plt.subplots(figsize=(10, 6))  
im = ax.imshow(heatmap_values, cmap=cmap, norm=norm, aspect='auto')

# Apply formatted dynamic ticks
ax.set_xticks(np.arange(len(eta_pretty_labels)))
ax.set_yticks(np.arange(len(pt_labels)))
ax.set_xticklabels(eta_pretty_labels, fontsize=12)
ax.set_yticklabels(pt_labels, fontsize=12)
ax.set_xlabel('η Region', fontsize=14)
ax.set_ylabel('$p_T$ [GeV]', fontsize=14)
plt.title(config['title'], fontsize=16)
ax.invert_yaxis()

# Label numerical values on matrix cells
for i in range(len(pt_labels)):
    for j in range(len(eta_pretty_labels)):
        val = heatmap_values[i, j]
        err = heatmap_errors[i, j]
        if np.isnan(val):
            ax.text(j, i, "  ", ha="center", va="center", color="black", fontsize=10)
        else:
            text = f"{val:.4f} ± {err:.4f}"
            ax.text(j, i, text, ha="center", va="center", color="black", fontsize=10)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label("EFF", fontsize=12)

plt.tight_layout()

# Safe directory creation before plot export
dir_name = os.path.dirname(config['save_path'])
if dir_name:
    os.makedirs(dir_name, exist_ok=True)

plt.savefig(config['save_path'])
print(f"Generated plot exported to: {config['save_path']}")
plt.show()