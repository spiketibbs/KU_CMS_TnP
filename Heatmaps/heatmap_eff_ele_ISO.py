import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import FormatStrFormatter
import mplhep

# Apply the CMS style
mplhep.style.use(mplhep.style.CMS)

# ---------------------------------------------------------
# MANUAL DATA ENTRY
# ---------------------------------------------------------
# Define the order of the eras as they should appear from top to bottom
eras_ordered = [
    '2026',
    '2025',
    '2024',
    '2023 postBPix',
    '2023 preBPix',
    '2022 postEE',
    '2022 preEE',
    '2018',
    '2017',
    '2016 postVFP',
    '2016 preVFP'
]
HEATMAP_TYPE = "DATA"  # Options: 'DATA' or 'MC' (affects naming of heatmap file)

# Column definitions (eta and pT regions matching the plot)
eta_labels = [
    '$|\eta| \leq 0.8$\n($7 < p_T < 20$)', 
    '$0.8 < |\eta| \leq 1.442$\n($7 < p_T < 20$)', 
    '$1.556 < |\eta| \leq 2.5$\n($10 < p_T < 20$)'
]

# Enter your manual data here: { 'EraName': [(val1, err1), (val2, err2), (val3, err3)] }
# Populated with values rounded to 4 decimals from your DATA efficiency image.
data_dict = {
    '2026':          [(0.9643, 0.0013), (0.9680, 0.0037), (0.9218, 0.0009)],
    '2025':          [(0.9321, 0.0024), (0.9432, 0.0029), (0.9545, 0.0027)],
    '2024':          [(0.9643, 0.0013), (0.9680, 0.0037), (0.9218, 0.0009)],
    '2023_postBPix': [(0.9321, 0.0024), (0.9432, 0.0029), (0.9545, 0.0027)],
    '2023_preBPix':  [(0.9267, 0.0034), (0.9358, 0.0031), (0.9569, 0.0026)],
    '2022 postEE':   [(0.8618, 0.0053), (0.9539, 0.0072), (0.9511, 0.0029)],
    '2022 preEE':    [(0.8467, 0.0076), (0.8633, 0.0076), (0.9458, 0.0039)],
    '2018':          [(0.9252, 0.0014), (0.9287, 0.0019), (0.9532, 0.0015)],
    '2017':          [(0.9502, 0.0009), (0.9548, 0.0010), (0.9703, 0.0017)],
    '2016 postVFP':  [(0.9524, 0.0019), (0.9611, 0.0022), (0.9695, 0.0024)],
    '2016 preVFP':   [(0.9439, 0.0020), (0.9539, 0.0027), (0.9627, 0.0027)]
}

# ---------------------------------------------------------
# MATRIX PREPARATION
# ---------------------------------------------------------
num_eras = len(eras_ordered)
num_eta = len(eta_labels)

heatmap_values = np.zeros((num_eras, num_eta))
heatmap_errors = np.zeros((num_eras, num_eta))

# Build matrices based on the strict row order defined in eras_ordered
for i, era in enumerate(eras_ordered):
    for j in range(num_eta):
        heatmap_values[i, j] = data_dict[era][j][0]
        heatmap_errors[i, j] = data_dict[era][j][1]

# ---------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 8))

# Matplotlib built-in 'autumn' or 'Wistia_r' closely match the CMS Red-Yellow scale
vmin = np.min(heatmap_values)
vmax = np.max(heatmap_values)
norm = colors.Normalize(vmin=vmin, vmax=vmax)

im = ax.imshow(heatmap_values, cmap='autumn', norm=norm, aspect='auto')

# Grid lines and structural tweaks to look exactly like standard ROOT/CMS heatmaps
ax.set_xticks(np.arange(num_eta))
ax.set_yticks(np.arange(num_eras))
ax.set_xticklabels(eta_labels, fontsize=14)
ax.set_yticklabels(eras_ordered, fontsize=14)

# Shift tick marks to act as cell borders rather than centering
ax.set_xticks(np.arange(num_eta + 1) - 0.5, minor=True)
ax.set_yticks(np.arange(num_eras + 1) - 0.5, minor=True)
ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
ax.tick_params(which="minor", bottom=False, left=False)

# Axis Titles
ax.set_xlabel('$\eta$ Region', fontsize=16, loc='right')
ax.set_ylabel('Era', fontsize=16, loc='top')
plt.title(f"{HEATMAP_TYPE} Eff vs. Era and $\eta$ Region (Isolation)", fontsize=18, pad=10)

# Add text strings into cells
for i in range(num_eras):
    for j in range(num_eta):
        val = heatmap_values[i, j]
        err = heatmap_errors[i, j]
        
        # Display with 4 decimal places precision (.4f)
        text = f"{val:.4f} $\pm$ {err:.4f}"
            
        ax.text(j, i, text, ha="center", va="center", color="black", fontsize=11, fontweight='medium')

# Colorbar adjustments
cbar = plt.colorbar(im, ax=ax, format='%.2f')
cbar.set_label("EFF", fontsize=14, loc='top')
cbar.ax.tick_params(labelsize=16)

# Final formatting and save
plt.tight_layout()
plt.savefig(f"{HEATMAP_TYPE}_ISOLATION_HEATMAPS.pdf", bbox_inches='tight')
plt.show()
