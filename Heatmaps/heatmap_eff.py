import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import mplhep
from matplotlib.colors import LinearSegmentedColormap
import os

mplhep.style.use(mplhep.style.CMS)

eta_data = {
    "Values": {
        "Eta_1": np.array([
[0.935000000, 0.933200000, 0.898200000, 0.834100000, 0.970000000, 0.943200000, 0.970800000, 0.962400000, 0.030000000, 0.056800000, 0.160183000, 0.221617880],
[0.983000000, 0.983000000, 0.904600000, 0.840300000, 0.971700000, 0.946200000, 0.971900000, 0.964200000, 0.028300000, 0.053800000, 0.110778200, 0.173985100],
[0.988300000, 0.987245985, 0.916800000, 0.850896325, 0.975100000, 0.952552315, 0.976000000, 0.969924903, 0.024900000, 0.047447685, 0.093926560, 0.159956019],
[0.989900000, 0.989617483, 0.951000000, 0.888029032, 0.984600000, 0.968203013, 0.980300000, 0.978228251, 0.015400000, 0.031796987, 0.058605100, 0.121190945],
[0.990700000, 0.990213861, 0.956000000, 0.890816230, 0.986800000, 0.971179596, 0.981400000, 0.980147242, 0.013200000, 0.028820404, 0.052890800, 0.117901421],
        ]),  
    },       
    "Errors": {
        "Eta_1": np.array([                    
[0.001700000, 0.001000000, 0.020000000, 0.020000000, 0.020000000, 0.020000000, 0.021000000, 0.020000000, 0.020000000, 0.020000000, 0.020072120, 0.020024984],
[0.000900000, 0.000500000, 0.010000000, 0.010000000, 0.010000000, 0.010000000, 0.010000000, 0.010000000, 0.010000000, 0.010000000, 0.010040418, 0.010012492],
[0.000500000, 0.000325654, 0.001100000, 0.000705374, 0.000600000, 0.000422670, 0.000600000, 0.000425913, 0.000600000, 0.000422670, 0.001208305, 0.000776919],
[0.000100000, 0.000037697, 0.000200000, 0.000121588, 0.000100000, 0.000064732, 0.000100000, 0.000051778, 0.000100000, 0.000064732, 0.000223607, 0.000127298],
[0.000100000, 0.000056170, 0.000300000, 0.000151386, 0.000100000, 0.000088620, 0.000100000, 0.000075520, 0.000100000, 0.000088620, 0.000316228, 0.000161471],
        ]),  
    },
}

eta_MC = {
    "Values": {
        "Eta_1": np.array([
[0.929300000, 0.928200000, 0.897400000, 0.888100000, 0.954500000, 0.952400000, 0.974900000, 0.966500000, 0.045500000, 0.047600000, 0.166046180, 0.175665580],
[0.985700000, 0.984700000, 0.903600000, 0.894300000, 0.956700000, 0.954800000, 0.974900000, 0.968400000, 0.043300000, 0.045200000, 0.109321480, 0.119382790],
[0.989800000, 0.989651902, 0.917100000, 0.907321047, 0.962600000, 0.960078531, 0.972200000, 0.972528756, 0.037400000, 0.039921469, 0.092254420, 0.102068000],
[0.991600000, 0.991617427, 0.948900000, 0.939636271, 0.974000000, 0.973940880, 0.982800000, 0.982601660, 0.026000000, 0.026059120, 0.059070760, 0.068240299],
[0.992100000, 0.992255333, 0.956500000, 0.946860105, 0.977600000, 0.976312183, 0.984800000, 0.984976754, 0.022400000, 0.023687817, 0.051056350, 0.060473011],
        ]),  
    },       
    "Errors": {
        "Eta_1": np.array([                    
[0.003100000, 0.003200000, 0.020000000, 0.020000000, 0.020000000, 0.020000000, 0.020000000, 0.020000000, 0.020000000, 0.020000000, 0.020238824, 0.020254382],
[0.000900000, 0.001600000, 0.010000000, 0.010000000, 0.010000000, 0.010000000, 0.010000000, 0.010000000, 0.010000000, 0.010000000, 0.010040418, 0.010127191],
[0.000400000, 0.000256738, 0.001000000, 0.000792312, 0.000900000, 0.000478352, 0.000800000, 0.000403908, 0.000900000, 0.000478352, 0.001077033, 0.000832870],
[0.000100000, 0.000038602, 0.000200000, 0.000112469, 0.000100000, 0.000071768, 0.000100000, 0.000055137, 0.000100000, 0.000071768, 0.000223607, 0.000118909],
[0.000100000, 0.000070520, 0.000300000, 0.000171333, 0.000400000, 0.000098837, 0.000200000, 0.000097742, 0.000400000, 0.000098837, 0.000316228, 0.000185278],
        ]),  
    },
}

eta_SF = {
    "Values": {
        "Eta_1": np.array([
[1.006230000, 1.005490000, 1.001100000, 0.939300000, 1.016200000, 0.990300000, 1.004300000, 0.995800000, 0.659340659, 1.193277311, 0.964689462, 1.261589664],
[0.997280000, 0.998310000, 1.001300000, 0.939700000, 1.015600000, 0.990800000, 1.003500000, 0.995700000, 0.653579677, 1.190265487, 1.013325103, 1.457371703],
[0.998530000, 0.997568927, 0.999630000, 0.937811735, 1.012930000, 0.992160833, 1.003990000, 0.997322595, 0.665775401, 1.188525527, 1.018125310, 1.567151500],
[0.998290000, 0.997983149, 1.002250000, 0.945077429, 1.010900000, 0.994108608, 0.997430000, 0.995549154, 0.592307692, 1.220186522, 0.992116912, 1.775943936],
[0.998570000, 0.997942594, 0.999480000, 0.940810818, 1.009370000, 0.994742883, 0.996510000, 0.995096826, 0.589285714, 1.216676235, 1.035929909, 1.949653558],
        ]),  
    },       
    "Errors": {
        "Eta_1": np.array([                    
[0.003800000, 0.003620000, 0.020000000, 0.020000000, 0.020000000, 0.020000000, 0.020000000, 0.020000000, 0.526506461, 0.654156038, 0.168636780, 0.184808342],
[0.001340000, 0.001700000, 0.010020000, 0.010000000, 0.010000000, 0.010000000, 0.010000000, 0.010000000, 0.275898544, 0.343934550, 0.130753923, 0.149391812],
[0.000660000, 0.000418633, 0.001650000, 0.001129181, 0.001110000, 0.000661955, 0.001060000, 0.000602794, 0.022672757, 0.017745718, 0.017686914, 0.014881839],
[0.000100000, 0.000054355, 0.000260000, 0.000171873, 0.000160000, 0.000098912, 0.000130000, 0.000076795, 0.004470198, 0.004178877, 0.005332313, 0.003613356],
[0.000180000, 0.000090745, 0.000400000, 0.000233545, 0.000400000, 0.000135574, 0.000170000, 0.000125018, 0.011430771, 0.006306152, 0.008917963, 0.006543006],
        ]),  
    },
}

# 1. Plotting Function
def plot_heatmap(values, errors, pt_labels, eta_labels, plot_type="EFF", 
                 title="Heatmap", output_file="heatmap.pdf"):
    
    # Setup Colormap
    if plot_type == "EFF":
        colors_list = [(0, "red"), (1, "yellow")]
        cmap = LinearSegmentedColormap.from_list("spike_at_one", colors_list)
        norm = colors.Normalize(vmin=np.nanmin(values), vmax=np.nanmax(values))
        cbar_label = "EFF"
    else: # SF
        deviation = max(abs(np.nanmin(values) - 1), abs(np.nanmax(values) - 1))
        mean = np.nanmean(values)
        vmin = 1 - deviation
        vmax = 1 + deviation
        epsilon = 1e-8
        mean = np.clip(mean, vmin + epsilon, vmax - epsilon)
        cmap = LinearSegmentedColormap.from_list("yellow_centered", ["red", "white", "blue"])
        norm = colors.TwoSlopeNorm(vcenter=mean, vmin=vmin, vmax=vmax)
        cbar_label = "SF"

    # CRITICAL: Always set bad pixels to white
    cmap.set_bad(color='white')

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(values, cmap=cmap, norm=norm, aspect='auto')

    # Axis Setup - Horizontal labels
    ax.set_xticks(np.arange(len(eta_labels)))
    ax.set_yticks(np.arange(len(pt_labels)))
    ax.set_xticklabels(eta_labels, fontsize=9, rotation=0, ha='center') 
    ax.set_yticklabels(pt_labels, fontsize=12)
    ax.set_xlabel('Eta Region', fontsize=14)
    ax.set_ylabel('$p_T$ [GeV]', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.invert_yaxis()

    # Cell values
    rows, cols = values.shape
    for i in range(rows):
        for j in range(cols):
            val = values[i, j]
            err = errors[i, j]
            # Only print if valid number
            if not np.isnan(val):
                text = f"{val:.3f} ± {err:.3f}"
                ax.text(j, i, text, ha="center", va="center", color="black", fontsize=8)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, fontsize=12)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file)
    print(f"Generated plot: {output_file}")
    plt.close()
    #plt.show()

# 2. Execution Loop
# (Keeping your eta_data, etc., defined as before)

eta_keys = ["barrel_1", "barrel_2", "Endcap"]
eta_labels_map = {
    "Eta_1": "|η| ≤ 0.8",
    "Eta_2": "0.8 < |η| ≤ 1.4442",
    "Eta_3": "1.4442 < |η| ≤ 2.8"
}
pretty_eta_labels = [eta_labels_map[k] for k in eta_keys]

selections = ['GoldID_2022_preEE', 'GoldID_2022_postEE',
              'ISO_2022_preEE'   , 'ISO_2022_postEE',
              'PROMPT_2022_preEE', 'PROMPT_2022_postEE',
              'blp_2022_preEE'   , 'blp_2022_postEE',
              'Not_Prompt_2022_preEE', 'Not_Prompt_2022_postEE',
              'ID_nor_ISO_2022_preEE', 'ID_nor_ISO_2022_postEE',
              ]

pt_labels = ['3-10', '10-20', '20-45', '45-75', '75-500']
eta_pretty_labels = ['|η| ≤ 0.8', '0.8 < |η| ≤ 1.442', '1.442 < |η| ≤ 2.8']

pt_labels = ['3-6', '6-10', '10-20', '20-45', '45-500']

all_datasets = {"Data": eta_data, "MC": eta_MC, "SF": eta_SF}

for set_name, dataset in all_datasets.items():
    plot_type = "SF" if set_name == "SF" else "EFF"
    
    for sel_idx, sel_name in enumerate(selections):
        pivoted_vals = np.zeros((5, 4))
        pivoted_errs = np.zeros((5, 4))
        
        for eta_idx, eta_key in enumerate(eta_keys):
            pivoted_vals[:, eta_idx] = dataset["Values"][eta_key][:, sel_idx]
            pivoted_errs[:, eta_idx] = dataset["Errors"][eta_key][:, sel_idx]
        
        # MASKING LOGIC: Find indices where Value is 1.0 and Error is 0.0
        # np.isclose is safer than == for float comparisons
        mask = np.isclose(pivoted_vals, 1.0) & np.isclose(pivoted_errs, 0.0)
        pivoted_vals[mask] = np.nan
        
        plot_heatmap(
            values=pivoted_vals,
            errors=pivoted_errs,
            pt_labels=pt_labels,
            eta_labels=pretty_eta_labels,
            plot_type=plot_type,
            title=f"2022_{set_name}_{sel_name}",
            output_file=f"Fitter/AN_HEATMAPS_ELE_1/2022/{set_name}/{set_name}_{sel_name}_HEATMAP.pdf"
        )