import uproot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re


# Load the ROOT file
file1 = uproot.open("/uscms/home/hortua/nobackup/egamma-tnp/examples/nanoaod_filters_custom/blp_3gt/DATA_NEW_2023D/get_1d_pt_eta_phi_tnp_histograms_1/DATA_NEW_23D_histos_pt_barrel_1.root")

for key in file1.keys():
    obj = file1[key]
    if isinstance(obj, uproot.behaviors.TTree.TTree):
        print(f"{key}: {obj.num_entries} entries")

#=== Loop over all "_Pass;1" histograms and compare ===
diff_list = []
pass_pattern = re.compile(r"bin\d+_.*?_Pass;1")
for key in file1.keys():
    if pass_pattern.match(key):
        h1 = file1[key]

        val1 = h1.values()

        total_diff = np.sum(val1)

        diff_list.append({
            "Histogram": key,
            "File1 Total": np.sum(val1),
            "Total Difference": total_diff
        })
# === Print all differences ===
diff_df = pd.DataFrame(diff_list)
print(diff_df.sort_values("Histogram"))