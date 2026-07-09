# Tag and Probe Analysis

Determining Scale Factors for electrons and muons in Eras 2 and 3 in the LHC.  
For a more detailed explanation or background on theory and the making of the fitter, please refer to:  
  Teams - lowptelec team - Presentations folder

### Prerequisites

* Source Code Editor / IDE (i.e. VSCode)
* Python 3.10+ (Local Computer)
* Download GitHub Command Line Interface
   * Windows: [Git for Windows](https://gitforwindows.org/)
   * Mac (Follow the displayed directions after running the command below to install Xcode):
  ```bash
  git --version
  ```

### Installation

Step-by-step commands to set up the environment:

1. Open Terminal

2. Make and enter the folder that you want the Git repository to go in
```bash
mkdir Physics
cd Physics
```

3. Clone the repository

```bash
git clone https://github.com/spiketibbs/KU_CMS_TnP
```

4. Navigate into the project directory

``` bash
cd KU_CMS_TnP
```

5. Install dependencies

``` bash
pip install numpy scipy pandas matplotlib mplhep uproot iminuit numba-stats rich openpyxl
```

## 🛠️ Usage & Examples

- [Fitter](#fitter)
- [Extrapolator](#extrapolator)
- [Heatmap Generators](#heatmap-generators)
   - [Efficiency Heatmap](#efficiency-heatmap)
      - [Electron ISO Efficiency Heatmap](#electron-iso-efficiency-heatmap)
   - [Scale Factor Heatmap](#scale-factor-heatmap)

# Fitter

To run the main application:
1. Retrieve all DATA and MC histograms from Teams - lowptelec team - EGamma Tag and Probe folder - scalefac.xlsx
2. Open config_MUONS.json in a text editor
3. Edit accordingly. Below is an explanation of each parameter in the config file. Note: The comments are for explanation only, as a JSON file cannot handle Python-style comments (#) and explanatory arrows (< -----). The actual config_MUONS.json file is clean and ready to use.
  ``` bash
#   {
#     "info_level": "INFO_2"       < ----- INFO, DEBUG, or INFO_2 (more verbose) (leave blank for no output)
#     "mass": "Z_muon",                 < ----- Determines what mass you are fitting (Z, Z_muon, JPsi, JPsi_muon)
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
#       "bin_ranges": [[3,6],[6,10],[10,20],[20,45],[45,500]],    < ----- Specify which pT range(s) you are fitting (in example, bin0 (3-6), bin1 (6-10), bin2 (10-20), bin3 (20-45), bin4 (45-500))
#       "bin": ["bin0", "bin1, etc"],    < ----- Specify which pT range(s) you are fitting (in example, bin0 (3-6), bin1 (6-10), bin2 (10-20), bin3 (20-45), bin4 (45-500))
#       "fit_type": "dcb_cms"    < ----- Format is: (signal shape)_(background shape). Signal shapes: (dcb, g, dv, cbg), Background shapes: (lin, exp, cms, bpoly, cheb, ps)
#       "use_cdf": false,        < ----- If a shape does not have a cdf version, defaults back to pdf
#       "sigmoid_eff": false,    < ----- Switches to an unbounded efficiency that is transformed back between 0 and 1
#       "interactive": true,     < ----- Turns on interactive window for fitting (very useful for difficult fits)
#       "x_min": 60,             < ----- x range minimum for plotting
#       "x_max": 140,            < ----- x range maximum for plotting
#       "abseta": 1,             < ----- ***Only impacts muon .root files. Defines absolute eta ranges
#       "numerator": "GoldID",     < ----- ***Only impacts muon .root files. Defines numerator for efficiencies
#       "denominator": "baselineplus"     < ----- ***Only impacts muon .root files. Defines denominator for efficiencies
#     },
#     "output": {
#       "plot_dir": "Fitter/Muons/2018/GoldID/",         < ----- Sets location to save plots to (if left blank, it won't save). Recommended: mass/era/numerator
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
```

4. Run from the repository root directory and run the fitter "run_fitter_p_bar.py" with the parameter "--config config_MUONS.json"
``` bash
cd ~/KU_CMS_TnP/
python3 Fitter/run_fitter_p_bar.py --config Fitter/config_MUONS.json
```
5. Interactive Fit:
   1. The first interactive plot canvas that pops up will be for DATA.
   2. You can adjust the type of minimizer to use. Migrad() is on by default.
   3. Click Fit while repeatedly varying the bar parameters until the top left box in the 3X3 table is green and says "Valid Minimum". This means that the fit is successful. Ensure that the Chi Squared value is reasonable as well (x<sup>2</sup>/ndof)
   
   <img width="385" height="107" alt="Screenshot 2026-07-07 at 6 47 22 PM" src="https://github.com/user-attachments/assets/d57456ae-335e-4c8d-a653-3bddca9e4715" />

   5. Click X on the top left and repeat for MC.
   6. If you listed multiple bins in the config files, more plot canvases will pop up, one DATA and one MC, for each bin with just one run.
6. Open file explorer and open the specified plot_dir from the config file to view 4plots and the Excel file with saved efficiency and scale factor values, along with error bars.


### Working with Samples

We have provided sample files in the `/Fitter` directory to help you get started:

* **`Fitter/Run2018_UL/Nominal/NUM_GoldID_DEN_baselineplus_abseta_pt.root`**: Use this file, as well as the other three identification criteria DATA files, to test the fitter, viewing outputted 4plots and .xlsx files.
* **`Fitter/DY_madgraph/Nominal/NUM_GoldID_DEN_baselineplus_abseta_pt`**: Use this file, as well as the other three identification criteria MC files, to test the fitter, viewing outputted 4plots and .xlsx files.
* Note that there are already example fitted files in the Fitter/Muons directory. Running the config file with the example parameters will replace all values in the example results. For a clean slate, you can also delete the Muons directory before running.


# Extrapolator
Linearly extrapolates Efficiencies, Scale Factors, and Errors for lower pT ranges for electrons (Extrapolation_ele.py) or Muons (Extrapolation_muon.py) based on fitted values obtained from higher pT ranges.

1. Fit values that you will use as baseline values for the extrapolation.
2. Ensure you are in the main repository
```bash
cd ~/KU_CMS_TnP/
```
3. Open either Extrapolation_ele.py (electron extrapolation) or Extrapolation_muon.py (Muon extrapolation) in a text editor
4. Modify 'excel_path' to point to the right spreadsheet, beginning on line 16.
5. Modify the image title and output file directory in lines 353 and 358 for electron, or 355 and 360 for muon.
```bash
plt.suptitle(f"BLP Data/MC Efficiency Scale Factor Fits (${eta_label}$)", fontsize=16, fontweight="bold") # line 353 (Electron) or 355 (Muon)
# ...
output_file = f"Extraps/EXTRAP_PLOTS_ELE/24/2024_blp_extrap_{dataset_key}.pdf" # line 358 (Electron) or 360 (Muon)
```

6. Enter the pT bin ranges you had put fit values in, in line 369 for electron and 371 for muon, and enter what bin ranges to extrapolate for in line 370 for electron and 372 for muon.
```bash
my_fit_bins = [(10, 15), (15, 20), (20, 25), (25, 30), (30, 35), (35, 40), (40, 45)] # line 369 (Electron) or 370 (Muon)
# ...
my_extrap_bins = [(2, 4), (4, 7)] # line 371 (Electron) or 372 (Muon)
```
   * If modifying the extrapolated bin ranges, you must also modify the error bar calculations in lines 214, 219, 271, and 275 for Electron, and 216, 221, 273, and 277 for Muon.
   ```bash
   # extrapolated to a<pT<b and b<pT<c (replace a, b, and c with preferred bin ranges)
   mask_3_6 = (xx_band >= a) & (xx_band < b) # line 214 (Electron) or 216 (Muon)
   mask_6_10 = (xx_band >= b) & (xx_band <= c) # line 219 (Electron) or 221 (Muon)
   # ...
   # extrapolated to a<pT<b and b<pT<c (replace a, b, and c with preferred bin ranges)
   if x1 == a and x2 == b: # line 271 (Electron) or 273 (Muon)
   elif x1 == b and x2 == c: # line 275 (Electron) or 277 (Muon)
```
7. Run the code
```bash
python3 Extraps/Extrapolation_ele.py
```
or
```bash
python3 Extraps/Extrapolation_muon.py
```
8. Add values from extrapolation files to the results spreadsheet (sfs_ecol.xlsx or sfs_mcol.xlsx) to prepare for generating heatmaps. Since extrapolated values are in the lowest bins, you might need to rename all the bins in the Excel sheet by adding 2 to each name(bin0 now becomes bin2, etc.) to account for a new bin0 and bin1 for the two extrapolated pT ranges. View the currently present Fitter/Muons/2018/GoldID/sfs_mcol.xlsx for an example.
# Heatmap Generators
1. Ensure you are in the main repository
```bash
cd ~/KU_CMS_TnP/
```
## Efficiency Heatmap
1. Open 2D_heatmap_eff.py in a text editor
2. Modify configurations LEPTON_TYPE, HEATMAP_TYPE, NUM, and ERA.
```bash
LEPTON_TYPE = 'Muon'  # Options: 'Electron' or 'Muon'
HEATMAP_TYPE = "DATA"  # Options: 'DATA' or 'MC' (affects which rows are read from the Excel file)
NUM = 'GoldID'  # Options: 'GoldID', 'ISO', 'Prompt', 'BLP', 'Not Prompt', or 'Not ID nor ISO'
ERA = "2018" # Electron Options: '2016', '2017', '2018', '2022 PreEE', '2022 PostEE', '2023 PreBPix', '2023 PostBPix'"
# Muon Options: '2016', '2017', '2018', '2022 PreEE', '2022 PostEE', '2023 PreBPix', '2023 PostBPix'"
```
3. Modify 'excel_path' under LEPTON_CONFIGS to point to the right spreadsheet, under LEPTON_CONFIGS in either Electron or Muon.
4. Run the code for both HEATMAP_TYPE = "DATA" and HEATMAP_TYPE = "MC" (adjust on line 16)
```bash
python3 Heatmaps/2D_heatmap_eff.py
```
## Electron ISO Efficiency Heatmap
1. Open heatmap_eff_ele_ISO.py in a text editor
2. Define the eras in current order, as well as eta labels. Manually enter all efficiencies and error values under data_dict in the format provided.
```bash
# ---------------------------------------------------------
# MANUAL DATA ENTRY
# ---------------------------------------------------------
# Define the order of the eras as they should appear from top to bottom
eras_ordered = [
    '2024',
    '2023_postBPix',
    '2023_preBPix',
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
# Example of manual entry layout:
data_dict = {
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
```
3. Run the code for both HEATMAP_TYPE = "DATA" and HEATMAP_TYPE = "MC" (adjust on line 25)
```bash
python3 Heatmaps/heatmap_eff_ele_ISO.py
```
## Scale Factor Heatmap
1. Open heatmap_sf.py in a text editor
2. Modify configurations LEPTON_TYPE, HEATMAP_TYPE, NUM, and ERA.
```bash
LEPTON_TYPE = 'Muon'  # Options: 'Electron' or 'Muon'
NUM = 'GoldID'  # Options: 'GoldID', 'ISO', 'Prompt', 'BLP', 'Not Prompt', or 'Not ID nor ISO'
ERA = "2018" # Electron Options: '2016', '2017', '2018', '2022 PreEE', '2022 PostEE', '2023 PreBPix', '2023 PostBPix'"
# Muon Options: '2016', '2017', '2018', '2022 PreEE', '2022 PostEE', '2023 PreBPix', '2023 PostBPix'"
```
3. Modify 'excel_path' under LEPTON_CONFIGS to point to the right spreadsheet, under LEPTON_CONFIGS in either Electron or Muon.

5. Run the code
``` bash
python3 Heatmaps/heatmap_sf.py
```

## Repository Structure
```text
├── Extraps/            # Extrapolation code and samples
├── Fitter/             # Fitter code, sample data and configuration templates
├── Heatmaps/           # Heatmap code and samples
├── README.md           # This file
