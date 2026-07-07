# Tag and Probe Analysis

Determining Scale Factors for electrons and muons in Eras 2 and 3 in the LHC
For a more detailed explanation or background on theory and the making of the fitter, please refer to Teams - lowptelec team - Presentations folder

### Prerequisites

* Source Code Editor / IDE (i.e. VSCode)
* Python 3.10+ (Local Computer)
* Download GitHub Command Line Interface
   * Windows: [Git for Windows](https://gitforwindows.org/)
   * Mac (Follow the displayed directions after running below command to install Xcode):
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
- [Heatmap Generator](#heatmap-generator)
   - [Efficiency Heatmap](#efficiency-heatmap)
      - [Electron ISO Efficiency Heatmap](#electron-iso-efficiency-heatmap)
   - [Scale Factor Heatmap](#scale-factor-heatmap)

# Fitter

To run the main application:
Open config_MUONS.json in a text editor
1. Edit accordingly. Below is an explanation of each parameter in the config file.
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
#       "plot_dir": "Muons/2018/GoldID/",         < ----- Sets location to save plots to (if left blank, it won't save). Recommended: mass/era/numerator
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

4. Enter the Fitter folder and run the fitter "run_fitter_p_bar.py" with the parameter "--config config_MUONS.json"
   ``` bash
   cd Fitter/
   python3 run_fitter_p_bar.py --config config_MUONS.json
   ```
6. Interactive Fit:
   1. The first interactive plot canvas that pops up will be for DATA.
   2. You can adjust the type of minimizer to use. Migrad() is on by default.
   3. Click Fit while repeatedly varying the bar parameters until the top left box in the 3X3 table is green and says "Valid Minimum". This means that the fit is successful. Ensure that the Chi Squared value is reasonable as well.
   4. Click X on the top left and repeat for MC.
   5. If you listed multiple bins in the config files, more plot canvases will pop up, one DATA and one MC, for each bin with just one run.
7. Open file explorer and open the specified plot_dir from the config file to view 4plots and the Excel file with saved efficiency and scale factor values, along with error bars.


### Working with Samples

We have provided sample files in the `/Fitter` directory to help you get started:

* **`Fitter/Run2018_UL/Nominal/NUM_GoldID_DEN_baselineplus_abseta_pt.root`**: Use this file, as well as the other three identification criteria DATA files, to test the fitter, viewing outputted 4plots and .xlsx files.
* **`Fitter/DY_madgraph/Nominal/NUM_GoldID_DEN_baselineplus_abseta_pt`**: Use this file, as well as the other three identification criteria MC files, to test the fitter, viewing outputted 4plots and .xlsx files.


# Extrapolator
Linearly extrapolates Efficiencies, Scale Factors, and Errors for lower pT ranges for electrons (Extrapolation_ele.py) or Muons (Extrapolation_muon.py) based on fitted values obtained from higher pT ranges.

1. Ensure you are in the main repository
```bash
cd .. #leaves Fitter folder
```
3. Open either Extrapolation_ele.py (electron extrapolation) or Extrapolation_muon.py (Muon extrapolation) in a text editor
4. Open the Excel file generated from fitting and enter all values obtained from fitting.
5. Modify the image title and output file directory in lines 334 and 339 for electron, or 350 and 355 for muon.
6. Enter the pT bin ranges you had put fit values in, in line 350/366, and enter what bin ranges to extrapolate for in line 351/367.
7. Run the code
```bash
python3 Extraps/Extrapolation_ele.py
```
or
```bash
python3 Extraps/Extrapolation_muon.py
```
# Heatmap Generator
1. Ensure you are in the main repository
```bash
cd .. #leaves Extraps folder
```
## Efficiency Heatmap
1. Open 2D_heatmap_eff.py in a text editor
2. Modify configurations LEPTON_TYPE, HEATMAP_TYPE, NUM, and ERA.
   ```bash
   LEPTON_TYPE = 'Electron'  # Options: 'Electron' or 'Muon'
   HEATMAP_TYPE = "DATA"  # Options: 'DATA' or 'MC' (affects which rows are read from the Excel file)
   NUM = 'Not Prompt'  # Options: 'GoldID', 'ISO', 'Prompt', 'BLP', 'Not Prompt', or 'Not ID nor ISO'
   ERA = "2023 PreBPix" # Electron Options: '2016', '2017', '2018', '2022 PreEE', '2022 PostEE', '2023 PreBPix', '2023 PostBPix'"
       # Muon Options: '2016', '2017', '2018', '2022 PreEE', '2022 PostEE', '2023 PreBPix', '2023 PostBPix'"
   ```
4. Modify 'excel_path' under LEPTON_CONFIGS to point to the right spreadsheet, in either Electron
5. Run the code for both HEATMAP_TYPE = "DATA" and HEATMAP_TYPE = "MC"
```bash
python3 Heatmaps/2D_heatmap_eff.py
```
## Electron ISO Efficiency Heatmap
1. Open heatmap_eff_ele_ISO.py in a text editor
2. Define the eras in current order, as well as eta labels. Manually enter all efficiencies and error values under data_dict in the format provided.
3. Run the code
```bash
python3 Heatmaps/heatmap_eff_ele_ISO.py
```
## Scale Factor Heatmap
1. Open heatmap_sf.py in a text editor
2. Fill in configurations, with options provided in comments. Modify the Excel path under LEPTON_CONFIGS to point to the right spreadsheet.
3. Run the code
``` bash
python3 Heatmaps/heatmap_sf.py
```

## Repository Structure

A quick map of where everything lives so users don't get lost:
```text
├── Extraps/            # Extrapolation code and samples
├── Fitter/             # Fitter code, sample data and configuration templates
├── Heatmaps/           # Heatmap code and samples
├── README.md           # This file
