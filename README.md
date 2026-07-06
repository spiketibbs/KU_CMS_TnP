# Tag and Probe Analysis

Determining Scale Factors for electrons and muons in Eras 2 and 3 in the LHC

### Prerequisites

* Source Code Editor / IDE (i.e. VSCode)
* Python 3.10+
* Download GitHub Command Line Interface
   * Windows: [Git for Windows](https://gitforwindows.org/)
   * Mac (Follow the displayed directions after running below command to install Xcode):
  ```bash
  git --version
  ```

### Installation

Step-by-step commands to set up the environment:

# Clone the repository

```bash
git clone https://github.com/spiketibbs/KU_CMS_TnP
```

Navigate into the project directory

``` bash
cd project-name
```

# Install dependencies

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
Explain how to actually use the project. Refer directly to your sample files here.

To run the main application:

1. Edit "mass" for what mass you are measuring (Z, Z_muon)
2. Edit the config Muon with the two input root files you are using, one for data and one for MC
3. Under fit
   1. Enter bin ranges present in the root files under "bin ranges" In the format [[x_1, x_2], [x_2, x_3], ...]
   2. Under bin, enter the bins you want to fit, starting at bin 0. i.e. [bin0, bin2, bin3]
   3. Enter fit type for signal and background signal shapes.
      1. Signals available: double crystal ball (dcb), gaussian (g), crystal ball x gaussian (cbg), double voigtian (dv).
      2. Backgrounds available: linear (lin), exponential (exp), phase space (ps), chebyshev polynomial (cheb), Bernstein polynomial (bpoly), CMS shape (cms).
   4. use_cdf: true or false. -- Use cdf for signal and background shapes, will fall back to pdf if either function doesn’t have a cdf available
   5. sigmoid_eff: true or false
   6. interactive: true or false. If used, adds a visual fitter that is interactive. (must have PySide6 installed)
   7. x_min and x_max: the minimum and invariant mass values, in int form
   8. abseta: the abseta barrel to fit
   9. numerator:
   10. denominator:
   11. separate_signal_shape: true or false
   12. plot_dir: directory that the 4plots, pass and fail plots, and excel file will go in
   13. results_file:
4. run the run_fitter_p_bar.py with parameter --config config_MUONS.json
5. Interactive Fit:
   1. You can adjust the type of minimizer to use. Migrad() is on by default.

npm start
\`\`\`

### Working with Samples

We have provided sample files in the `/samples` directory to help you get started:

* **`samples/input_data.json`**: Use this file to test the data ingestion tool.
* **`samples/config.example.env`**: Duplicate this file, rename it to `.env`, and add your local API keys.


# Extrapolator
Linearly extrapolates Efficiencies, Scale Factors, and Errors for lower pT ranges for electrons (Extrapolation_ele.py) or Muons (Extrapolation_muon.py) based on fitted values obtained from higher pT ranges.

1. Open either Extrapolation_ele.py (electron extrapolation) or Extrapolation_muon.py (Muon extrapolation)
2. Enter all values obtained from fitting as applicable.
3. Modify the image title and output file directory in lines 334 and 339 for electron, or 350 and 355 for muon.
4. Enter the pT bin ranges you had put fit values in, in line 350/366, and enter what bin ranges to extrapolate for in line 351/367.
# Heatmap Generator
## Efficiency Heatmap
1. Open 2D_heatmap_eff.py
2. Fill in configurations, with options provided in comments. Modify the Excel path under LEPTON_CONFIGS to point to the right spreadsheet.
3. Run the code for both HEATMAP_TYPE = "DATA" and HEATMAP_TYPE = "MC"
## Electron ISO Efficiency Heatmap
1. Open heatmap_eff_ele_ISO.py
2. Define the eras in current order, as well as eta labels. Manually enter all efficiencies and error values under data_dict in the format provided.
3. Run the code
## Scale Factor Heatmap
1. Open heatmap_sf.py
2. Fill in configurations, with options provided in comments. Modify the Excel path under LEPTON_CONFIGS to point to the right spreadsheet.
3. Run the code
## 📁 Repository Structure

A quick map of where everything lives so users don't get lost:

├── src/                # Source code
├── samples/            # Sample data and configuration templates
├── docs/               # Deep-dive documentation and presentation slides
├── README.md           # This file
└── package.json        # Project dependencies
