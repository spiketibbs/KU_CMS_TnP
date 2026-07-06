# Tag and Probe Analysis

Determining Scale Factors for electrons and muons in Eras 2 and 3 in the LHC

### Prerequisites

* Source Code Editor / IDE (i.e. VSCode)
* Python 3.10+
* Git

### Installation

Step-by-step commands to set up the environment:
\`\`\`bash

# Clone the repository

```bash
$ git clone https://[github.com/spiketibbs/KU_CMS_TnP](https://github.com/spiketibbs/KU_CMS_TnP/)
```

Navigate into the project directory

``` bash
$ cd project-name
```

# Install dependencies

``` bash
pip install numpy scipy pandas matplotlib mplhep uproot iminuit numba-stats rich openpyxl
```

## 🛠️ Usage & Examples

- [Fitter](#fitter)
- [Extrapolator](#extrapolator)
- [Heatmap Generator](#heatmap-generator)

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

# Heatmap Generator 
## 📁 Repository Structure

A quick map of where everything lives so users don't get lost:

├── src/                # Source code
├── samples/            # Sample data and configuration templates
├── docs/               # Deep-dive documentation and presentation slides
├── README.md           # This file
└── package.json        # Project dependencies
