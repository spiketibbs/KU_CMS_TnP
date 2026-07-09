import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from test_int import (
    double_crystal_ball, phase_space, linear, exponential, chebyshev_background,
    create_combined_model, FIT_CONFIGS, SIGNAL_MODELS, BACKGROUND_MODELS
)

# --- CONFIGURATION ---
fit_type = "dcb_lin"  # You can change to any other valid type from FIT_CONFIGS
x_min, x_max = 70, 110
x = np.linspace(x_min, x_max, 500)
n_bins_pass = 30  # Just a placeholder, adjust as needed

# Dummy histogram centers and values (replace with real data)
centers_pass = np.linspace(75, 105, n_bins_pass)
values_pass = np.random.poisson(100, size=n_bins_pass)
centers_fail = np.linspace(75, 105, n_bins_pass)
values_fail = np.random.poisson(50, size=n_bins_pass)

# Retrieve model configuration
signal_func = SIGNAL_MODELS[fit_type.split('_')[0]]["func"]
signal_params_names = SIGNAL_MODELS[fit_type.split('_')[0]]["params"]
bg_func = BACKGROUND_MODELS[fit_type.split('_')[1]]["func"]
bg_param_names = BACKGROUND_MODELS[fit_type.split('_')[1]]["params"]

# Background parameters (pass and fail independently)
bg_pass_params = {p: 1.0 for p in bg_param_names}
bg_fail_params = {p: 1.5 for p in bg_param_names}

# --- Interactive Setup ---
def build_signal(params):
    return signal_func(x, *[params[name] for name in signal_params_names])

def build_bg(params_dict):
    return bg_func(x, *[params_dict[name] for name in bg_param_names])

# Initial shared parameters
shared_params = {
    "N": 1000,
    "epsilon": 0.95,
    **{p: 90 if 'mu' in p else 2 for p in signal_params_names},
}

fig, (ax_pass, ax_fail) = plt.subplots(2, 1, figsize=(12, 8))
plt.subplots_adjust(bottom=0.35)

line_pass, = ax_pass.plot([], [], 'k-', label="Total Fit")
line_fail, = ax_fail.plot([], [], 'k-', label="Total Fit")

ax_pass.errorbar(centers_pass, values_pass, fmt='o', color='blue', label="Pass Data")
ax_fail.errorbar(centers_fail, values_fail, fmt='o', color='red', label="Fail Data")

ax_pass.set_title("Pass Fit")
ax_fail.set_title("Fail Fit")

ax_pass.legend()
ax_fail.legend()

# Create sliders
slider_axes = {}
sliders = {}
y_base = 0.25

for i, param in enumerate(["N", "epsilon"] + signal_params_names):
    ax = plt.axes([0.1, y_base - i*0.04, 0.65, 0.03])
    slider_axes[param] = ax
    valinit = shared_params[param]
    valmin = 0.7 if param == "epsilon" else valinit * 0.5
    valmax = 1.0 if param == "epsilon" else valinit * 2
    sliders[param] = Slider(ax, param, valmin, valmax, valinit=valinit)

# Update function
def update(val):
    for key in shared_params:
        shared_params[key] = sliders[key].val

    signal = build_signal(shared_params)
    bg_pass = build_bg(bg_pass_params)
    bg_fail = build_bg(bg_fail_params)

    y_pass = shared_params["N"] * shared_params["epsilon"] * signal + 100 * bg_pass
    y_fail = shared_params["N"] * (1 - shared_params["epsilon"]) * signal + 100 * bg_fail

    line_pass.set_data(x, y_pass)
    line_fail.set_data(x, y_fail)

    ax_pass.relim()
    ax_pass.autoscale_view()
    ax_fail.relim()
    ax_fail.autoscale_view()

    fig.canvas.draw_idle()

for s in sliders.values():
    s.on_changed(update)

update(None)
plt.show()