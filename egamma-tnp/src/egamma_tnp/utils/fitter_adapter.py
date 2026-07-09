from iminuit.interactive import make_widget

def build_interactive_widget_for_file(
    hist_pass, hist_fail, *, fit_type, use_cdf, args_bin, args_data, args_mass,
    sigmoid_eff, x_min, x_max, raise_on_exception=False
):
    """
    Return (widget, result_getter). The widget is *not* executing the event loop.
    result_getter is a function we can call after the window closes to retrieve
    whatever results we want to save.
    """
    # 1) Build your model/minuit exactly like fit_function does, but stop before
    #    make_widget(app.exec()) blocks. You may need to carve the code out of
    #    fit_function into a helper.
    minuit, plot_callable, kwargs = _prepare_minuit_and_plot(   # <-- you implement
        hist_pass, hist_fail, fit_type, use_cdf, args_bin, args_data,
        args_mass, sigmoid_eff, x_min, x_max
    )

    # 2) make_widget WITHOUT starting the Qt loop
    widget = make_widget(
        minuit=minuit,
        plot=plot_callable,
        kwargs=kwargs,
        raise_on_exception=raise_on_exception,
        run_event_loop=False,         # <--- critical
    )

    # 3) Return a result getter to serialize/save what you need after closing
    def result_getter():
        # Whatever you want to extract (e.g. minuit.values, fmin, cov, etc.)
        return {
            "data_type": args_data,
            "bin": args_bin,
            "summary": getattr(minuit.fmin, "_repr_html_", lambda: str(minuit.fmin))(),
            "values": dict(minuit.values),
            "errors": dict(minuit.errors),
            "fval": minuit.fmin.fval,
        }

    return widget, result_getter