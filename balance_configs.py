"""
Functions returning parameter dictionaries for the DSGC balance model.
"""


def sac_mode_config(
    ttx=False,
    leaky=False,
    high_kv=False,
    non_ds_ach=False,
    offset_ampa_ach=False,
    vc_mode=False,
    record_tree=True,
    poisson_rates=None,
    plexus=0,
    plexus_share=None,
):
    params = {
        # hoc settings
        "tstop": 350,
        "steps_per_ms": 10,
        "dt": 0.1,
        # synapse organization
        "term_syn_only": False,
        "first_order": 4,  # 2,
        # membrane properties
        "active_terms": False,
        "soma_Na": 0.2,
        "soma_K": 0.07,
        "soma_gleak_hh": 0.0001667,
        "soma_gleak_pas": 0.0001667,
        "prime_Na": 0.07,  # 0.045,  # 0.03 # 0.1,  # 0.011,
        "prime_K": 0.07,  # 0.035,  # 0.03,
        "prime_gleak_hh": 0.0001667,
        "prime_gleak_pas": 0.0001667,
        "dend_Na": 0.013,  # 0.011,
        "dend_K": 0.035,  # 0.03,  # 0.035,  # 0.03,
        "dend_gleak_hh": 0.0001667,
        "dend_gleak_pas": 0.0001667,
        # synapse variability
        "jitter": 0,
        "space_rho": 1,  # 0.9,
        "time_rho": 1,  # 0.9,
        "synprops": {
            "E": {
                "var": 15,  # 10,  # 12,  # 15,
                "delay": 5,  # 5,
                "null_prob": 0.0,
                "pref_prob": 1.0,  # 0.8,
                "weight": 0.0015,  # 0.0015,
                "tau1": 0.1,
                "tau2": 4,
            },
            "I": {
                "var": 15,  # 10,  # 12,  # 15,
                "delay": 0,
                "null_prob": 0.0,
                "pref_prob": 1.0,  # 0.8,
                "weight": 0.006,  # 0.006,
                "tau1": 0.5,
                "tau2": 16,
                "rev": -65,  # -60.,
            },
            "NMDA": {
                "var": 7,
                "delay": 0,
                "null_prob": 0.8,  # 0.8,  # 0.4,  # 0.8,
                "pref_prob": 0.8,  # 0.8,  # 0.4,  # 0.8,
                "weight": 0.002,  # 0.002,  # 0.0045,
                "tau1": 2,
                "tau2": 7,
            },
            "AMPA": {
                "var": 7,
                "delay": 0,
                "null_prob": 0.0,
                "pref_prob": 0.0,
                "weight": 0.0015,
                "tau1": 0.1,
                "tau2": 4,
            },
            "PLEX": {
                "tau1": 0.1,  # excitatory conductance rise tau [ms]
                "tau2": 4,  # excitatory conductance decay tau [ms]
                "rev": 0,  # excitatory reversal potential [mV]
                "weight": 0.00025,  # weight of exc NetCons [uS] .00023
                "prob": 0.5,  # probability of release
                "null_prob": 0.5,  # probability of release
                "pref_prob": 0.5,  # probability of release
                "delay": 0,  # [ms] mean temporal offset
                "var": 10,  # [ms] temporal variance of onset
            },
        },
        # quanta
        "max_quanta": 3,  # 3,
        "quanta_Pr_decay": 0.9,
        "quanta_inter": 5,  # 5
        "quanta_inter_var": 3,  # 3
        # stimulus
        "light_bar": {"speed": 1.0, "x_motion": True, "x_start": -60, "y_start": -70},
        # SAC Network
        "sac_mode": True,
        "sac_rho": 1.0,
        "sac_angle_rho_mode": True,
        "sac_uniform_dist": {0: True, 1: False},
        "sac_shared_var": 60,  # 45,
        "sac_theta_vars": {"E": 60, "I": 60},
        "sac_gaba_coverage": 0.5,
        "sac_offset": 30,
        "sac_theta_mode": "PN",
        "record_tree": record_tree,
        "n_plexus_ach": 0,
    }

    # PN theta mode, uniform inner + outer
    if 0:
        params["sac_uniform_dist"][1] = True
        params["sac_shared_var"] = 60
        params["sac_theta_vars"] = {"E": 90, "I": 90}
        params["sac_angle_rho_mode"] = True

    # Uniform theta mode, uniform outer (no inner)
    if 1:
        params["sac_uniform_dist"][1] = True
        params["sac_theta_mode"] = "uniform"
        params["sac_theta_vars"] = {"E": 60, "I": 60}
        params["sac_angle_rho_mode"] = True

    # testing out diameter scaling mode
    if 1:
        # params["diam_scaling_mode"] = "order"
        params["diam_scaling_mode"] = "cable"
        # params["diam_range"] = {"max": 2, "min": .4, "decay": .92, "scale": 1.5}
        params["diam_range"] = {"max": 1.5, "min": 0.4, "decay": 0.92, "scale": 1.0}
        # increase weight to counter incr diam
        # for s in params["synprops"].keys():
        #     params["synprops"][s]["weight"] *= 2
        # increase Nav to counter incr diam (want to lean on this more)
        # params["dend_Na"] = 0.015
        # params["dend_K"] = 0.035
        params["dend_Na"] = 0.05  # .04 # 0.03
        params["dend_K"] = 0.07

        # params["synprops"]["E"]["weight"] *= 1.
        # params["synprops"]["E"]["delay"] = 3
        # params["synprops"]["NMDA"]["weight"] *= 1.5

    # TTX
    if ttx:
        params["soma_Na"] = 0.0
        params["prime_Na"] = 0.0
        params["dend_Na"] = 0.0

    # MORE LEAK
    if leaky:
        params["soma_gleak_hh"] = 0.0001667 * 2
        params["soma_gleak_pas"] = 0.0001667 * 2
        params["prime_gleak_hh"] = 0.0001667 * 2
        params["prime_gleak_pas"] = 0.0001667 * 2
        params["dend_gleak_hh"] = 0.0001667 * 2
        params["dend_gleak_pas"] = 0.0001667 * 2

    # MORE Kv
    if high_kv:
        params["soma_K"] = 0.15
        params["prime_K"] = 0.15
        # params["dend_K"]  = 0.15
        # params["synprops"]["E"]["weight"]    *= 1.5
        # params["synprops"]["NMDA"]["weight"] *= 0
        # params["synprops"]["I"]["weight"]    *= 2.

    if vc_mode:
        params["vc_pas"] = True

    # non-ds ACH
    if non_ds_ach:
        params["synprops"]["E"]["null_prob"] = 0.5
        params["synprops"]["E"]["pref_prob"] = 0.5

    if offset_ampa_ach:
        params["synprops"]["E"]["null_prob"] = 0.0
        params["synprops"]["E"]["pref_prob"] = 0.0
        params["synprops"]["AMPA"]["delay"] = -30.0
        params["synprops"]["AMPA"]["pref_prob"] = 0.5
        params["synprops"]["AMPA"]["null_prob"] = 0.5

    if poisson_rates is not None:
        params["poisson_mode"] = True
        params["tstop"] = 550
        params["sac_rate"] = poisson_rates["sac"]
        params["glut_rate"] = poisson_rates["glut"]
        params["rate_dt"] = poisson_rates["dt"] * 1000.0
        for t in ["E", "I", "NMDA", "AMPA", "PLEX"]:
            params["synprops"][t]["tau1"] = 0.14
            params["synprops"][t]["tau2"] = 0.54

        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2278860/ (ach mini)
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6772523/ (gaba mini)
        params["synprops"]["I"]["tau2"] = 2.0
        params["synprops"]["E"]["tau2"] = 0.3
        params["synprops"]["NMDA"]["tau2"] = 1.0
        params["synprops"]["E"]["delay"] = 0.0

        base_w = 0.000313 * 3 * 1.5
        params["synprops"]["E"]["weight"] = base_w
        params["synprops"]["I"]["weight"] = base_w * 4
        params["synprops"]["NMDA"]["weight"] = base_w * 1.33
        params["synprops"]["PLEX"]["weight"] = base_w

    if plexus > 0:
        w = params["synprops"]["E"]["weight"]
        if plexus_share is not None:
            plex_w = w * plexus_share / plexus
            syn_w = w * (1.0 - plexus_share)
        else:
            plex_w = w / (plexus + 1)
            syn_w = plex_w

        params["n_plexus_ach"] = plexus
        params["synprops"]["PLEX"]["weight"] = plex_w
        params["synprops"]["E"]["weight"] = syn_w

    return params


def offset_mode_config():
    params = {
        # hoc settings
        "tstop": 750,
        "steps_per_ms": 10,
        "dt": 0.1,
        # synapse organization
        "first_order": 1,
        # membrane properties
        "soma_Na": 0.15,
        "soma_K": 0.07,
        "soma_gleak_hh": 0.0001667,
        "soma_gleak_pas": 0.0001667,
        "prime_Na": 0.015,
        "prime_K": 0.07,
        "prime_gleak_hh": 0.0001667,
        "prime_gleak_pas": 0.0001667,
        "dend_Na": 0.03,
        "dend_K": 0.035,
        "dend_gleak_hh": 0.0001667,
        "dend_gleak_pas": 0.0001667,
        # synapse variability
        "synprops": {
            "E": {
                "var": 15,
                "delay": 0,
                "weight": 0.0005,
                "tau1": 0.1,
                "tau2": 4,
                "null_offset": -50,
                "pref_offset": -50,
            },
            "I": {
                "var": 15,
                "delay": 0,
                "weight": 0.004,
                "tau1": 0.5,
                "tau2": 16,
                "null_offset": -55,
                "pref_offset": 0,
            },
            "AMPA": {
                "var": 7,
                "delay": 0,
                "weight": 0.0005,
                "tau1": 0.1,
                "tau2": 4,
                "null_offset": 0,
                "pref_offset": 0,
            },
            "NMDA": {"null_prob": 0, "pref_prob": 0},
        },
        # quanta
        "max_quanta": 15,
        "quanta_Pr_decay": 0.95,
        "quanta_inter": 5,
        "quanta_inter_var": 3,
        # stimulus
        "light_bar": {"speed": 0.5, "x_motion": True, "x_start": -60, "y_start": -70},
        # SAC Network
        "sac_mode": False,
    }

    # no offsets at all
    if 0:
        params["synprops"]["E"]["null_offset"] = 0
        params["synprops"]["E"]["pref_offset"] = 0
        params["synprops"]["I"]["null_offset"] = 0
        params["synprops"]["I"]["pref_offset"] = 0
    # symmetrical (non-ds) offsets
    if 0:
        params["synprops"]["E"]["null_offset"] = -50
        params["synprops"]["E"]["pref_offset"] = -50
        params["synprops"]["I"]["null_offset"] = -55
        params["synprops"]["I"]["pref_offset"] = -55

    # non-directional probability of inhibition
    if 0:
        params["synprops"]["I"]["pref_prob"] = params["synprops"]["I"]["null_prob"]

    return params


def ball_stick_config(
    ttx=False,
    leaky=False,
    high_kv=False,
    vc_mode=False,
    poisson_rates=None,
    record_tree=True,
):
    params = {
        # hoc settings
        "tstop": 350,
        "steps_per_ms": 10,
        "dt": 0.1,
        # membrane properties
        "soma_na": 0.2,
        "soma_k": 0.07,
        "soma_gleak_hh": 0.0001667,
        "soma_gleak_pas": 0.0001667,
        "initial_na": 0.07,
        "initial_k": 0.07,
        "initial_gleak_hh": 0.0001667,
        "initial_gleak_pas": 0.0001667,
        "dend_na": 0.013,
        "dend_k": 0.035,
        "dend_gleak_hh": 0.0001667,
        "dend_gleak_pas": 0.0001667,
        "term_na": 0.013,
        "term_k": 0.035,
        "term_gleak_hh": 0.0001667,
        "term_gleak_pas": 0.0001667,
        # synapse variability
        "space_rho": 0.9,  # 0.9,
        "time_rho": 0.9,  # 0.9,
        "synprops": {
            "E": {
                "var": 15,  # 10,  # 12,  # 15,
                "delay": 0,  # 5,
                "null_prob": 0.5,
                "pref_prob": 0.5,  # 0.8,
                "weight": 0.0015,  # 0.0015,
                "tau1": 0.1,
                "tau2": 4,
            },
            "I": {
                "var": 15,  # 10,  # 12,  # 15,
                "delay": 0,
                "null_prob": 0.8,
                "pref_prob": 0.05,  # 0.8,
                "weight": 0.006,  # 0.006,
                "tau1": 0.5,
                "tau2": 16,
                "rev": -65,  # -60.,
                "null_offset": 0,
                "pref_offset": 0,
            },
            "NMDA": {
                "var": 7,
                "delay": 0,
                "null_prob": 0.8,  # 0.8,  # 0.4,  # 0.8,
                "pref_prob": 0.8,  # 0.8,  # 0.4,  # 0.8,
                "weight": 0.002,  # 0.002,  # 0.0045,
                "tau1": 2,
                "tau2": 7,
            },
            "AMPA": {
                "var": 7,
                "delay": 0,
                "null_prob": 0.0,
                "pref_prob": 0.0,
                "weight": 0.0015,
                "tau1": 0.1,
                "tau2": 4,
            },
        },
        # stimulus
        "light_bar": {"speed": 1.0, "x_motion": True, "x_start": -60, "y_start": -70},
        "record_tree": record_tree,
    }

    # TTX
    if ttx:
        params["soma_na"] = 0.0
        params["initial_na"] = 0.0
        params["dend_na"] = 0.0

    # MORE LEAK
    if leaky:
        params["soma_gleak_hh"] = 0.0001667 * 2
        params["soma_gleak_pas"] = 0.0001667 * 2
        params["initial_gleak_hh"] = 0.0001667 * 2
        params["initial_gleak_pas"] = 0.0001667 * 2
        params["dend_gleak_hh"] = 0.0001667 * 2
        params["dend_gleak_pas"] = 0.0001667 * 2

    # MORE Kv
    if high_kv:
        params["soma_k"] = 0.15
        params["initial_k"] = 0.15
        # params["dend_K"]  = 0.15
        # params["synprops"]["E"]["weight"]    *= 1.5
        # params["synprops"]["NMDA"]["weight"] *= 0
        # params["synprops"]["I"]["weight"]    *= 2.

    if vc_mode:
        params["vc_pas"] = True

    if poisson_rates is not None:
        params["poisson_mode"] = True
        params["tstop"] = 550
        params["sac_rate"] = poisson_rates["sac"]
        params["glut_rate"] = poisson_rates["glut"]
        params["rate_dt"] = poisson_rates["dt"] * 1000.0
        for t in ["E", "I", "NMDA", "AMPA"]:
            params["synprops"][t]["tau1"] = 0.14
            params["synprops"][t]["tau2"] = 0.54

        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2278860/ (ach mini)
        # https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6772523/ (gaba mini)
        params["synprops"]["I"]["tau2"] = 2.0
        params["synprops"]["E"]["tau2"] = 0.3
        params["synprops"]["NMDA"]["tau2"] = 1.0
        params["synprops"]["E"]["delay"] = 0.0

        base_w = 0.000313 * 3 * 1.5
        params["synprops"]["E"]["weight"] = base_w
        params["synprops"]["I"]["weight"] = base_w * 1  # * 2  # * 4
        params["synprops"]["NMDA"]["weight"] = base_w * 1.33

    return params
