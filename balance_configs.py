"""
Functions returning parameter dictionaries for the DSGC balance model.
"""


def sac_mode_config():
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
    if 0:
        params["soma_Na"] = 0.0
        params["prime_Na"] = 0.0
        params["dend_Na"] = 0.0

    # MORE LEAK
    if 0:
        params["soma_gleak_hh"] = 0.0001667 * 2
        params["soma_gleak_pas"] = 0.0001667 * 2
        params["prime_gleak_hh"] = 0.0001667 * 2
        params["prime_gleak_pas"] = 0.0001667 * 2
        params["dend_gleak_hh"] = 0.0001667 * 2
        params["dend_gleak_pas"] = 0.0001667 * 2

    # MORE Kv
    if 0:
        params["soma_K"] = 0.15
        params["prime_K"] = 0.15
        # params["dend_K"]  = 0.15
        # params["synprops"]["E"]["weight"]    *= 1.5
        # params["synprops"]["NMDA"]["weight"] *= 0
        # params["synprops"]["I"]["weight"]    *= 2.

    if 0:
        params["vc_pas"] = True

    # non-ds ACH
    if 0:
        params["synprops"]["E"]["null_prob"] = 0.5
        params["synprops"]["E"]["pref_prob"] = 0.5

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
