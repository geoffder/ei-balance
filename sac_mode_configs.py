import numpy as np

"""
Functions returning parameter dictionaries for the DSGC balance model.
"""


def conf(
    ttx=False,
    non_ds_ach=False,
    offset_ampa_ach=False,
    vc_mode=False,
    record_tree=True,
    poisson_rates=None,
    plexus=0,
    plexus_share=None,
):
    gleak_hh = 0.0001667  # NOTE: USUAL
    # gleak_hh = 0.000275  # poleg-polsky 2016
    eleak_hh = -60

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
        "vc_pas": vc_mode,
        "vshift_hh": 0,
        "soma_Na": 0.2,
        "soma_K": 0.07,
        "soma_eleak_hh": eleak_hh,
        "soma_gleak_hh": gleak_hh,
        "soma_gleak_pas": 0.0001667,  # alon 2016 -> 5e-5
        "prime_Na": 0.07,
        "prime_K": 0.07,
        "prime_eleak_hh": eleak_hh,
        "prime_gleak_hh": gleak_hh,
        "prime_gleak_pas": 0.0001667,
        "dend_Na": 0.05,
        "dend_K": 0.07,
        "dend_eleak_hh": eleak_hh,
        "dend_gleak_hh": gleak_hh,
        "dend_gleak_pas": 0.0001667,
        # synapse variability
        "jitter": 0,
        "space_rho": 0.9,
        "time_rho": 0.9,
        "synprops": {
            "E": {
                "var": 15,
                "delay": 5,
                "null_prob": 0.0,
                "pref_prob": 1.0,  # 0.8,
                "weight": 0.0015,
                "tau1": 0.1,
                "tau2": 4,
            },
            "I": {
                "var": 15,
                "delay": 0,
                "null_prob": 0.0,
                "pref_prob": 1.0,  # 0.8,
                "weight": 0.006,
                "tau1": 0.5,
                "tau2": 16,
                "rev": -65,
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
        "light_bar": {
            "speed": 1.0,
            "x_motion": True,
            "x_start": -60,
            "y_start": -70,
            "rel_start_pos": np.array([-150, 0]),
            "rel_end_pos": np.array([150, 0]),
        },
        # SAC Network
        "sac_mode": True,
        "sac_rho": 1.0,
        "sac_theta_mode": "experimental",
        "sac_angle_rho_mode": True,
        "sac_gaba_coverage": 0.5,
        "sac_offset": 30,
        "record_tree": record_tree,
        "n_plexus_ach": 0,
        "diam_scaling_mode": "cable",
        "diam_range": {"max": 1.5, "min": 0.4, "decay": 0.92, "scale": 1.0},
    }

    # TTX
    if ttx:
        params["soma_Na"] = 0.0
        params["prime_Na"] = 0.0
        params["dend_Na"] = 0.0

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
        params["tstop"] = 550 if not vc_mode else 750
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


def decr_weight(
    ttx=False,
    non_ds_ach=False,
    offset_ampa_ach=False,
    vc_mode=False,
    record_tree=True,
    poisson_rates=None,
    plexus=0,
    plexus_share=None,
):
    gleak_hh = 0.0001667  # NOTE: USUAL
    # gleak_hh = 0.000275  # poleg-polsky 2016
    eleak_hh = -65

    # TODO: adjust active properties of the dendrites (Na/K)
    # would like to turn down E a bit and *increase* spiking if possible
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
        "vc_pas": vc_mode,
        "vshift_hh": -1,
        "soma_Ra": 200,  # NOTE: NEW
        "dend_Ra": 200,  # NOTE: NEW
        "soma_Na": 0.2,
        "soma_K": 0.07,
        "soma_Km": 0.0005,  # NOTE: NEW
        "soma_eleak_hh": eleak_hh,
        "soma_gleak_hh": gleak_hh,
        "soma_gleak_pas": 0.0001667,  # alon 2016 -> 5e-5
        "prime_Na": 0.07,  # 0.045,  # 0.03 # 0.1,  # 0.011,
        "prime_K": 0.07,  # 0.035,  # 0.03, # NOTE: USUAL
        # "prime_K": 0.035,  # 0.035,  # 0.03,
        # "prime_Km": 0.0005,  # NOTE: NEW (decreasing weights)
        "prime_Km": 0.0,  # NOTE: NEW (decreasing weights)
        "prime_eleak_hh": eleak_hh,
        "prime_gleak_hh": gleak_hh,
        "prime_gleak_pas": 0.0001667,
        # "dend_Na": 0.013,  # NOTE: USUAL
        # "dend_Na": 0.035,
        "dend_Na": 0.05,  # 0.011, # NOTE: usual with cable on
        # "dend_K": 0.035,  # 0.03,  # 0.035,  # 0.03, # NOTE: USUAL
        "dend_K": 0.07,  # NOTE: usual with cable on
        "dend_Km": 0.0,  # NOTE: NEW (decreasing weights)
        "dend_eleak_hh": eleak_hh,
        "dend_gleak_hh": gleak_hh,
        "dend_gleak_pas": 0.0001667,
        # synapse variability
        "jitter": 0,
        "space_rho": 0.9,
        "time_rho": 0.9,
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
                "rev": -70,  # -65,  # -60.,
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
        "light_bar": {
            "speed": 1.0,
            "x_motion": True,
            "x_start": -60,
            "y_start": -70,
            "rel_start_pos": np.array([-150, 0]),
            "rel_end_pos": np.array([150, 0]),
        },
        # SAC Network
        "sac_mode": True,
        "sac_rho": 1.0,
        "sac_theta_mode": "experimental",
        "sac_angle_rho_mode": True,
        "sac_gaba_coverage": 0.5,
        "sac_offset": 30,
        "n_plexus_ach": 0,
        "diam_scaling_mode": "cable",
        "diam_range": {"max": 1.5, "min": 0.4, "decay": 0.92, "scale": 1.0},
        "record_tree": record_tree,
    }

    # TTX
    if ttx:
        params["soma_Na"] = 0.0
        params["prime_Na"] = 0.0
        params["dend_Na"] = 0.0

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
        params["tstop"] = 550 if not vc_mode else 750
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
        # params["synprops"]["NMDA"]["tau2"] = 1.0
        params["synprops"]["NMDA"]["tau2"] = 2.0
        params["synprops"]["E"]["delay"] = 0.0

        # base_w = 0.000313 * 3 * 1.5  # NOTE: USUAL
        # base_w = 0.000313 * 2
        base_w = 0.000313
        params["synprops"]["E"]["weight"] = base_w
        # params["synprops"]["E"]["weight"] = base_w * 0.8
        # params["synprops"]["I"]["weight"] = base_w * 4  # NOTE: USUAL
        # params["synprops"]["I"]["weight"] = base_w * 3
        params["synprops"]["I"]["weight"] = base_w * 1.2  # NOTE: USUAL
        # params["synprops"]["I"]["weight"] = base_w
        # params["synprops"]["I"]["weight"] = base_w
        # params["synprops"]["NMDA"]["weight"] = base_w * 1.33  # NOTE: USUAL
        params["synprops"]["NMDA"]["weight"] = base_w * 1.5
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