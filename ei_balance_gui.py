from ei_balance import *

import os
from skimage import io
import balance_configs as configs


if __name__ == "__main__":
    nrn_path = "/mnt/Data/NEURONoutput/sac_net/"
    # proj_path = os.path.join(nrn_path, "ttx/")
    # proj_path = os.path.join(nrn_path, "gaba_titration/ttx/")
    # proj_path = os.path.join(nrn_path, "committee_runs/")
    proj_path = os.path.join(nrn_path, "test_sigmoid/spiking/")
    # proj_path = os.path.join(nrn_path, "test_sigmoid/voltage_clamp/")

    os.makedirs(proj_path, exist_ok=True)

    h.xopen("new_balance.ses")  # open neuron gui session

    dsgc = Model(configs.sac_mode_config())
    # dsgc = Model(configs.offset_mode_config())
    rig = Rig(dsgc, proj_path)
    rig.synaptic_density()

    # rig.dir_run(3)
    # rig.rough_rho_compare(3)
    rig.sac_net_run(n_nets=3, n_trials=3, rho_steps=[0.0, 1.0])
    # rig.gaba_coverage_run(n_nets=3, n_trials=3, rho_steps=[0., 1.])
    # rig.gaba_titration_run(n_nets=3, n_trials=3, rho_steps=[0., 1.])
    # rig.vc_dir_run(10)
    # rig.sac_net_vc_run(n_nets=3, n_trials=10)
    # rig.offset_run(1)
