from neuron import h
import h5py as h5

from copy import deepcopy
import multiprocessing

from ei_balance import Model
from Rig import Rig
from hdf_utils import pack_dataset

h.load_file("stdgui.hoc")  # headless, still sets up environment


def sacnet_run(
    save_path,
    model_config,
    n_nets=3,
    n_trials=3,
    rho_steps=[0.0, 1.0],
    pool_sz=8,
):
    global _sacnet_repeat  # required to allow pickling for Pool

    def _sacnet_repeat(i):
        params = deepcopy(model_config)
        params["seed"] = i
        dsgc = Model(params)
        runner = Rig(dsgc)

        data = {}
        for rho in rho_steps:
            runner.model.nz_seed = 0
            runner.model.build_sac_net(rho=rho)
            data[rho] = runner.dir_run(
                n_trials, save_name=None, plot_summary=False, quiet=True
            )

        return data

    with multiprocessing.Pool(pool_sz) as pool, h5.File(save_path, "w") as pckg:
        idx = 0
        while idx < n_nets:
            n = min(pool_sz, n_nets - idx)
            print(
                "sac net trials %i to %i (of %i)..." % (idx + 1, idx + n, n_nets),
                flush=True,
            )
            res = pool.map(_sacnet_repeat, [idx + i for i in range(n)])
            for i in range(n):
                # pack_dataset(pckg, {idx + i: res[0]}, compression=None)
                data = {r: {idx + i: res[0][r]} for r in res[0].keys()}
                pack_dataset(pckg, data, compression=None)
                del data, res[0]  # delete head
            idx = idx + n
    print("Done!")


if __name__ == "__main__":
    import balance_configs as configs

    save_path = "/mnt/Data/NEURONoutput/sac_net/test_par_exp.h5"
    sacnet_run(
        save_path,
        configs.sac_mode_config(),
        n_nets=3,
        n_trials=3,
        rho_steps=[0.0, 1.0],
        pool_sz=8,
    )
