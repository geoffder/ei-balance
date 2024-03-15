from neuron import h
import h5py as h5

from copy import deepcopy
import multiprocessing
from functools import partial

from ei_balance import Model
from Rig import Rig
from hdf_utils import pack_dataset, pack_key

h.load_file("stdgui.hoc")  # headless, still sets up environment


def sacnet_run(
    save_path,
    model_config,
    n_nets=3,
    n_trials=3,
    rho_steps=[0.0, 1.0],
    pool_sz=8,
    vc_mode=False,
    vc_simul=True,
    vc_isolate=True,
    reset_seed_between_rho=False,
    reset_rng_before_runs=False,
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
            runner.model.build_sac_net(rho=rho, reset_rng=reset_seed_between_rho)
            if reset_rng_before_runs:
                runner.model.reset_rng()
            if vc_mode:
                data[rho] = runner.vc_dir_run(
                    n_trials,
                    simultaneous=vc_simul,
                    isolate_agonists=vc_isolate,
                    save_name=None,
                    quiet=True,
                )
            else:
                data[rho] = runner.dir_run(
                    n_trials, save_name=None, plot_summary=False, quiet=True
                )

        return data

    if pool_sz > 1:
        with multiprocessing.Pool(pool_sz) as pool, h5.File(save_path, "w") as pckg:
            idx = 0
            while idx < n_nets:
                n = min(pool_sz, n_nets - idx)
                print(
                    "sac net trials %i to %i (of %i)..." % (idx + 1, idx + n, n_nets),
                    flush=True,
                )
                res = pool.map(_sacnet_repeat, [idx + i for i in range(n)])
                for _ in range(n):
                    data = {r: {idx: res[0][r]} for r in res[0].keys()}
                    pack_dataset(pckg, data, compression=None)
                    del data, res[0]  # delete head
                    idx += 1
    else:
        with h5.File(save_path, "w") as pckg:
            for n in range(n_nets):
                e = "\r" if n < n_nets - 1 else "\n"
                print("sac net trial %i of %i..." % (n + 1, n_nets), end=e, flush=True)
                res = _sacnet_repeat(n)
                data = {r: {n: res[r]} for r in res.keys()}
                pack_dataset(pckg, data, compression=None)
                del data, res  # delete head
    print("Done!")


def sacnet_titration_run(
    save_path,
    model_config,
    param_paths,
    titration_steps,
    n_nets=3,
    n_trials=3,
    rho_steps=[0.0, 1.0],
    pool_sz=8,
    vc_mode=False,
    vc_simul=True,
    vc_isolate=True,
    reset_seed_between_rho=False,
):
    """'Titration' of the parameters pointed to by param_paths (string list or
    string list list) in the given model_config. titration_steps is still a
    float list, so each parameter is scaled by the same multiplication factor at
    each step. For context, this extension beyond the original gaba titration
    script is mainly to enable scaling ACh probability which requires changing
    'E' and 'PLEX' at the same time."""
    global _sacnet_titration_repeat  # required to allow pickling for Pool
    param_paths = [param_paths] if type(param_paths[0]) != list else param_paths

    def new_params(factor):
        params = deepcopy(model_config)
        for pth in param_paths:
            param_ref = params
            for p in pth[:-1]:
                param_ref = param_ref[p]
            param_ref[pth[-1]] *= factor
        return params

    lbls = map(lambda p: "/".join(p), param_paths)
    lbl = " and ".join(lbls)

    def _sacnet_titration_repeat(step, i):
        params = new_params(step)
        params["seed"] = i
        dsgc = Model(params)
        runner = Rig(dsgc)

        data = {}
        for rho in rho_steps:
            runner.model.nz_seed = 0
            runner.model.build_sac_net(rho=rho, reset_rng=reset_seed_between_rho)
            if vc_mode:
                data[rho] = runner.vc_dir_run(
                    n_trials,
                    simultaneous=vc_simul,
                    isolate_agonists=vc_isolate,
                    save_name=None,
                    quiet=True,
                )
            else:
                data[rho] = runner.dir_run(
                    n_trials, save_name=None, plot_summary=False, quiet=True
                )

        return data

    with multiprocessing.Pool(pool_sz) as pool, h5.File(save_path, "w") as pckg:
        for factor in titration_steps:
            print("Running with %s scaled by factor of %.2f" % (lbl, factor))
            grp = pckg.create_group(pack_key(factor))
            f = partial(_sacnet_titration_repeat, factor)
            idx = 0
            while idx < n_nets:
                n = min(pool_sz, n_nets - idx)
                print(
                    "  sac net trials %i to %i (of %i)..." % (idx + 1, idx + n, n_nets),
                    flush=True,
                )
                res = pool.map(f, [idx + i for i in range(n)])
                for _ in range(n):
                    data = {r: {idx: res[0][r]} for r in res[0].keys()}
                    pack_dataset(grp, data, compression=None)
                    del data, res[0]  # delete head
                    idx += 1
    print("Done!")


def sacnet_gaba_titration_run(
    save_path,
    model_config,
    n_nets=3,
    n_trials=3,
    rho_steps=[0.0, 1.0],
    gaba_steps=[round(s * 0.1, 2) for s in range(1, 16)],
    pool_sz=8,
    vc_mode=False,
    vc_simul=True,
    vc_isolate=True,
    reset_seed_between_rho=False,
):
    param_path = ["synprops", "I", "weight"]
    sacnet_titration_run(
        save_path,
        model_config,
        param_path,
        titration_steps=gaba_steps,
        n_nets=n_nets,
        n_trials=n_trials,
        rho_steps=rho_steps,
        pool_sz=pool_sz,
        vc_mode=vc_mode,
        vc_simul=vc_simul,
        vc_isolate=vc_isolate,
        reset_seed_between_rho=reset_seed_between_rho,
    )


def sacnet_gaba_titration_run_(
    save_path,
    model_config,
    n_nets=3,
    n_trials=3,
    rho_steps=[0.0, 1.0],
    gaba_steps=[round(s * 0.1, 2) for s in range(1, 16)],
    pool_sz=8,
    vc_mode=False,
    vc_simul=True,
    vc_isolate=True,
    reset_seed_between_rho=False,
):
    global _sacnet_gaba_titration_repeat  # required to allow pickling for Pool

    def _sacnet_gaba_titration_repeat(factor, i):
        params = deepcopy(model_config)
        params["seed"] = i
        params["synprops"]["I"]["weight"] = params["synprops"]["I"]["weight"] * factor
        dsgc = Model(params)
        runner = Rig(dsgc)

        data = {}
        for rho in rho_steps:
            runner.model.nz_seed = 0
            runner.model.build_sac_net(rho=rho, reset_rng=reset_seed_between_rho)
            if vc_mode:
                data[rho] = runner.vc_dir_run(
                    n_trials,
                    simultaneous=vc_simul,
                    isolate_agonists=vc_isolate,
                    save_name=None,
                    quiet=True,
                )
            else:
                data[rho] = runner.dir_run(
                    n_trials, save_name=None, plot_summary=False, quiet=True
                )

        return data

    with multiprocessing.Pool(pool_sz) as pool, h5.File(save_path, "w") as pckg:
        for factor in gaba_steps:
            print("Running with GABA scaled by factor of %.2f" % factor)
            grp = pckg.create_group(pack_key(factor))
            f = partial(_sacnet_gaba_titration_repeat, factor)
            idx = 0
            while idx < n_nets:
                n = min(pool_sz, n_nets - idx)
                print(
                    "  sac net trials %i to %i (of %i)..." % (idx + 1, idx + n, n_nets),
                    flush=True,
                )
                res = pool.map(f, [idx + i for i in range(n)])
                for _ in range(n):
                    data = {r: {idx: res[0][r]} for r in res[0].keys()}
                    pack_dataset(grp, data, compression=None)
                    del data, res[0]  # delete head
                    idx += 1
    print("Done!")


def sacnet_rho_run(
    save_path,
    model_config,
    n_nets=3,
    n_trials=3,
    rho_steps=[round(s * 0.1, 2) for s in range(0, 11)],
    pool_sz=8,
    vc_mode=False,
    vc_simul=True,
    vc_isolate=True,
):
    global _sacnet_rho_repeat  # required to allow pickling for Pool

    def _sacnet_rho_repeat(rho, i):
        params = deepcopy(model_config)
        params["seed"] = i
        dsgc = Model(params)
        runner = Rig(dsgc)

        runner.model.nz_seed = 0
        runner.model.build_sac_net(rho=rho)
        if vc_mode:
            return runner.vc_dir_run(
                n_trials,
                simultaneous=vc_simul,
                isolate_agonists=vc_isolate,
                save_name=None,
                quiet=True,
            )
        else:
            return runner.dir_run(
                n_trials, save_name=None, plot_summary=False, quiet=True
            )

    with multiprocessing.Pool(pool_sz) as pool, h5.File(save_path, "w") as pckg:
        for rho in rho_steps:
            print("Running with rho = %.2f" % rho)
            grp = pckg.create_group(pack_key(rho))
            f = partial(_sacnet_rho_repeat, rho)
            idx = 0
            while idx < n_nets:
                n = min(pool_sz, n_nets - idx)
                print(
                    "  sac net trials %i to %i (of %i)..." % (idx + 1, idx + n, n_nets),
                    flush=True,
                )
                res = pool.map(f, [idx + i for i in range(n)])
                for _ in range(n):
                    data = {idx: res[0]}
                    pack_dataset(grp, data, compression=None)
                    del data, res[0]  # delete head
                    idx += 1
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
