import h5py as h5


def merge(old, new):
    """Recursively merge nested dictionaries. Used to updating parameter
    dictionaries."""
    for k, v in new.items():
        if type(v) is dict:
            old[k] = merge(old[k], v)
        else:
            old[k] = v

    return old


def pack_hdf(pth, data_dict):
    """Takes data organized in a python dict, and creates an hdf5 with the
    same structure."""
    def rec(data, grp):
        for k, v in data.items():
            if type(v) is dict:
                rec(v, grp.create_group(k))
            else:
                grp.create_dataset(k, data=v)

    with h5.File(pth + ".h5", "w") as pckg:
        rec(data_dict, pckg)


def unpack_hdf(group):
    """Recursively unpack an hdf5 of nested Groups (and Datasets) to dict."""
    return {
        k: v[()] if type(v) is h5._hl.dataset.Dataset else unpack_hdf(v)
        for k, v in group.items()
    }


def clean_axes(axes):
    """A couple basic changes I often make to pyplot axes."""
    # wrap single axis in list
    axes = axes if hasattr(axes, "__iter__") else [axes]
    for a in axes:
        a.spines["right"].set_visible(False)
        a.spines["top"].set_visible(False)
        for ticks in (a.get_yticklabels()):
            ticks.set_fontsize(11)
