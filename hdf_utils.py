import numpy as np
import h5py as h5
from typing import Optional, Union, Any

INT_PREFIX = "packed_int_"
FLOAT_PREFIX = "packed_float_"


def pack_key(k):
    if isinstance(k, int):
        return "%s%i" % (INT_PREFIX, k)
    elif isinstance(k, float):
        return "%s%f" % (FLOAT_PREFIX, k)
    else:
        return k


def unpack_key(k):
    if isinstance(k, str):
        if k.startswith(INT_PREFIX):
            return int(k[len(INT_PREFIX) :])
        if k.startswith(FLOAT_PREFIX):
            return float(k[len(FLOAT_PREFIX) :])
        return k
    else:
        return k


def pack_dataset(
    h5_file: h5.File, data_dict: dict[Any, Any], compression: Optional[str] = None
):
    """Takes data organized in a python dict, and stores it in the given hdf5
    with the same structure. Keys are converted to strings to comply to hdf5
    group naming convention. In `unpack_hdf`, if the key is all digits, it will
    be converted back from string."""

    def rec(data, grp):
        for k, v in data.items():
            k = pack_key(k)

            if type(v) is dict:
                if k in grp:
                    rec(v, grp[k])
                else:
                    rec(v, grp.create_group(k))
            elif isinstance(v, (float, int, str, np.integer)):  # type: ignore
                grp.create_dataset(k, data=v)  # can't compress scalars
            else:
                grp.create_dataset(k, data=v, compression=compression)

    rec(data_dict, h5_file)


def pack_hdf(pth, data_dict, compression: Optional[str] = None):
    """Takes data organized in a python dict, and creates an hdf5 with the
    same structure. Keys are converted to strings to comply to hdf5 group naming
    convention. In `unpack_hdf`, if the key is all digits, it will be converted
    back from string."""
    with h5.File(pth + ".h5", "w") as pckg:
        pack_dataset(pckg, data_dict, compression=compression)


def unpack_hdf_(group):
    """Recursively unpack an hdf5 of nested Groups (and Datasets) to dict."""

    return {
        unpack_key(k): v[()] if type(v) is h5.Dataset else unpack_hdf(v)
        for k, v in group.items()
    }


def unpack_hdf(h5_group):
    """Recursively unpack an hdf5 of nested Groups (and Datasets) to dict."""
    d = {}
    items = [list(h5_group.items())]
    grps = [d]
    while True:
        if len(items) > 0 and len(items[-1]) > 0:
            k, v = items[-1].pop()
            k = unpack_key(k)
            if type(v) is h5.Dataset:
                grps[-1][k] = v[()]
            else:
                grps[-1][k] = {}
                grps.append(grps[-1][k])
                items.append(list(v.items()))
        elif len(items) > 0:
            items.pop()
            grps.pop()
        else:
            break
    return d


class Workspace:
    _data: Union[dict, h5.File]

    def __init__(self, path=None):
        if path is None:
            self.is_hdf = False
            self._data = {}
        else:
            self.is_hdf = True
            self._data = h5.File(path, mode="w")

    def __setitem__(self, key, item):
        if self.is_hdf and key in self._data:
            try:
                self._data[key][...] = item  # type:ignore
            except TypeError:  # if new item has a different shape
                del self._data[key]
                self._data[key] = item
        elif self.is_hdf and type(item) is dict:
            if key in self._data:
                del self._data[key]
            pack_dataset(self._data, {key: item})  # type:ignore
        else:
            self._data[key] = item

    def __getitem__(self, key):
        return self._data[key]

    def __repr__(self):
        return repr(self._data)

    def __len__(self):
        return len(self._data)

    def __delitem__(self, key):
        del self._data[key]

    def __contains__(self, item):
        return item in self._data

    def __iter__(self):
        return iter(self._data)

    def has_key(self, k):
        return k in self._data

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def create_group(self, key):
        if self.is_hdf:
            if key in self._data:
                del self._data[key]
            return self._data.create_group(key)  # type:ignore
        else:
            self._data[key] = {}
            return self._data[key]

    def close(self):
        if self.is_hdf:
            self._data.close()  # type:ignore
