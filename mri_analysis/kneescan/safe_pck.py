"""Load KneeMRI .pck volumes without running arbitrary pickle code: only numpy array reconstruction is allowed."""
import pickle, codecs, numpy as np
try:
    from numpy._core.multiarray import _reconstruct
except ImportError:  # older numpy
    from numpy.core.multiarray import _reconstruct
ALLOWED = {('numpy', 'dtype'): np.dtype, ('numpy', 'ndarray'): np.ndarray,
           ('numpy.core.multiarray', '_reconstruct'): _reconstruct, ('numpy._core.multiarray', '_reconstruct'): _reconstruct,
           ('_codecs', 'encode'): codecs.encode}


class _U(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in ALLOWED: return ALLOWED[(module, name)]
        raise pickle.UnpicklingError(f'blocked global {module}.{name}')


def load(path):
    with open(path, 'rb') as f: obj = _U(f, encoding='latin1').load()
    if not isinstance(obj, np.ndarray): raise ValueError('not an array')
    return obj
