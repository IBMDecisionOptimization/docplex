# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2022
# --------------------------------------------------------------------------


# gendoc: ignore

try:
    import scipy.sparse as sp
except ImportError:
    sp = None

def is_scipy_sparse(m):
    return sp and sp.issparse(m)