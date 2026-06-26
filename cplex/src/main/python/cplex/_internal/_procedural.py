# --------------------------------------------------------------------------
# File: _procedural.py
# ---------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2008, 2026. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# ------------------------------------------------------------------------
"""Wrapper functions for the CPLEX C Callable Library"""
from collections import namedtuple
from contextlib import contextmanager
import os

from . import _constants as _const
from . import _list_array_utils as LAU
from . import _pycplex as CR

from ..exceptions import CplexSolverError, CplexError, ErrorChannelMessage

CPLEX_PY_DISABLE_SIGHANDLER = os.getenv("CPLEX_PY_DISABLE_SIGHANDLER")

# pylint: disable=missing-docstring


def _safeDoubleArray(arraylen):
    # Make sure that we never request a zero-length array.  This results in
    # a malloc(0) call in the SWIG layer.  On AIX this returns NULL which
    # causes problems.  By ensuring that the array is at least size 1, we
    # avoid these problems and the overhead should be negligable.
    if arraylen <= 0:
        arraylen = 1
    return CR.doubleArray(arraylen)


def _safeIntArray(arraylen):
    # See comment for _safeDoubleArray above.
    if arraylen <= 0:
        arraylen = 1
    return CR.intArray(arraylen)


def _safeLongArray(arraylen):
    # See comment for _safeDoubleArray above.
    if arraylen <= 0:
        arraylen = 1
    return CR.longArray(arraylen)


def _arraylen(seq):
    """If seq is None, return 0, else len(seq).

    CPLEX often requires a count argument to specify the length of
    subsequent array arguments. This function allows us to return a
    length of 0 for None (i.e., NULL) arrays.
    """
    if seq is None:
        return 0
    return len(seq)


def _rangelen(begin, end):
    """Returns length of the range specified by begin and end.

    As this is typically used to calculate the length of a buffer, it
    always returns a result >= 0.

    See functions like `_safeDoubleArray` and `safeLongArray`.
    """
    # We allow arguments like begin=0, end=-1 on purpose. This represents
    # an empty range; the callable library should do nothing in this case
    # (see RTC-31484).
    result = end - begin + 1
    if result < 0:
        return 0
    return result


def getstatstring(env, statind):
    output = []
    CR.CPXXgetstatstring(env, statind, output)
    return output[0]


def geterrorstring(env, errcode):
    output = []
    CR.CPXXgeterrorstring(env, errcode, output)
    return output[0]


def cb_geterrorstring(env, status):
    return CR.cb_geterrorstring(env, status)


def new_native_int():
    return CR.new_native_int()


def delete_native_int(p):
    CR.delete_native_int(p)


def set_native_int(p, v):
    CR.set_native_int(p, v)


def get_native_int(p):
    return CR.get_native_int(p)


def setterminate(env, env_lp_ptr, p):
    status = CR.setterminate(env_lp_ptr, p)
    check_status(env, status)


# If the CPLEX_PY_DISABLE_SIGHANDLER environment variable is defined,
# we will not install our SIGINT handler (i.e., for Ctrl+C handling).
# This may be useful if the user wants to install their own handler.
if CPLEX_PY_DISABLE_SIGHANDLER:

    class SigIntHandler():
        """A no-op signal handler (no handler installed).

        :undocumented
        """

        def __init__(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            pass

else:

    # By default (i.e., if the CPLEX_PY_DISABLE_SIGHANDLER environment
    # variable is not defined), then we install a custom SIGINT handler
    # around long running CPLEX operations. This allows the user to abort
    # the current optimization with a Ctrl+C.

    class SigIntHandler():
        """Handle Ctrl-C events during long running processes.

        :undocumented
        """

        def __init__(self):
            CR.sigint_register()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            CR.sigint_unregister()


def pack_env_lp_ptr(env, lp):
    return CR.pack_env_lp_ptr(env, lp)


@contextmanager
def chbmatrix(lolmat, env_lp_ptr, r_c):
    """See matrix_conversion.c:Pylolmat_to_CHBmat()."""
    mat = Pylolmat_to_CHBmat(lolmat, env_lp_ptr, r_c)
    try:
        # yields ([matbeg, matind, matval], nnz)
        yield mat[:-1], mat[-1]
    finally:
        free_CHBmat(mat)


def Pylolmat_to_CHBmat(lolmat, env_lp_ptr, r_c):
    return CR.Pylolmat_to_CHBmat(lolmat, env_lp_ptr, r_c)


def free_CHBmat(lolmat):
    CR.free_CHBmat(lolmat)


def _handle_cb_error(env, cberror):
    """Handle the callback exception.

    These can be triggered either in the SWIG Python C API layer
    (e.g., SWIG_callback.c) or in _ostream.py.
    """
    if isinstance(cberror, Exception):
        # If cberror is already an exception, then just throw it as is.
        # We can only get here from: _ostream.py:_write_wrap.
        raise cberror
    if isinstance(cberror[1], Exception):
        # In this case the first item is the type of exception and
        # the second item is the exception.  This is raised from the
        # SWIG C layer (e.g., SWIG_callback.c:).
        cberror = cberror[1]
    elif isinstance(cberror[1], tuple):
        # The second item is a tuple containing the error string and
        # the error number.  We can get this from, for example:
        # SWIG_callback.c:fast_getcallbackinfo.
        assert len(cberror[1]) == 2
        cberror = cberror[0](cberror[1][0], env, cberror[1][1])
    else:
        # The second item is the error string or perhaps None.
        # See code in SWIG_callback.c where the _callback_exception
        # attribute is set.
        cberror = cberror[0](cberror[1])
    raise cberror


class StatusChecker():
    """A callable object used for checking status codes.

    :undocumented
    """

    def __init__(self):
        class NoOp():
            pass
        self._pyenv = NoOp()
        self._pyenv._callback_exception = None

    def __call__(self, env, status, from_cb=False):
        error_string = None
        try:
            if self._pyenv._callback_exception is not None:
                callback_exception = self._pyenv._callback_exception
                self._pyenv._callback_exception = None
                if isinstance(callback_exception, ErrorChannelMessage):
                    # We can only get here from _ostream.py:_write_wrap.
                    # If we encounter an error, we use the last message
                    # from the error channel for the message (i.e., rather
                    # than calling CPXXgeterrorstring).
                    error_string = callback_exception.args[0]
                else:
                    _handle_cb_error(env, callback_exception)
        except ReferenceError:
            pass
        if status == CR.CPXERR_NO_ENVIRONMENT:
            raise ValueError('illegal method invocation after Cplex.end()')
        elif status != 0:
            if error_string is None:
                if from_cb:
                    error_string = cb_geterrorstring(env, status)
                else:
                    error_string = geterrorstring(env, status)
            raise CplexSolverError(error_string, env, status)


check_status = StatusChecker()


def set_status_checker():
    CR.set_status_checker(check_status)

# Environment


def version(env):
    return CR.CPXXversion(env)


def versionnumber(env):
    ver = CR.intPtr()
    status = CR.CPXXversionnumber(env, ver)
    check_status(env, status)
    return ver.value()


def openCPLEX():
    status = CR.intPtr()
    env = CR.CPXXopenCPLEX(status)
    check_status(env, status.value())
    # Always set the pyterminate flag immediately when initializing
    # a CPLEX environment.
    CR.setpyterminate(env)
    return env


def closeCPLEX(env):
    envp = CR.CPXENVptrPtr()
    envp.assign(env)
    status = CR.CPXXcloseCPLEX(envp)
    check_status(env, status)


def getchannels(env):
    results = CR.CPXCHANNELptrPtr()
    warning = CR.CPXCHANNELptrPtr()
    error = CR.CPXCHANNELptrPtr()
    log = CR.CPXCHANNELptrPtr()
    status = CR.CPXXgetchannels(env, results, warning, error, log)
    check_status(env, status)
    return (results.value(), warning.value(), error.value(), log.value())


def addfuncdest(env, channel, fileobj):
    status = CR.CPXXaddfuncdest(env, channel, fileobj)
    check_status(env, status)


def delfuncdest(env, channel, fileobj):
    status = CR.CPXXdelfuncdest(env, channel, fileobj)
    check_status(env, status)


def setlpcallbackfunc(env, cbhandle):
    status = CR.CPXXsetlpcallbackfunc(env, cbhandle)
    check_status(env, status)


def setnetcallbackfunc(env, cbhandle):
    status = CR.CPXXsetnetcallbackfunc(env, cbhandle)
    check_status(env, status)


def settuningcallbackfunc(env, cbhandle):
    status = CR.CPXXsettuningcallbackfunc(env, cbhandle)
    check_status(env, status)


def setheuristiccallbackfunc(env, cbhandle):
    status = CR.CPXXsetheuristiccallbackfunc(env, cbhandle)
    check_status(env, status)


def setlazyconstraintcallbackfunc(env, cbhandle):
    status = CR.CPXXsetlazyconstraintcallbackfunc(env, cbhandle)
    check_status(env, status)


def setusercutcallbackfunc(env, cbhandle):
    status = CR.CPXXsetusercutcallbackfunc(env, cbhandle)
    check_status(env, status)


def setincumbentcallbackfunc(env, cbhandle):
    status = CR.CPXXsetincumbentcallbackfunc(env, cbhandle)
    check_status(env, status)


def setnodecallbackfunc(env, cbhandle):
    status = CR.CPXXsetnodecallbackfunc(env, cbhandle)
    check_status(env, status)


def setbranchcallbackfunc(env, cbhandle):
    status = CR.CPXXsetbranchcallbackfunc(env, cbhandle)
    check_status(env, status)


def setbranchnosolncallbackfunc(env, cbhandle):
    status = CR.CPXXsetbranchnosolncallbackfunc(env, cbhandle)
    check_status(env, status)


def setsolvecallbackfunc(env, cbhandle):
    status = CR.CPXXsetsolvecallbackfunc(env, cbhandle)
    check_status(env, status)


def setinfocallbackfunc(env, cbhandle):
    status = CR.CPXXsetinfocallbackfunc(env, cbhandle)
    check_status(env, status)


def setmipcallbackfunc(env, cbhandle):
    status = CR.CPXXsetmipcallbackfunc(env, cbhandle)
    check_status(env, status)

# Parameters

def setdefaults(env):
    status = CR.CPXXsetdefaults(env)
    check_status(env, status)


def setintparam(env, whichparam, newvalue):
    status = CR.CPXXsetintparam(env, whichparam, newvalue)
    check_status(env, status)


def setlongparam(env, whichparam, newvalue):
    status = CR.CPXXsetlongparam(env, whichparam, newvalue)
    check_status(env, status)


def setdblparam(env, whichparam, newvalue):
    status = CR.CPXXsetdblparam(env, whichparam, newvalue)
    check_status(env, status)


def setstrparam(env, whichparam, newvalue):
    status = CR.CPXXsetstrparam(env, whichparam, newvalue)
    check_status(env, status)


def getintparam(env, whichparam):
    curval = CR.intPtr()
    status = CR.CPXXgetintparam(env, whichparam, curval)
    check_status(env, status)
    return curval.value()


def getlongparam(env, whichparam):
    curval = CR.cpxlongPtr()
    status = CR.CPXXgetlongparam(env, whichparam, curval)
    check_status(env, status)
    return curval.value()


def getdblparam(env, whichparam):
    curval = CR.doublePtr()
    status = CR.CPXXgetdblparam(env, whichparam, curval)
    check_status(env, status)
    return curval.value()


def getstrparam(env, whichparam):
    output = []
    status = CR.CPXXgetstrparam(env, whichparam, output)
    check_status(env, status)
    return output[0]


def infointparam(env, whichparam):
    default = CR.intPtr()
    minimum = CR.intPtr()
    maximum = CR.intPtr()
    status = CR.CPXXinfointparam(env, whichparam, default, minimum, maximum)
    check_status(env, status)
    return (default.value(), minimum.value(), maximum.value())


def infolongparam(env, whichparam):
    default = CR.cpxlongPtr()
    minimum = CR.cpxlongPtr()
    maximum = CR.cpxlongPtr()
    status = CR.CPXXinfolongparam(env, whichparam, default, minimum, maximum)
    check_status(env, status)
    return (default.value(), minimum.value(), maximum.value())


def infodblparam(env, whichparam):
    default = CR.doublePtr()
    minimum = CR.doublePtr()
    maximum = CR.doublePtr()
    status = CR.CPXXinfodblparam(env, whichparam, default, minimum, maximum)
    check_status(env, status)
    return (default.value(), minimum.value(), maximum.value())


def infostrparam(env, whichparam):
    output = []
    status = CR.CPXXinfostrparam(env, whichparam, output)
    check_status(env, status)
    return output[0]


def getparamtype(env, param_name):
    output = CR.intPtr()
    status = CR.CPXXgetparamtype(env, param_name, output)
    check_status(env, status)
    return output.value()


def readcopyparam(env, filename):
    status = CR.CPXXreadcopyparam(env, filename)
    check_status(env, status)


def writeparam(env, filename):
    status = CR.CPXXwriteparam(env, filename)
    check_status(env, status)


def tuneparam(env, lp, int_param_values, dbl_param_values, str_param_values):
    tuning_status = CR.intPtr()
    intcnt = len(int_param_values)
    dblcnt = len(dbl_param_values)
    strcnt = len(str_param_values)
    intnum = [int_param_values[i][0] for i in range(intcnt)]
    intval = [int_param_values[i][1] for i in range(intcnt)]
    dblnum = [dbl_param_values[i][0] for i in range(dblcnt)]
    dblval = [dbl_param_values[i][1] for i in range(dblcnt)]
    strnum = [str_param_values[i][0] for i in range(strcnt)]
    strval = [str_param_values[i][1] for i in range(strcnt)]
    with SigIntHandler():
        status = CR.CPXXtuneparam(
            env, lp, intcnt,
            LAU.int_list_to_array(intnum),
            LAU.int_list_to_array_trunc_int32(intval),
            dblcnt,
            LAU.int_list_to_array(dblnum),
            LAU.double_list_to_array(dblval),
            strcnt,
            LAU.int_list_to_array(strnum),
            strval,
            tuning_status)
    check_status(env, status)
    return tuning_status.value()


def tuneparamprobset(env, filenames, filetypes, int_param_values,
                     dbl_param_values, str_param_values):
    tuning_status = CR.intPtr()
    intcnt = len(int_param_values)
    dblcnt = len(dbl_param_values)
    strcnt = len(str_param_values)
    intnum = [int_param_values[i][0] for i in range(intcnt)]
    intval = [int_param_values[i][1] for i in range(intcnt)]
    dblnum = [dbl_param_values[i][0] for i in range(dblcnt)]
    dblval = [dbl_param_values[i][1] for i in range(dblcnt)]
    strnum = [str_param_values[i][0] for i in range(strcnt)]
    strval = [str_param_values[i][1] for i in range(strcnt)]
    with SigIntHandler():
        status = CR.CPXXtuneparamprobset(
            env, len(filenames),
            filenames,
            filetypes,
            intcnt, LAU.int_list_to_array(intnum),
            LAU.int_list_to_array_trunc_int32(intval),
            dblcnt, LAU.int_list_to_array(dblnum),
            LAU.double_list_to_array(dblval),
            strcnt, LAU.int_list_to_array(strnum),
            strval,
            tuning_status)
    check_status(env, status)
    return tuning_status.value()


def fixparam(env, paramnum):
    status = CR.CPXXEfixparam(env, paramnum)
    check_status(env, status)

########################################################################
# Parameter Set API
########################################################################

def paramsetadd(env, ps, whichparam, newvalue, paramtype=None):
    if paramtype is None:
        paramtype = getparamtype(env, whichparam)
    if paramtype == _const.CPX_PARAMTYPE_INT:
        if isinstance(newvalue, float):
            newvalue = int(newvalue)  # will upconvert to long, if necc.
        paramsetaddint(env, ps, whichparam, newvalue)
    elif paramtype == _const.CPX_PARAMTYPE_DOUBLE:
        if isinstance(newvalue, int):
            newvalue = float(newvalue)
        paramsetadddbl(env, ps, whichparam, newvalue)
    elif paramtype == _const.CPX_PARAMTYPE_STRING:
        paramsetaddstr(env, ps, whichparam, newvalue)
    else:
        assert paramtype == _const.CPX_PARAMTYPE_LONG
        if isinstance(newvalue, float):
            newvalue = int(newvalue)  # will upconvert to long, if necc.
        paramsetaddlong(env, ps, whichparam, newvalue)


def paramsetadddbl(env, ps, whichparam, newvalue):
    status = CR.CPXXparamsetadddbl(env, ps, whichparam, newvalue)
    check_status(env, status)

def paramsetaddint(env, ps, whichparam, newvalue):
    status = CR.CPXXparamsetaddint(env, ps, whichparam, newvalue)
    check_status(env, status)

def paramsetaddlong(env, ps, whichparam, newvalue):
    status = CR.CPXXparamsetaddlong(env, ps, whichparam, newvalue)
    check_status(env, status)

def paramsetaddstr(env, ps, whichparam, newvalue):
    status = CR.CPXXparamsetaddstr(env, ps, whichparam, newvalue)
    check_status(env, status)

def paramsetapply(env, ps):
    status = CR.CPXXparamsetapply(env, ps)
    check_status(env, status)

def paramsetcopy(env, targetps, sourceps):
    status = CR.CPXXparamsetcopy(env, targetps, sourceps)
    check_status(env, status)

def paramsetcreate(env):
    status = CR.intPtr()
    ps = CR.CPXXparamsetcreate(env, status)
    check_status(env, status.value())
    return ps

def paramsetdel(env, ps, whichparam):
    status = CR.CPXXparamsetdel(env, ps, whichparam)
    check_status(env, status)

def paramsetfree(env, ps):
    ps_p = CR.CPXPARAMSETptrPtr()
    ps_p.assign(ps)
    status = CR.CPXXparamsetfree(env, ps_p)
    check_status(env, status)

def paramsetget(env, ps, whichparam, paramtype=None):
    if paramtype is None:
        paramtype = getparamtype(env, whichparam)
    switcher = {
        _const.CPX_PARAMTYPE_INT: paramsetgetint,
        _const.CPX_PARAMTYPE_DOUBLE: paramsetgetdbl,
        _const.CPX_PARAMTYPE_STRING: paramsetgetstr,
        _const.CPX_PARAMTYPE_LONG: paramsetgetlong
    }
    func = switcher[paramtype]
    return func(env, ps, whichparam)

def paramsetgetdbl(env, ps, whichparam):
    value = CR.doublePtr()
    status = CR.CPXXparamsetgetdbl(env, ps, whichparam, value)
    check_status(env, status)
    return value.value()

def paramsetgetint(env, ps, whichparam):
    value = CR.intPtr()
    status = CR.CPXXparamsetgetint(env, ps, whichparam, value)
    check_status(env, status)
    return value.value()

def paramsetgetlong(env, ps, whichparam):
    value = CR.cpxlongPtr()
    status = CR.CPXXparamsetgetlong(env, ps, whichparam, value)
    check_status(env, status)
    return value.value()

def paramsetgetstr(env, ps, whichparam):
    output = []
    status = CR.CPXXparamsetgetstr(env, ps, whichparam, output)
    check_status(env, status)
    return output[0]

def paramsetgetids(env, ps):
    cnt = paramsetgetnum(env, ps)
    if cnt == 0:
        return []
    inout_list = [cnt]
    status = CR.CPXXparamsetgetids(env, ps, inout_list)
    check_status(env, status)
    # We expect to get [whichparams]
    assert len(inout_list) == 1
    return inout_list[0]

def paramsetreadcopy(env, ps, filename):
    status = CR.CPXXparamsetreadcopy(env, ps, filename)
    check_status(env, status)

def paramsetwrite(env, ps, filename):
    status = CR.CPXXparamsetwrite(env, ps, filename)
    check_status(env, status)

def paramsetgetnum(env, ps):
    inout_list = [0]
    status = CR.CPXXparamsetgetids(env, ps, inout_list)
    if status != CR.CPXERR_NEGATIVE_SURPLUS:
        check_status(env, status)
    return inout_list[0]

########################################################################
# Runseeds
########################################################################

def runseeds(env, lp, cnt):
    with SigIntHandler():
        status = CR.CPXErunseeds(env, lp, cnt)
    check_status(env, status)

# Cplex


def createprob(env, probname):
    status = CR.intPtr()
    lp = CR.CPXXcreateprob(env, status, probname)
    check_status(env, status.value())
    return lp


def readcopyprob(env, lp, filename, filetype=""):
    if filetype == "":
        status = CR.CPXXreadcopyprob(env, lp, filename)
    else:
        status = CR.CPXXreadcopyprob(env, lp, filename, filetype)
    check_status(env, status)


def cloneprob(env, lp):
    status = CR.intPtr()
    lp = CR.CPXXcloneprob(env, lp, status)
    check_status(env, status.value())
    return lp


def freeprob(env, lp):
    lpp = CR.CPXLPptrPtr()
    lpp.assign(lp)
    status = CR.CPXXfreeprob(env, lpp)
    check_status(env, status)


def mipopt(env, lp):
    with SigIntHandler():
        status = CR.CPXXmipopt(env, lp)
    check_status(env, status)


def qpopt(env, lp):
    with SigIntHandler():
        status = CR.CPXXqpopt(env, lp)
    check_status(env, status)


def baropt(env, lp):
    with SigIntHandler():
        status = CR.CPXXbaropt(env, lp)
    check_status(env, status)


def hybbaropt(env, lp, method):
    with SigIntHandler():
        status = CR.CPXXhybbaropt(env, lp, method)
    check_status(env, status)


def hybnetopt(env, lp, method):
    with SigIntHandler():
        status = CR.CPXXhybnetopt(env, lp, method)
    check_status(env, status)


def lpopt(env, lp):
    with SigIntHandler():
        status = CR.CPXXlpopt(env, lp)
    check_status(env, status)


def primopt(env, lp):
    status = CR.CPXXprimopt(env, lp)
    check_status(env, status)


def dualopt(env, lp):
    status = CR.CPXXdualopt(env, lp)
    check_status(env, status)


def siftopt(env, lp):
    status = CR.CPXXsiftopt(env, lp)
    check_status(env, status)


def feasoptext(env, lp, grppref, grpbeg, grpind, grptype):
    grpcnt = len(grppref)
    concnt = len(grpind)
    with SigIntHandler(), \
            LAU.double_c_array(grppref) as c_grppref, \
            LAU.long_c_array(grpbeg) as c_grpbeg, \
            LAU.int_c_array(grpind) as c_grpind:
        status = CR.CPXXfeasoptext(env, lp, grpcnt, concnt,
                                   c_grppref, c_grpbeg,
                                   c_grpind, grptype)
    check_status(env, status)


def delnames(env, lp):
    status = CR.CPXXdelnames(env, lp)
    check_status(env, status)


def writeprob(env, lp, filename, filetype=""):
    if filetype == "":
        status = CR.CPXXwriteprob(env, lp, filename)
    else:
        status = CR.CPXXwriteprob(env, lp, filename, filetype)
    check_status(env, status)


def writeprobdev(env, lp, stream, filename, filetype):
    arg_list = [stream, filename, filetype]
    status = CR.CPXEwriteprobdev(env, lp, arg_list)
    check_status(env, status)


def embwrite(env, lp, filename):
    status = CR.CPXXembwrite(env, lp, filename)
    check_status(env, status)


def dperwrite(env, lp, filename, epsilon):
    status = CR.CPXXdperwrite(env, lp, filename, epsilon)
    check_status(env, status)


def pperwrite(env, lp, filename, epsilon):
    status = CR.CPXXpperwrite(env, lp, filename, epsilon)
    check_status(env, status)


def preslvwrite(env, lp, filename):
    objoff = CR.doublePtr()
    status = CR.CPXXpreslvwrite(env, lp, filename, objoff)
    check_status(env, status)
    return objoff.value()


def dualwrite(env, lp, filename):
    objshift = CR.doublePtr()
    status = CR.CPXXdualwrite(env, lp, filename, objshift)
    check_status(env, status)
    return objshift.value()


def chgprobtype(env, lp, probtype):
    status = CR.CPXXchgprobtype(env, lp, probtype)
    check_status(env, status)


def chgprobtypesolnpool(env, lp, probtype, soln):
    status = CR.CPXXchgprobtypesolnpool(env, lp, probtype, soln)
    check_status(env, status)


def getprobtype(env, lp):
    return CR.CPXXgetprobtype(env, lp)


def chgprobname(env, lp, probname):
    status = CR.CPXXchgprobname(env, lp, probname)
    check_status(env, status)


def getprobname(env, lp):
    namefn = CR.CPXXgetprobname
    return _getnamesingle(env, lp, namefn)


def getnumcols(env, lp):
    return CR.CPXXgetnumcols(env, lp)


def getnumint(env, lp):
    return CR.CPXXgetnumint(env, lp)


def getnumbin(env, lp):
    return CR.CPXXgetnumbin(env, lp)


def getnumsemicont(env, lp):
    return CR.CPXXgetnumsemicont(env, lp)


def getnumsemiint(env, lp):
    return CR.CPXXgetnumsemiint(env, lp)


def getnumrows(env, lp):
    return CR.CPXXgetnumrows(env, lp)


def populate(env, lp):
    with SigIntHandler():
        status = CR.CPXXpopulate(env, lp)
    check_status(env, status)


def _getnumusercuts(env, lp):
    return CR.CPXXgetnumusercuts(env, lp)


def _getnumlazyconstraints(env, lp):
    return CR.CPXXgetnumlazyconstraints(env, lp)


def getnumqconstrs(env, lp):
    return CR.CPXXgetnumqconstrs(env, lp)


def getnumindconstrs(env, lp):
    return CR.CPXXgetnumindconstrs(env, lp)


def getnumsos(env, lp):
    return CR.CPXXgetnumsos(env, lp)


def cleanup(env, lp, eps):
    status = CR.CPXXcleanup(env, lp, eps)
    check_status(env, status)


def basicpresolve(env, lp):
    numcols = CR.CPXXgetnumcols(env, lp)
    numrows = CR.CPXXgetnumrows(env, lp)
    redlb = _safeDoubleArray(numcols)
    redub = _safeDoubleArray(numcols)
    rstat = _safeIntArray(numrows)
    status = CR.CPXXbasicpresolve(env, lp, redlb, redub, rstat)
    check_status(env, status)
    return (LAU.array_to_list(redlb, numcols),
            LAU.array_to_list(redub, numcols),
            LAU.array_to_list(rstat, numrows))


def pivotin(env, lp, rlist):
    status = CR.CPXXpivotin(env, lp,
                            LAU.int_list_to_array(rlist),
                            len(rlist))
    check_status(env, status)


def pivotout(env, lp, clist):
    status = CR.CPXXpivotout(env, lp,
                             LAU.int_list_to_array(clist),
                             len(clist))
    check_status(env, status)


def pivot(env, lp, jenter, jleave, leavestat):
    status = CR.CPXXpivot(env, lp, jenter, jleave, leavestat)
    check_status(env, status)


def strongbranch(env, lp, goodlist, itlim):
    goodlen = len(goodlist)
    downpen = _safeDoubleArray(goodlen)
    uppen = _safeDoubleArray(goodlen)
    with SigIntHandler():
        status = CR.CPXXstrongbranch(
            env, lp, LAU.int_list_to_array(goodlist), goodlen,
            downpen, uppen, itlim)
    check_status(env, status)
    return (LAU.array_to_list(downpen, goodlen),
            LAU.array_to_list(uppen, goodlen))


def completelp(env, lp):
    status = CR.CPXXcompletelp(env, lp)
    check_status(env, status)

# Variables
@contextmanager
def fast_getrows(env, lp):
    env_lp_ptr = pack_env_lp_ptr(env, lp)
    rows = CR.CPXX_fast_get_rows(env_lp_ptr)
    try:
        yield rows
    finally:
        CR.CPXX_free_rows(rows)

@contextmanager
def fast_getcolname(env, lp, begin, end):
    env_lp_ptr = pack_env_lp_ptr(env, lp)
    names = CR.CPXX_fast_getcolname(env_lp_ptr, begin, end)
    try:
        yield names
    finally:
        CR.CPXX_free_getname(names)

@contextmanager
def fast_getrowname(env, lp, begin, end):
    env_lp_ptr = pack_env_lp_ptr(env, lp)
    names = CR.CPXX_fast_getrowname(env_lp_ptr, begin, end)
    try:
        yield names
    finally:
        CR.CPXX_free_getname(names)

@contextmanager
def fast_getsosname(env, lp, begin, end):
    env_lp_ptr = pack_env_lp_ptr(env, lp)
    names = CR.CPXX_fast_getsosname(env_lp_ptr, begin, end)
    try:
        yield names
    finally:
        CR.CPXX_free_getname(names)

@contextmanager
def fast_getmipstartname(env, lp, begin, end):
    env_lp_ptr = pack_env_lp_ptr(env, lp)
    names = CR.CPXX_fast_getmipstartname(env_lp_ptr, begin, end)
    try:
        yield names
    finally:
        CR.CPXX_free_getname(names)

@contextmanager
def fast_getobj(env, lp, begin, end):
    env_lp_ptr = pack_env_lp_ptr(env, lp)
    coefs = CR.CPXX_fast_getobj(env_lp_ptr, begin, end)
    try:
        yield coefs
    finally:
        CR.CPXX_free_getobj(coefs)
@contextmanager
def fast_multiobjgetobj(env, lp, objidx, begin, end):
    env_lp_ptr = pack_env_lp_ptr(env, lp)
    coefs = CR.CPXX_fast_multiobjgetobj(env_lp_ptr, objidx, begin, end)
    try:
        yield coefs
    finally:
        CR.CPXX_free_getobj(coefs)

def fast_multiobjgetoffset(env, lp, objidx):
    env_lp_ptr = pack_env_lp_ptr(env, lp)
    return CR.CPXX_fast_multiobjgetoffset(env_lp_ptr, objidx)
def fast_multiobjgetweight(env, lp, objidx):
    env_lp_ptr = pack_env_lp_ptr(env, lp)
    return CR.CPXX_fast_multiobjgetweight(env_lp_ptr, objidx)
def fast_multiobjgetpriority(env, lp, objidx):
    env_lp_ptr = pack_env_lp_ptr(env, lp)
    return CR.CPXX_fast_multiobjgetpriority(env_lp_ptr, objidx)
def fast_multiobjgetabstol(env, lp, objidx):
    env_lp_ptr = pack_env_lp_ptr(env, lp)
    return CR.CPXX_fast_multiobjgetabstol(env_lp_ptr, objidx)
def fast_multiobjgetreltol(env, lp, objidx):
    env_lp_ptr = pack_env_lp_ptr(env, lp)
    return CR.CPXX_fast_multiobjgetreltol(env_lp_ptr, objidx)


def fast_newcols(env, lp, nb, lb, ub, xctype):
    env_lp_ptr = pack_env_lp_ptr(env, lp)
    CR.CPXX_fast_newcols(env_lp_ptr, nb, lb, ub, xctype)

def has_name(env, lp, start, end):
    env_lp_ptr = pack_env_lp_ptr(env, lp)
    return True if CR.CPXX_has_name(env_lp_ptr, start, end) != 0 else False
def has_non_default_lb(env, lp, start, end):
    env_lp_ptr = pack_env_lp_ptr(env, lp)
    return True if CR.CPXX_has_non_default_lb(env_lp_ptr, start, end) != 0 else False
def has_non_default_ub(env, lp, start, end):
    env_lp_ptr = pack_env_lp_ptr(env, lp)
    return True if CR.CPXX_has_non_default_ub(env_lp_ptr, start, end) != 0 else False


def newcols(env, lp, obj, lb, ub, xctype, colname):
    ccnt = max(len(obj), len(lb), len(ub), len(xctype), len(colname))
    with LAU.double_c_array(obj) as c_obj, \
            LAU.double_c_array(lb) as c_lb, \
            LAU.double_c_array(ub) as c_ub:
        status = CR.CPXXnewcols(
            env, lp, ccnt, c_obj, c_lb, c_ub,
            xctype, colname)
    check_status(env, status)


def addcols(env, lp, ccnt, nzcnt, obj, cmat, lb, ub, colname):
    with LAU.double_c_array(obj) as c_obj, \
            LAU.double_c_array(lb) as c_lb, \
            LAU.double_c_array(ub) as c_ub:
        status = CR.CPXXaddcols(
            env, lp, ccnt, nzcnt,
            c_obj, cmat, c_lb, c_ub, colname)
    check_status(env, status)


def delcols(env, lp, begin, end):
    delfn = CR.CPXXdelcols
    _delbyrange(delfn, env, lp, begin, end)


def chgbds(env, lp, indices, lu, bd):
    with LAU.int_c_array(indices) as c_ind, \
            LAU.double_c_array(bd) as c_bd:
        status = CR.CPXXchgbds(env, lp, len(indices),
                               c_ind, lu, c_bd)
    check_status(env, status)


def chgcolname(env, lp, indices, newnames):
    with LAU.int_c_array(indices) as c_indices:
        status = CR.CPXXchgcolname(env, lp, len(indices),
                                   c_indices, newnames)
    check_status(env, status)


def chgctype(env, lp, indices, xctype):
    with LAU.int_c_array(indices) as c_indices:
        status = CR.CPXXchgctype(env, lp, len(indices),
                                 c_indices, xctype)
    check_status(env, status)


def getcolindex(env, lp, colname):
    index = CR.intPtr()
    status = CR.CPXXgetcolindex(env, lp, colname, index)
    check_status(env, status)
    return index.value()


def getcolname(env, lp, begin, end):
    namefn = CR.CPXXgetcolname
    return _getnamebyrange(env, lp, begin, end, namefn)


def getctype(env, lp, begin, end):
    inout_list = [begin, end]
    status = CR.CPXXgetctype(env, lp, inout_list)
    check_status(env, status)
    # We expect to get [sense]
    assert len(inout_list) == 1
    return inout_list[0]


def getlb(env, lp, begin, end):
    lblen = _rangelen(begin, end)
    lb = _safeDoubleArray(lblen)
    status = CR.CPXXgetlb(env, lp, lb, begin, end)
    check_status(env, status)
    return LAU.array_to_list(lb, lblen)


def getub(env, lp, begin, end):
    ublen = _rangelen(begin, end)
    ub = _safeDoubleArray(ublen)
    status = CR.CPXXgetub(env, lp, ub, begin, end)
    check_status(env, status)
    return LAU.array_to_list(ub, ublen)


def getcols(env, lp, begin, end):
    inout_list = [0, begin, end]
    status = CR.CPXXgetcols(env, lp, inout_list)
    if status != CR.CPXERR_NEGATIVE_SURPLUS:
        check_status(env, status)
    if inout_list == [0]:
        return ([0] * _rangelen(begin, end), [], [])
    inout_list.extend([begin, end])
    status = CR.CPXXgetcols(env, lp, inout_list)
    check_status(env, status)
    return tuple(inout_list)


def copyprotected(env, lp, indices):
    status = CR.CPXXcopyprotected(env, lp, len(indices),
                                  LAU.int_list_to_array(indices))
    check_status(env, status)


def getprotected(env, lp):
    count = CR.intPtr()
    surplus = CR.intPtr()
    indices = LAU.int_list_to_array([])
    pspace = 0
    status = CR.CPXXgetprotected(env, lp, count, indices, pspace, surplus)
    if status != CR.CPXERR_NEGATIVE_SURPLUS:
        check_status(env, status)
    if surplus.value() == 0:
        return []
    pspace = -surplus.value()
    indices = _safeIntArray(pspace)
    status = CR.CPXXgetprotected(env, lp, count, indices, pspace, surplus)
    check_status(env, status)
    return LAU.array_to_list(indices, pspace)


def tightenbds(env, lp, indices, lu, bd):
    status = CR.CPXXtightenbds(env, lp, len(indices),
                               LAU.int_list_to_array(indices),
                               lu, LAU.double_list_to_array(bd))
    check_status(env, status)

# Linear Constraints


def newrows(env, lp, rhs, sense, rngval, rowname):
    rcnt = max(len(rhs), len(sense), len(rngval), len(rowname))
    with LAU.double_c_array(rhs) as c_rhs, \
            LAU.double_c_array(rngval) as c_rng:
        status = CR.CPXXnewrows(env, lp, rcnt, c_rhs, sense,
                                c_rng, rowname)
    check_status(env, status)


def addrows(env, lp, ccnt, rcnt, nzcnt, rhs, sense, rmat, colname, rowname):
    with LAU.double_c_array(rhs) as c_rhs:
        status = CR.CPXXaddrows(
            env, lp, ccnt, rcnt, nzcnt, c_rhs,
            sense, rmat, colname, rowname)
    check_status(env, status)


def delrows(env, lp, begin, end):
    delfn = CR.CPXXdelrows
    _delbyrange(delfn, env, lp, begin, end)


def chgrowname(env, lp, indices, newnames):
    with LAU.int_c_array(indices) as c_indices:
        status = CR.CPXXchgrowname(env, lp, len(indices), c_indices,
                                   newnames)
    check_status(env, status)


def chgcoeflist(env, lp, rowlist, collist, vallist):
    with LAU.int_c_array(rowlist) as c_rowlist, \
            LAU.int_c_array(collist) as c_collist, \
            LAU.double_c_array(vallist) as c_vallist:
        status = CR.CPXXchgcoeflist(env, lp, len(rowlist),
                                    c_rowlist, c_collist, c_vallist)
    check_status(env, status)


def chgrhs(env, lp, indices, values):
    with LAU.int_c_array(indices) as c_ind, \
            LAU.double_c_array(values) as c_val:
        status = CR.CPXXchgrhs(env, lp, len(indices), c_ind, c_val)
    check_status(env, status)


def chgrngval(env, lp, indices, values):
    with LAU.int_c_array(indices) as c_ind, \
            LAU.double_c_array(values) as c_val:
        status = CR.CPXXchgrngval(env, lp, len(indices), c_ind, c_val)
    check_status(env, status)


def chgsense(env, lp, indices, senses):
    with LAU.int_c_array(indices) as c_indices:
        status = CR.CPXXchgsense(env, lp, len(indices), c_indices,
                                 senses)
    check_status(env, status)


def getrhs(env, lp, begin, end):
    rhslen = _rangelen(begin, end)
    rhs = _safeDoubleArray(rhslen)
    status = CR.CPXXgetrhs(env, lp, rhs, begin, end)
    check_status(env, status)
    return LAU.array_to_list(rhs, rhslen)


def getsense(env, lp, begin, end):
    inout_list = [begin, end]
    status = CR.CPXXgetsense(env, lp, inout_list)
    check_status(env, status)
    # We expect to get [sense]
    assert len(inout_list) == 1
    return inout_list[0]


def getrngval(env, lp, begin, end):
    rngvallen = _rangelen(begin, end)
    rngval = _safeDoubleArray(rngvallen)
    status = CR.CPXXgetrngval(env, lp, rngval, begin, end)
    check_status(env, status)
    return LAU.array_to_list(rngval, rngvallen)


def getrowname(env, lp, begin, end):
    namefn = CR.CPXXgetrowname
    return _getnamebyrange(env, lp, begin, end, namefn)


def getcoef(env, lp, i, j):
    coef = CR.doublePtr()
    status = CR.CPXXgetcoef(env, lp, i, j, coef)
    check_status(env, status)
    return coef.value()


def getrowindex(env, lp, rowname):
    index = CR.intPtr()
    status = CR.CPXXgetrowindex(env, lp, rowname, index)
    check_status(env, status)
    return index.value()


def getrows(env, lp, begin, end):
    inout_list = [0, begin, end]
    status = CR.CPXXgetrows(env, lp, inout_list)
    if status != CR.CPXERR_NEGATIVE_SURPLUS:
        check_status(env, status)
    if inout_list == [0]:
        return ([0] * _rangelen(begin, end), [], [])
    inout_list.extend([begin, end])
    status = CR.CPXXgetrows(env, lp, inout_list)
    check_status(env, status)
    return tuple(inout_list)


def getnumnz(env, lp):
    return CR.CPXXgetnumnz(env, lp)


def addlazyconstraints(env, lp, rhs, sense, lin_expr, names):
    env_lp_ptr = pack_env_lp_ptr(env, lp)
    with chbmatrix(lin_expr, env_lp_ptr, 0) as (rmat, nnz), \
            LAU.double_c_array(rhs) as c_rhs:
        rmatbeg, rmatind, rmatval = rmat
        status = CR.CPXXaddlazyconstraints(
            env, lp, len(rhs), nnz,
            c_rhs, sense,
            rmatbeg, rmatind, rmatval,
            names)
    check_status(env, status)


def addusercuts(env, lp, rhs, sense, lin_expr, names):
    env_lp_ptr = pack_env_lp_ptr(env, lp)
    with chbmatrix(lin_expr, env_lp_ptr, 0) as (rmat, nnz), \
            LAU.double_c_array(rhs) as c_rhs:
        rmatbeg, rmatind, rmatval = rmat
        status = CR.CPXXaddusercuts(
            env, lp, len(rhs), nnz,
            c_rhs, sense,
            rmatbeg, rmatind, rmatval,
            names)
    check_status(env, status)


def freelazyconstraints(env, lp):
    status = CR.CPXXfreelazyconstraints(env, lp)
    check_status(env, status)


def freeusercuts(env, lp):
    status = CR.CPXXfreeusercuts(env, lp)
    check_status(env, status)


# CPXLIBAPI int CPXPUBLIC
# CPXXcopylpwnames (CPXCENVptr env,
#                   CPXLPptr lp,
#                   CPXDIM numcols,
#                   CPXDIM numrows,
#                   int objsense,
#                   const double *objective,
#                   const double *rhs,
#                   const char *sense,
#                   CPXNNZ int *matbeg,
#                   CPXDIM int *matcnt,
#                   CPXDIM int *matind,
#                   const double *matval,
#                   const double *lb,
#                   const double *ub,
#                   const double *rngval,
#                   char const *const *colname,
#                   char const *const *rowname);
def copylpwnames(env, lp, numcols, numrows, objsense, obj, rhs, sense,
                 matbeg, matcnt, matind, matval, lb, ub, rngval, colname,
                 rowname):
    with LAU.long_c_array(matbeg) as c_matbeg, \
         LAU.int_c_array(matcnt) as c_matcnt, \
         LAU.int_c_array(matind) as c_matind, \
         LAU.double_c_array(matval) as c_matval, \
         LAU.double_c_array(obj) as c_obj, \
         LAU.double_c_array(rhs) as c_rhs, \
         LAU.double_c_array(lb) as c_lb, \
         LAU.double_c_array(ub) as c_ub, \
         LAU.double_c_array(rngval) as c_rngval:  # noqa: E126
        status = CR.CPXXcopylpwnames(env, lp, numcols, numrows, objsense,
                                     c_obj, c_rhs, sense,
                                     c_matbeg, c_matcnt, c_matind, c_matval,
                                     c_lb, c_ub, c_rngval,
                                     colname, rowname)
        check_status(env, status)


########################################################################
# SOS API
########################################################################


def addsos(env, lp, sostype, sosbeg, sosind, soswt, sosnames):
    with LAU.long_c_array(sosbeg) as c_sosbeg, \
            LAU.int_c_array(sosind) as c_sosind, \
            LAU.double_c_array(soswt) as c_soswt:
        status = CR.CPXXaddsos(env, lp, len(sosbeg), len(sosind), sostype,
                               c_sosbeg, c_sosind, c_soswt,
                               sosnames)
    check_status(env, status)


def delsos(env, lp, begin, end):
    delfn = CR.CPXXdelsos
    _delbyrange(delfn, env, lp, begin, end)


def getsos_info(env, lp, begin, end):
    inout_list = [0, begin, end]
    status = CR.CPXXgetsos(env, lp, inout_list)
    if status != CR.CPXERR_NEGATIVE_SURPLUS:
        check_status(env, status)
    # We expect to get [sostype, surplus]
    assert len(inout_list) == 2
    return tuple(inout_list)


def getsos(env, lp, begin, end):
    numsos = _rangelen(begin, end)
    _, surplus = getsos_info(env, lp, begin, end)
    if surplus == 0:
        return ([0] * numsos, [], [])
    inout_list = [surplus, begin, end]
    status = CR.CPXXgetsos(env, lp, inout_list)
    check_status(env, status)
    # We expect to get [sosbeg, sosind, soswt]
    assert len(inout_list) == 3
    return tuple(inout_list)


def getsosindex(env, lp, name):
    indexfn = CR.CPXXgetsosindex
    return _getindex(env, lp, name, indexfn)


def getsosname(env, lp, begin, end):
    namefn = CR.CPXXgetsosname
    return _getnamebyrange(env, lp, begin, end, namefn)

########################################################################
# Indicator Constraint API
########################################################################


def addindconstr(env, lp, indcnt, indvar, complemented, rhs, sense, linmat,
                 indtype, name, nzcnt):
    with LAU.int_c_array(indtype) as c_indtype, \
            LAU.int_c_array(indvar) as c_indvar, \
            LAU.int_c_array(complemented) as c_complemented, \
            LAU.double_c_array(rhs) as c_rhs:
        status = CR.CPXXaddindconstraints(
            env, lp, indcnt, c_indtype, c_indvar,
            c_complemented, nzcnt, c_rhs,
            sense, linmat,
            name)
    check_status(env, status)


def getindconstr(env, lp, begin, end):
    _, _, _, _, _, surplus = getindconstr_constant(env, lp, begin, end)
    if surplus == 0:
        return ([0] * _rangelen(begin, end), [], [])
    # inout_list contains the linspace, begin, and end args to
    # CPXXgetindconstraints.
    inout_list = [surplus, begin, end]
    status = CR.CPXXgetindconstraints(env, lp, inout_list)
    check_status(env, status)
    # We expect to get [linbeg, linind, linval]
    assert len(inout_list) == 3
    return tuple(inout_list)


def getindconstr_constant(env, lp, begin, end):
    # inout_list contains the linspace, begin, and end args to
    # CPXXgetindconstraints.
    inout_list = [0, begin, end]
    status = CR.CPXXgetindconstraints(env, lp, inout_list)
    if status != CR.CPXERR_NEGATIVE_SURPLUS:
        check_status(env, status)
    # We expect to get:
    # [type, indvar, complemented, rhs, sense, surplus]
    assert len(inout_list) == 6
    return tuple(inout_list)


def getindconstrindex(env, lp, name):
    indexfn = CR.CPXXgetindconstrindex
    return _getindex(env, lp, name, indexfn)


def delindconstrs(env, lp, begin, end):
    delfn = CR.CPXXdelindconstrs
    _delbyrange(delfn, env, lp, begin, end)


def getindconstrslack(env, lp, begin, end):
    slacklen = _rangelen(begin, end)
    slacks = _safeDoubleArray(slacklen)
    status = CR.CPXXgetindconstrslack(env, lp, slacks, begin, end)
    check_status(env, status)
    return LAU.array_to_list(slacks, slacklen)


def getindconstrname(env, lp, which):
    namefn = CR.CPXXgetindconstrname
    return _getname(env, lp, which, namefn, index_first=False)

########################################################################
# Quadratic Constraints
########################################################################


def addqconstr(env, lp, rhs, sense, linind, linval, quadrow, quadcol,
               quadval, name):
    with LAU.int_c_array(linind) as c_linind, \
            LAU.double_c_array(linval) as c_linval, \
            LAU.int_c_array(quadrow) as c_quadrow, \
            LAU.int_c_array(quadcol) as c_quadcol, \
            LAU.double_c_array(quadval) as c_quadval:
        status = CR.CPXXaddqconstr(env, lp, len(linind), len(quadrow),
                                   rhs, sense,
                                   c_linind, c_linval,
                                   c_quadrow, c_quadcol, c_quadval,
                                   name)
    check_status(env, status)


def getqconstr_info(env, lp, which):
    inout_list = [0, 0, which]
    status = CR.CPXXgetqconstr(env, lp, inout_list)
    if status != CR.CPXERR_NEGATIVE_SURPLUS:
        check_status(env, status)
    # We expect to get [rhs, sense, linsurplus, quadsurplus]
    assert len(inout_list) == 4
    assert len(inout_list[1]) == 1  # sense string should be one char
    return tuple(inout_list)


def getqconstr_lin(env, lp, which):
    _, _, linsurplus, _ = getqconstr_info(env, lp, which)
    if linsurplus == 0:
        return ([], [])
    inout_list = [linsurplus, 0, which]
    status = CR.CPXXgetqconstr(env, lp, inout_list)
    if status != CR.CPXERR_NEGATIVE_SURPLUS:
        check_status(env, status)
    # We expect to get [linind, linval, quadrow, quadcol, quadval]
    assert len(inout_list) == 5
    return tuple(inout_list[:2])  # slice off the quad info


def getqconstr_quad(env, lp, which):
    _, _, _, quadsurplus = getqconstr_info(env, lp, which)
    if quadsurplus == 0:
        return ([], [], [])
    inout_list = [0, quadsurplus, which]
    status = CR.CPXXgetqconstr(env, lp, inout_list)
    if status != CR.CPXERR_NEGATIVE_SURPLUS:
        check_status(env, status)
    # We expect to get [linind, linval, quadrow, quadcol, quadval]
    assert len(inout_list) == 5
    return tuple(inout_list[2:])  # slice off the lin info


def delqconstrs(env, lp, begin, end):
    delfn = CR.CPXXdelqconstrs
    _delbyrange(delfn, env, lp, begin, end)


def getqconstrindex(env, lp, name):
    indexfn = CR.CPXXgetqconstrindex
    return _getindex(env, lp, name, indexfn)


def getqconstrslack(env, lp, begin, end):
    slacklen = _rangelen(begin, end)
    slacks = _safeDoubleArray(slacklen)
    status = CR.CPXXgetqconstrslack(env, lp, slacks, begin, end)
    check_status(env, status)
    return LAU.array_to_list(slacks, slacklen)


def getqconstrname(env, lp, which):
    namefn = CR.CPXXgetqconstrname
    return _getname(env, lp, which, namefn, index_first=False)

########################################################################
# Generic helper methods
########################################################################


def _delbyrange(delfn, env, lp, begin, end=None):
    if end is None:
        end = begin
    status = delfn(env, lp, begin, end)
    check_status(env, status)


def _getindex(env, lp, name, indexfn):
    idx = CR.intPtr()
    status = indexfn(env, lp, name, idx)
    check_status(env, status)
    return idx.value()


def _getname(env, lp, idx, namefn, index_first=True):
    # Some name functions have the index argument first and some have it
    # last.  Thus, we do this little dance, so things are called in the
    # right way depending on index_first.
    def _namefn(env, lp, idx, inoutlist):
        if index_first:
            return namefn(env, lp, idx, inoutlist)
        return namefn(env, lp, inoutlist, idx)
    inoutlist = [0]
    # NB: inoutlist will be modified as a side effect
    status = _namefn(env, lp, idx, inoutlist)
    if status != CR.CPXERR_NEGATIVE_SURPLUS:
        check_status(env, status)
    if inoutlist == [0]:
        return None
    status = _namefn(env, lp, idx, inoutlist)
    check_status(env, status)
    return inoutlist[0]


def _getnamebyrange(env, lp, begin, end, namefn):
    inout_list = [0, begin, end]
    status = namefn(env, lp, inout_list)
    if status != CR.CPXERR_NEGATIVE_SURPLUS:
        check_status(env, status)
    if inout_list == [0]:
        return [None] * _rangelen(begin, end)
    inout_list.extend([begin, end])
    status = namefn(env, lp, inout_list)
    check_status(env, status)
    return inout_list


def _getnamesingle(env, lp, namefn):
    inoutlist = [0]
    status = namefn(env, lp, inoutlist)
    if status != CR.CPXERR_NEGATIVE_SURPLUS:
        check_status(env, status)
    if inoutlist == [0]:
        return None
    status = namefn(env, lp, inoutlist)
    check_status(env, status)
    return inoutlist[0]

########################################################################
# Annotation API
########################################################################


def _newanno(env, lp, name, defval, newfn):
    status = newfn(env, lp, name, defval)
    check_status(env, status)


def newlonganno(env, lp, name, defval):
    newfn = CR.CPXXnewlongannotation
    _newanno(env, lp, name, defval, newfn)


def newdblanno(env, lp, name, defval):
    newfn = CR.CPXXnewdblannotation
    _newanno(env, lp, name, defval, newfn)


def dellonganno(env, lp, begin, end):
    delfn = CR.CPXXdellongannotations
    _delbyrange(delfn, env, lp, begin, end)


def deldblanno(env, lp, begin, end):
    delfn = CR.CPXXdeldblannotations
    _delbyrange(delfn, env, lp, begin, end)


def getlongannoindex(env, lp, name):
    indexfn = CR.CPXXgetlongannotationindex
    return _getindex(env, lp, name, indexfn)


def getdblannoindex(env, lp, name):
    indexfn = CR.CPXXgetdblannotationindex
    return _getindex(env, lp, name, indexfn)


def getlongannoname(env, lp, idx):
    namefn = CR.CPXXgetlongannotationname
    return _getname(env, lp, idx, namefn)


def getdblannoname(env, lp, idx):
    namefn = CR.CPXXgetdblannotationname
    return _getname(env, lp, idx, namefn)


def getnumlonganno(env, lp):
    return CR.CPXXgetnumlongannotations(env, lp)


def getnumdblanno(env, lp):
    return CR.CPXXgetnumdblannotations(env, lp)


def getlongannodefval(env, lp, idx):
    defval = CR.cpxlongPtr()
    status = CR.CPXXgetlongannotationdefval(env, lp, idx, defval)
    check_status(env, status)
    return int(defval.value())


def getdblannodefval(env, lp, idx):
    defval = CR.doublePtr()
    status = CR.CPXXgetdblannotationdefval(env, lp, idx, defval)
    check_status(env, status)
    return defval.value()


def setlonganno(env, lp, idx, objtype, ind, val):
    assert len(ind) == len(val)
    cnt = len(ind)
    status = CR.CPXXsetlongannotations(env, lp, idx, objtype, cnt,
                                       LAU.int_list_to_array(ind),
                                       LAU.long_list_to_array(val))
    check_status(env, status)


def setdblanno(env, lp, idx, objtype, ind, val):
    assert len(ind) == len(val)
    cnt = len(ind)
    status = CR.CPXXsetdblannotations(env, lp, idx, objtype, cnt,
                                      LAU.int_list_to_array(ind),
                                      LAU.double_list_to_array(val))
    check_status(env, status)


def getlonganno(env, lp, idx, objtype, begin, end):
    annolen = _rangelen(begin, end)
    val = _safeLongArray(annolen)
    status = CR.CPXXgetlongannotations(env, lp, idx, objtype, val,
                                       begin, end)
    check_status(env, status)
    return [int(i) for i in LAU.array_to_list(val, annolen)]


def getdblanno(env, lp, idx, objtype, begin, end):
    annolen = _rangelen(begin, end)
    val = _safeDoubleArray(annolen)
    status = CR.CPXXgetdblannotations(env, lp, idx, objtype, val,
                                      begin, end)
    check_status(env, status)
    return LAU.array_to_list(val, annolen)


def readcopyanno(env, lp, filename):
    status = CR.CPXXreadcopyannotations(env, lp, filename)
    check_status(env, status)


def writeanno(env, lp, filename):
    status = CR.CPXXwriteannotations(env, lp, filename)
    check_status(env, status)


def writebendersanno(env, lp, filename):
    status = CR.CPXXwritebendersannotation(env, lp, filename)
    check_status(env, status)

########################################################################
# PWL API
########################################################################


def addpwl(env, lp, vary, varx, preslope, postslope, nbreaks,
           breakx, breaky, name):
    assert len(breakx) == nbreaks
    assert len(breaky) == nbreaks
    with LAU.double_c_array(breakx) as c_breakx, \
            LAU.double_c_array(breaky) as c_breaky:
        status = CR.CPXXaddpwl(env, lp, vary, varx, preslope, postslope,
                               nbreaks, c_breakx, c_breaky, name)
    check_status(env, status)


def delpwl(env, lp, begin, end):
    delfn = CR.CPXXdelpwl
    _delbyrange(delfn, env, lp, begin, end)


def getnumpwl(env, lp):
    return CR.CPXXgetnumpwl(env, lp)


def getpwl(env, lp, idx):
    # Initially, the inout_list contains the pwlindex and breakspace args
    # to CPXXgetpwl.  We use zero (0) for breakspace to query the
    # surplus.
    inout_list = [idx, 0]
    status = CR.CPXXgetpwl(env, lp, inout_list)
    if status != CR.CPXERR_NEGATIVE_SURPLUS:
        check_status(env, status)
    # We expect to get [vary, varx, preslope, postslope, surplus]
    assert len(inout_list) == 5
    vary, varx, preslope, postslope, surplus = inout_list
    # FIXME: Should we assert surplus is > 0?
    inout_list = [idx, surplus]
    status = CR.CPXXgetpwl(env, lp, inout_list)
    check_status(env, status)
    # We expect to get [breakx, breaky]
    assert len(inout_list) == 2
    breakx, breaky = inout_list
    return [vary, varx, preslope, postslope, breakx, breaky]


def getpwlindex(env, lp, name):
    indexfn = CR.CPXXgetpwlindex
    return _getindex(env, lp, name, indexfn)


def getpwlname(env, lp, idx):
    namefn = CR.CPXXgetpwlname
    return _getname(env, lp, idx, namefn, index_first=False)

########################################################################
# Objective API
########################################################################


def copyobjname(env, lp, objname):
    status = CR.CPXXcopyobjname(env, lp, objname)
    check_status(env, status)


def chgobj(env, lp, indices, values):
    with LAU.int_c_array(indices) as c_ind, \
            LAU.double_c_array(values) as c_val:
        status = CR.CPXXchgobj(env, lp, len(indices), c_ind, c_val)
    check_status(env, status)


def chgobjsen(env, lp, maxormin):
    status = CR.CPXXchgobjsen(env, lp, maxormin)
    check_status(env, status)


def getobjsen(env, lp):
    return CR.CPXXgetobjsen(env, lp)


def getobjoffset(env, lp):
    objoffset = CR.doublePtr()
    status = CR.CPXXgetobjoffset(env, lp, objoffset)
    check_status(env, status)
    return objoffset.value()


def chgobjoffset(env, lp, offset):
    status = CR.CPXXchgobjoffset(env, lp, offset)
    check_status(env, status)


def getobj(env, lp, begin, end):
    objlen = _rangelen(begin, end)
    obj = _safeDoubleArray(objlen)
    status = CR.CPXXgetobj(env, lp, obj, begin, end)
    check_status(env, status)
    return LAU.array_to_list(obj, objlen)


def getobjname(env, lp):
    namefn = CR.CPXXgetobjname
    return _getnamesingle(env, lp, namefn)


def copyquad(env, lp, qmatbeg, qmatind, qmatval):
    nqmatbeg = len(qmatbeg)
    if nqmatbeg > 0:
        qmatcnt = [qmatbeg[i + 1] - qmatbeg[i]
                   for i in range(nqmatbeg - 1)]
        qmatcnt.append(len(qmatind) - qmatbeg[-1])
    else:
        qmatcnt = []
    with LAU.long_c_array(qmatbeg) as c_qmatbeg, \
            LAU.int_c_array(qmatcnt) as c_qmatcnt, \
            LAU.int_c_array(qmatind) as c_qmatind, \
            LAU.double_c_array(qmatval) as c_qmatval:
        status = CR.CPXXcopyquad(env, lp, c_qmatbeg, c_qmatcnt,
                                 c_qmatind, c_qmatval)
    check_status(env, status)


def copyqpsep(env, lp, qsepvec):
    with LAU.double_c_array(qsepvec) as c_qsepvec:
        status = CR.CPXXcopyqpsep(env, lp, c_qsepvec)
    check_status(env, status)


def chgqpcoef(env, lp, row, col, value):
    status = CR.CPXXchgqpcoef(env, lp, row, col, value)
    check_status(env, status)


def getquad(env, lp, begin, end):
    nzcnt = CR.cpxlongPtr()
    ncols = _rangelen(begin, end)
    qmatbeg = _safeLongArray(ncols)
    qmatind = LAU.int_list_to_array([])
    qmatval = LAU.double_list_to_array([])
    space = 0
    surplus = CR.cpxlongPtr()
    status = CR.CPXXgetquad(env, lp, nzcnt, qmatbeg, qmatind, qmatval,
                            space, surplus, begin, end)
    if status != CR.CPXERR_NEGATIVE_SURPLUS:
        check_status(env, status)
    if surplus.value() == 0:
        return ([], [], [])
    space = -surplus.value()
    qmatind = _safeIntArray(space)
    qmatval = _safeDoubleArray(space)
    status = CR.CPXXgetquad(env, lp, nzcnt, qmatbeg, qmatind, qmatval,
                            space, surplus, begin, end)
    check_status(env, status)
    return (LAU.array_to_list(qmatbeg, ncols),
            LAU.array_to_list(qmatind, space),
            LAU.array_to_list(qmatval, space))


def getqpcoef(env, lp, row, col):
    val = CR.doublePtr()
    status = CR.CPXXgetqpcoef(env, lp, row, col, val)
    check_status(env, status)
    return val.value()


def getnumquad(env, lp):
    return CR.CPXXgetnumquad(env, lp)


def getnumqpnz(env, lp):
    return CR.CPXXgetnumqpnz(env, lp)

########################################################################
# Multi-Objective API
########################################################################

def getnumobjs(env, lp):
    return CR.CPXXgetnumobjs(env, lp)

def multiobjchgattribs(env, lp, objidx,
                       offset=_const.CPX_NO_OFFSET_CHANGE,
                       weight=_const.CPX_NO_WEIGHT_CHANGE,
                       priority=_const.CPX_NO_PRIORITY_CHANGE,
                       abstol=_const.CPX_NO_ABSTOL_CHANGE,
                       reltol=_const.CPX_NO_RELTOL_CHANGE,
                       name=None):
    status = CR.CPXXmultiobjchgattribs(env, lp, objidx, offset, weight,
                                       priority, abstol, reltol,
                                       name)
    check_status(env, status)

def multiobjgetindex(env, lp, name):
    indexfn = CR.CPXXmultiobjgetindex
    return _getindex(env, lp, name, indexfn)

def multiobjgetname(env, lp, objidx):
    namefn = CR.CPXXmultiobjgetname
    return _getname(env, lp, objidx, namefn, index_first=True)

def multiobjgetobj(env, lp, objidx, begin, end):
    coeffslen = _rangelen(begin, end)
    coeffs = _safeDoubleArray(coeffslen)
    offset = CR.doublePtr()
    weight = CR.doublePtr()
    priority = CR.intPtr()
    abstol = CR.doublePtr()
    reltol = CR.doublePtr()
    status = CR.CPXXmultiobjgetobj(env, lp, objidx, coeffs, begin, end,
                                   offset, weight, priority, abstol, reltol)
    check_status(env, status)
    return [LAU.array_to_list(coeffs, coeffslen), offset.value(),
            weight.value(), priority.value(), abstol.value(),
            reltol.value()]

def multiobjsetobj(env, lp, objidx, objind, objval,
                   offset=_const.CPX_NO_OFFSET_CHANGE,
                   weight=_const.CPX_NO_WEIGHT_CHANGE,
                   priority=_const.CPX_NO_PRIORITY_CHANGE,
                   abstol=_const.CPX_NO_ABSTOL_CHANGE,
                   reltol=_const.CPX_NO_RELTOL_CHANGE,
                   objname=None):
    objnz = len(objind)
    assert len(objval) == objnz
    with LAU.int_c_array(objind) as c_objind, \
         LAU.double_c_array(objval) as c_objval:  # noqa: E127
        status = CR.CPXXmultiobjsetobj(env, lp, objidx, objnz, c_objind,
                                       c_objval, offset, weight,
                                       priority, abstol, reltol,
                                       objname)
    check_status(env, status)

def setnumobjs(env, lp, n):
    status = CR.CPXXsetnumobjs(env, lp, n)
    check_status(env, status)

def multiobjopt(env, lp, paramsets):
    with SigIntHandler():
        status = CR.CPXXmultiobjopt(env, lp, paramsets)
    check_status(env, status)

def multiobjgetobjval(env, lp, objidx):
    objval_p = CR.doublePtr()
    status = CR.CPXXmultiobjgetobjval(env, lp, objidx, objval_p)
    check_status(env, status)
    return objval_p.value()

def multiobjgetobjvalbypriority(env, lp, priority):
    objval_p = CR.doublePtr()
    status = CR.CPXXmultiobjgetobjvalbypriority(env, lp, priority, objval_p)
    check_status(env, status)
    return objval_p.value()

def _multiobjgetinfo(fn, env, lp, subprob, info_p, what):
    status = fn(env, lp, subprob, info_p, what)
    check_status(env, status)
    return info_p.value()

def multiobjgetdblinfo(env, lp, subprob, what):
    info_p = CR.doublePtr()
    return _multiobjgetinfo(CR.CPXXmultiobjgetdblinfo, env, lp, subprob,
                            info_p, what)

def multiobjgetintinfo(env, lp, subprob, what):
    info_p = CR.intPtr()
    return _multiobjgetinfo(CR.CPXXmultiobjgetintinfo, env, lp, subprob,
                            info_p, what)

def multiobjgetlonginfo(env, lp, subprob, what):
    info_p = CR.cpxlongPtr()
    return _multiobjgetinfo(CR.CPXXmultiobjgetlonginfo, env, lp, subprob,
                            info_p, what)

def multiobjgetnumsolves(env, lp):
    return CR.CPXXmultiobjgetnumsolves(env, lp)

def getnumprios(env, lp):
    return CR.CPXEgetnumprios(env, lp)

def ismultiobj(env, lp):
    return CR.CPXEismultiobj(env, lp) != 0

# Optimizing Problems

# Accessing LP results

def solninfo(env, lp):
    lpstat = CR.intPtr()
    stype = CR.intPtr()
    pfeas = CR.intPtr()
    dfeas = CR.intPtr()
    status = CR.CPXXsolninfo(env, lp, lpstat, stype, pfeas, dfeas)
    check_status(env, status)
    return (lpstat.value(), stype.value(), pfeas.value(), dfeas.value())


def getstat(env, lp):
    return CR.CPXXgetstat(env, lp)


def getmethod(env, lp):
    return CR.CPXXgetmethod(env, lp)


def getobjval(env, lp):
    objval = CR.doublePtr()
    status = CR.CPXXgetobjval(env, lp, objval)
    check_status(env, status)
    return objval.value()


def getx(env, lp, begin, end):
    xlen = _rangelen(begin, end)
    x = _safeDoubleArray(xlen)
    status = CR.CPXXgetx(env, lp, x, begin, end)
    check_status(env, status)
    return LAU.array_to_list(x, xlen)


def getnumcores(env):
    numcores = CR.intPtr()
    status = CR.CPXXgetnumcores(env, numcores)
    check_status(env, status)
    return numcores.value()


def getax(env, lp, begin, end):
    axlen = _rangelen(begin, end)
    ax = _safeDoubleArray(axlen)
    status = CR.CPXXgetax(env, lp, ax, begin, end)
    check_status(env, status)
    return LAU.array_to_list(ax, axlen)


def getxqxax(env, lp, begin, end):
    qaxlen = _rangelen(begin, end)
    qax = _safeDoubleArray(qaxlen)
    status = CR.CPXXgetxqxax(env, lp, qax, begin, end)
    check_status(env, status)
    return LAU.array_to_list(qax, qaxlen)


def getpi(env, lp, begin, end):
    pilen = _rangelen(begin, end)
    pi = _safeDoubleArray(pilen)
    status = CR.CPXXgetpi(env, lp, pi, begin, end)
    check_status(env, status)
    return LAU.array_to_list(pi, pilen)


def getslack(env, lp, begin, end):
    slacklen = _rangelen(begin, end)
    slack = _safeDoubleArray(slacklen)
    status = CR.CPXXgetslack(env, lp, slack, begin, end)
    check_status(env, status)
    return LAU.array_to_list(slack, slacklen)


def getdj(env, lp, begin, end):
    djlen = _rangelen(begin, end)
    dj = _safeDoubleArray(djlen)
    status = CR.CPXXgetdj(env, lp, dj, begin, end)
    check_status(env, status)
    return LAU.array_to_list(dj, djlen)


def getqconstrdslack(env, lp, qind):
    inout_list = [0, qind]
    status = CR.CPXXgetqconstrdslack(env, lp, inout_list)
    if status != CR.CPXERR_NEGATIVE_SURPLUS:
        check_status(env, status)
    if inout_list == [0]:
        return ([], [])
    inout_list.extend([qind])
    status = CR.CPXXgetqconstrdslack(env, lp, inout_list)
    check_status(env, status)
    return tuple(inout_list)


# Infeasibility

def getrowinfeas(env, lp, x, begin, end):
    infeasoutlen = _rangelen(begin, end)
    infeasout = _safeDoubleArray(infeasoutlen)
    status = CR.CPXXgetrowinfeas(env, lp, LAU.double_list_to_array(x),
                                 infeasout, begin, end)
    check_status(env, status)
    return LAU.array_to_list(infeasout, infeasoutlen)


def getcolinfeas(env, lp, x, begin, end):
    infeasoutlen = _rangelen(begin, end)
    infeasout = _safeDoubleArray(infeasoutlen)
    status = CR.CPXXgetcolinfeas(env, lp, LAU.double_list_to_array(x),
                                 infeasout, begin, end)
    check_status(env, status)
    return LAU.array_to_list(infeasout, infeasoutlen)


def getqconstrinfeas(env, lp, x, begin, end):
    infeasoutlen = _rangelen(begin, end)
    infeasout = _safeDoubleArray(infeasoutlen)
    status = CR.CPXXgetqconstrinfeas(env, lp, LAU.double_list_to_array(x),
                                     infeasout, begin, end)
    check_status(env, status)
    return LAU.array_to_list(infeasout, infeasoutlen)


def getindconstrinfeas(env, lp, x, begin, end):
    infeasoutlen = _rangelen(begin, end)
    infeasout = _safeDoubleArray(infeasoutlen)
    status = CR.CPXXgetindconstrinfeas(env, lp, LAU.double_list_to_array(x),
                                       infeasout, begin, end)
    check_status(env, status)
    return LAU.array_to_list(infeasout, infeasoutlen)


def getsosinfeas(env, lp, x, begin, end):
    infeasoutlen = _rangelen(begin, end)
    infeasout = _safeDoubleArray(infeasoutlen)
    status = CR.CPXXgetsosinfeas(env, lp, LAU.double_list_to_array(x),
                                 infeasout, begin, end)
    check_status(env, status)
    return LAU.array_to_list(infeasout, infeasoutlen)

# Basis


def getbase(env, lp):
    numcols = CR.CPXXgetnumcols(env, lp)
    numrows = CR.CPXXgetnumrows(env, lp)
    cstat = _safeIntArray(numcols)
    rstat = _safeIntArray(numrows)
    status = CR.CPXXgetbase(env, lp, cstat, rstat)
    check_status(env, status)
    return (LAU.array_to_list(cstat, numcols),
            LAU.array_to_list(rstat, numrows))


def getbhead(env, lp):
    headlen = CR.CPXXgetnumrows(env, lp)
    head = _safeIntArray(headlen)
    x = _safeDoubleArray(headlen)
    status = CR.CPXXgetbhead(env, lp, head, x)
    check_status(env, status)
    return (LAU.array_to_list(head, headlen),
            LAU.array_to_list(x, headlen))


def mbasewrite(env, lp, filename):
    status = CR.CPXXmbasewrite(env, lp, filename)
    check_status(env, status)


def getijrow(env, lp, idx, is_row_index):
    row = CR.intPtr()
    if is_row_index:
        i, j = (idx, -1)  # Seek a basic row.
    else:
        i, j = (-1, idx)  # Seek a basic column.
    status = CR.CPXXgetijrow(env, lp, i, j, row)
    if status == CR.CPXERR_INDEX_NOT_BASIC:
        return -1
    check_status(env, status)
    return row.value()


def getpnorms(env, lp):
    numcols = CR.CPXXgetnumcols(env, lp)
    numrows = CR.CPXXgetnumrows(env, lp)
    cnorm = _safeDoubleArray(numcols)
    rnorm = _safeDoubleArray(numrows)
    length = CR.intPtr()
    status = CR.CPXXgetpnorms(env, lp, cnorm, rnorm, length)
    check_status(env, status)
    return (LAU.array_to_list(cnorm, length.value()),
            LAU.array_to_list(rnorm, numrows))


def getdnorms(env, lp):
    numrows = CR.CPXXgetnumrows(env, lp)
    norm = _safeDoubleArray(numrows)
    head = _safeIntArray(numrows)
    length = CR.intPtr()
    status = CR.CPXXgetdnorms(env, lp, norm, head, length)
    check_status(env, status)
    return (LAU.array_to_list(norm, length.value()),
            LAU.array_to_list(head, length.value()))


def getbasednorms(env, lp):
    numcols = CR.CPXXgetnumcols(env, lp)
    numrows = CR.CPXXgetnumrows(env, lp)
    cstat = _safeIntArray(numcols)
    rstat = _safeIntArray(numrows)
    dnorm = _safeDoubleArray(numrows)
    status = CR.CPXXgetbasednorms(env, lp, cstat, rstat, dnorm)
    check_status(env, status)
    return (LAU.array_to_list(cstat, numcols),
            LAU.array_to_list(rstat, numrows),
            LAU.array_to_list(dnorm, numrows))


def getpsbcnt(env, lp):
    return CR.CPXXgetpsbcnt(env, lp)


def getdsbcnt(env, lp):
    return CR.CPXXgetdsbcnt(env, lp)


def getdblquality(env, lp, what):
    quality = CR.doublePtr()
    status = CR.CPXXgetdblquality(env, lp, quality, what)
    check_status(env, status)
    return quality.value()


def getintquality(env, lp, what):
    quality = CR.intPtr()
    status = CR.CPXXgetintquality(env, lp, quality, what)
    check_status(env, status)
    return quality.value()


# Sensitivity Analysis Results

def boundsa_lower(env, lp, begin, end):
    listlen = _rangelen(begin, end)
    lblower = _safeDoubleArray(listlen)
    lbupper = _safeDoubleArray(listlen)
    ublower = LAU.double_list_to_array([])
    ubupper = LAU.double_list_to_array([])
    status = CR.CPXXboundsa(env, lp, begin, end, lblower, lbupper,
                            ublower, ubupper)
    check_status(env, status)
    return (LAU.array_to_list(lblower, listlen),
            LAU.array_to_list(lbupper, listlen))


def boundsa_upper(env, lp, begin, end):
    listlen = _rangelen(begin, end)
    lblower = LAU.double_list_to_array([])
    lbupper = LAU.double_list_to_array([])
    ublower = _safeDoubleArray(listlen)
    ubupper = _safeDoubleArray(listlen)
    status = CR.CPXXboundsa(env, lp, begin, end, lblower, lbupper,
                            ublower, ubupper)
    check_status(env, status)
    return (LAU.array_to_list(ublower, listlen),
            LAU.array_to_list(ubupper, listlen))


def boundsa(env, lp, begin, end):
    listlen = _rangelen(begin, end)
    lblower = _safeDoubleArray(listlen)
    lbupper = _safeDoubleArray(listlen)
    ublower = _safeDoubleArray(listlen)
    ubupper = _safeDoubleArray(listlen)
    status = CR.CPXXboundsa(env, lp, begin, end, lblower, lbupper,
                            ublower, ubupper)
    check_status(env, status)
    return (LAU.array_to_list(lblower, listlen),
            LAU.array_to_list(lbupper, listlen),
            LAU.array_to_list(ublower, listlen),
            LAU.array_to_list(ubupper, listlen))


def objsa(env, lp, begin, end):
    listlen = _rangelen(begin, end)
    lower = _safeDoubleArray(listlen)
    upper = _safeDoubleArray(listlen)
    status = CR.CPXXobjsa(env, lp, begin, end, lower, upper)
    check_status(env, status)
    return (LAU.array_to_list(lower, listlen),
            LAU.array_to_list(upper, listlen))


def rhssa(env, lp, begin, end):
    listlen = _rangelen(begin, end)
    lower = _safeDoubleArray(listlen)
    upper = _safeDoubleArray(listlen)
    status = CR.CPXXrhssa(env, lp, begin, end, lower, upper)
    check_status(env, status)
    return (LAU.array_to_list(lower, listlen),
            LAU.array_to_list(upper, listlen))


# Conflicts

def refinemipstartconflictext(env, lp, mipstartindex, grppref, grpbeg,
                              grpind, grptype):
    grpcnt = _arraylen(grppref)
    concnt = _arraylen(grpind)
    with SigIntHandler(), \
            LAU.double_c_array_or_none(grppref) as c_grppref, \
            LAU.long_c_array_or_none(grpbeg) as c_grpbeg, \
            LAU.int_c_array_or_none(grpind) as c_grpind:
        status = CR.CPXXrefinemipstartconflictext(
            env, lp, mipstartindex, grpcnt, concnt,
            c_grppref, c_grpbeg, c_grpind, grptype)
    check_status(env, status)


def refineconflictext(env, lp, grppref, grpbeg, grpind, grptype):
    grpcnt = _arraylen(grppref)
    concnt = _arraylen(grpind)
    with SigIntHandler(), \
            LAU.double_c_array_or_none(grppref) as c_grppref, \
            LAU.long_c_array_or_none(grpbeg) as c_grpbeg, \
            LAU.int_c_array_or_none(grpind) as c_grpind:
        status = CR.CPXXrefineconflictext(
            env, lp, grpcnt, concnt,
            c_grppref, c_grpbeg, c_grpind, grptype)
    check_status(env, status)


def getconflictext(env, lp, begin, end):
    grpstatlen = _rangelen(begin, end)
    grpstat = _safeIntArray(grpstatlen)
    status = CR.CPXXgetconflictext(env, lp, grpstat, begin, end)
    check_status(env, status)
    return LAU.array_to_list(grpstat, grpstatlen)


def getconflictnumgroups(env, lp):
    return CR.CPXXgetconflictnumgroups(env, lp)


def getconflictgroups(env, lp, begin, end):
    inout_list = [0, begin, end]
    status = CR.CPXXgetconflictgroups(env, lp, inout_list)
    if status != CR.CPXERR_NEGATIVE_SURPLUS:
        check_status(env, status)
    if inout_list == [0]:
        return ([], [0] * _rangelen(begin, end), [], [])
    inout_list.extend([begin, end])
    status = CR.CPXXgetconflictgroups(env, lp, inout_list)
    check_status(env, status)
    # We expect to get [grppref, grpbeg, grpind, grptype].
    assert len(inout_list) == 4, str(inout_list)
    return tuple(inout_list)


def getconflictnumpasses(env, lp):
    return CR.CPXXgetconflictnumpasses(env, lp)


def clpwrite(env, lp, filename):
    status = CR.CPXXclpwrite(env, lp, filename)
    check_status(env, status)

# Problem Modification Routines

# File Reading Routines

# File Writing Routines


def solwrite(env, lp, filename):
    status = CR.CPXXsolwrite(env, lp, filename)
    check_status(env, status)

# Message Handling Routines

# Advanced LP Routines


def binvcol(env, lp, j):
    xlen = CR.CPXXgetnumrows(env, lp)
    x = _safeDoubleArray(xlen)
    status = CR.CPXXbinvcol(env, lp, j, x)
    check_status(env, status)
    return LAU.array_to_list(x, xlen)


def binvrow(env, lp, i):
    ylen = CR.CPXXgetnumrows(env, lp)
    y = _safeDoubleArray(ylen)
    status = CR.CPXXbinvrow(env, lp, i, y)
    check_status(env, status)
    return LAU.array_to_list(y, ylen)


def binvacol(env, lp, j):
    xlen = CR.CPXXgetnumrows(env, lp)
    x = _safeDoubleArray(xlen)
    status = CR.CPXXbinvacol(env, lp, j, x)
    check_status(env, status)
    return LAU.array_to_list(x, xlen)


def binvarow(env, lp, i):
    zlen = CR.CPXXgetnumcols(env, lp)
    z = _safeDoubleArray(zlen)
    status = CR.CPXXbinvarow(env, lp, i, z)
    check_status(env, status)
    return LAU.array_to_list(z, zlen)


def ftran(env, lp, x):
    x_array = LAU.double_list_to_array(x)
    status = CR.CPXXftran(env, lp, x_array)
    check_status(env, status)
    return LAU.array_to_list(x_array, len(x))


def btran(env, lp, y):
    y_array = LAU.double_list_to_array(y)
    status = CR.CPXXbtran(env, lp, y_array)
    check_status(env, status)
    return LAU.array_to_list(y_array, len(y))


# Advanced Solution functions

def getgrad(env, lp, j):
    numrows = CR.CPXXgetnumrows(env, lp)
    head = _safeIntArray(numrows)
    y = _safeDoubleArray(numrows)
    status = CR.CPXXgetgrad(env, lp, j, head, y)
    check_status(env, status)
    return (LAU.array_to_list(head, numrows),
            LAU.array_to_list(y, numrows))


def slackfromx(env, lp, x):
    numrows = CR.CPXXgetnumrows(env, lp)
    slack = _safeDoubleArray(numrows)
    status = CR.CPXXslackfromx(env, lp, LAU.double_list_to_array(x), slack)
    check_status(env, status)
    return LAU.array_to_list(slack, numrows)


def qconstrslackfromx(env, lp, x):
    numqcon = CR.CPXXgetnumqconstrs(env, lp)
    slack = _safeDoubleArray(numqcon)
    status = CR.CPXXqconstrslackfromx(env, lp,
                                      LAU.double_list_to_array(x), slack)
    check_status(env, status)
    return LAU.array_to_list(slack, numqcon)


def djfrompi(env, lp, pi):
    numcols = CR.CPXXgetnumcols(env, lp)
    dj = _safeDoubleArray(numcols)
    status = CR.CPXXdjfrompi(env, lp, LAU.double_list_to_array(pi), dj)
    check_status(env, status)
    return LAU.array_to_list(dj, numcols)


def qpdjfrompi(env, lp, pi, x):
    numcols = CR.CPXXgetnumcols(env, lp)
    dj = _safeDoubleArray(numcols)
    status = CR.CPXXqpdjfrompi(env, lp, LAU.double_list_to_array(pi),
                               LAU.double_list_to_array(x), dj)
    check_status(env, status)
    return LAU.array_to_list(dj, numcols)


def mdleave(env, lp, goodlist):
    goodlen = len(goodlist)
    downratio = _safeDoubleArray(goodlen)
    upratio = _safeDoubleArray(goodlen)
    status = CR.CPXXmdleave(env, lp, LAU.int_list_to_array(goodlist),
                            goodlen, downratio, upratio)
    check_status(env, status)
    return (LAU.array_to_list(downratio, goodlen),
            LAU.array_to_list(upratio, goodlen))


def qpindefcertificate(env, lp):
    certlen = CR.CPXXgetnumquad(env, lp)
    cert = _safeDoubleArray(certlen)
    status = CR.CPXXqpindefcertificate(env, lp, cert)
    check_status(env, status)
    return LAU.array_to_list(cert, certlen)


def dualfarkas(env, lp):
    ylen = CR.CPXXgetnumrows(env, lp)
    y = _safeDoubleArray(ylen)
    proof = CR.doublePtr()
    status = CR.CPXXdualfarkas(env, lp, y, proof)
    check_status(env, status)
    return (LAU.array_to_list(y, ylen), proof.value())


def getijdiv(env, lp):
    idiv = CR.intPtr()
    jdiv = CR.intPtr()
    status = CR.CPXXgetijdiv(env, lp, idiv, jdiv)
    check_status(env, status)
    if idiv.value() != -1:
        return idiv.value() + getnumcols(env, lp)
    if jdiv.value() != -1:
        return jdiv.value()
    # problem is not diverging
    return -1


def getray(env, lp):
    zlen = CR.CPXXgetnumcols(env, lp)
    z = _safeDoubleArray(zlen)
    status = CR.CPXXgetray(env, lp, z)
    check_status(env, status)
    return LAU.array_to_list(z, zlen)


# Advanced Presolve Routines

def presolve(env, lp, method):
    status = CR.CPXXpresolve(env, lp, method)
    check_status(env, status)


def freepresolve(env, lp):
    status = CR.CPXXfreepresolve(env, lp)
    check_status(env, status)


def crushx(env, lp, x):
    redlp = CR.CPXLPptrPtr()
    status = CR.CPXXgetredlp(env, lp, redlp)
    check_status(env, status)
    if redlp.value() is None:
        raise CplexError("No presolved problem found")
    numcols = CR.CPXXgetnumcols(env, redlp.value())
    prex = _safeDoubleArray(numcols)
    status = CR.CPXXcrushx(env, lp, LAU.double_list_to_array(x), prex)
    check_status(env, status)
    return LAU.array_to_list(prex, numcols)


def uncrushx(env, lp, prex):
    numcols = CR.CPXXgetnumcols(env, lp)
    x = _safeDoubleArray(numcols)
    status = CR.CPXXuncrushx(env, lp, x, LAU.double_list_to_array(prex))
    check_status(env, status)
    return LAU.array_to_list(x, numcols)


def crushpi(env, lp, pi):
    redlp = CR.CPXLPptrPtr()
    status = CR.CPXXgetredlp(env, lp, redlp)
    check_status(env, status)
    if redlp.value() is None:
        raise CplexError("No presolved problem found")
    numrows = CR.CPXXgetnumrows(env, redlp.value())
    prepi = _safeDoubleArray(numrows)
    status = CR.CPXXcrushpi(env, lp, LAU.double_list_to_array(pi), prepi)
    check_status(env, status)
    return LAU.array_to_list(prepi, numrows)


def uncrushpi(env, lp, prepi):
    numrows = CR.CPXXgetnumrows(env, lp)
    pi = _safeDoubleArray(numrows)
    status = CR.CPXXuncrushpi(env, lp, pi, LAU.double_list_to_array(prepi))
    check_status(env, status)
    return LAU.array_to_list(pi, numrows)


def crushform(env, lp, ind, val):
    plen = CR.intPtr()
    poffset = CR.doublePtr()
    redlp = CR.CPXLPptrPtr()
    status = CR.CPXXgetredlp(env, lp, redlp)
    check_status(env, status)
    if redlp.value() is None:
        raise CplexError("No presolved problem found")
    numcols = CR.CPXXgetnumcols(env, redlp.value())
    pind = _safeIntArray(numcols)
    pval = _safeDoubleArray(numcols)
    status = CR.CPXXcrushform(env, lp, len(ind),
                              LAU.int_list_to_array(ind),
                              LAU.double_list_to_array(val),
                              plen, poffset, pind, pval)
    check_status(env, status)
    return (poffset.value(), LAU.array_to_list(pind, plen.value()),
            LAU.array_to_list(pval, plen.value()))


def uncrushform(env, lp, pind, pval):
    length = CR.intPtr()
    offset = CR.doublePtr()
    maxlen = CR.CPXXgetnumcols(env, lp) + CR.CPXXgetnumrows(env, lp)
    ind = _safeIntArray(maxlen)
    val = _safeDoubleArray(maxlen)
    status = CR.CPXXuncrushform(env, lp, len(pind),
                                LAU.int_list_to_array(pind),
                                LAU.double_list_to_array(pval),
                                length, offset, ind, val)
    check_status(env, status)
    return (offset.value(), LAU.array_to_list(ind, length.value()),
            LAU.array_to_list(val, length.value()))


def getprestat_status(env, lp):
    redlp = CR.CPXLPptrPtr()
    status = CR.CPXXgetredlp(env, lp, redlp)
    check_status(env, status)
    if redlp.value() is None:
        raise CplexError("No presolved problem found")
    prestat = CR.intPtr()
    pcstat = LAU.int_list_to_array([])
    prstat = LAU.int_list_to_array([])
    ocstat = LAU.int_list_to_array([])
    orstat = LAU.int_list_to_array([])
    status = CR.CPXXgetprestat(env, lp, prestat, pcstat, prstat,
                               ocstat, orstat)
    check_status(env, status)
    return prestat.value()


def getprestat_r(env, lp):
    redlp = CR.CPXLPptrPtr()
    status = CR.CPXXgetredlp(env, lp, redlp)
    check_status(env, status)
    if redlp.value() is None:
        raise CplexError("No presolved problem found")
    nrows = CR.CPXXgetnumrows(env, lp)
    prestat = CR.intPtr()
    pcstat = LAU.int_list_to_array([])
    prstat = _safeIntArray(nrows)
    ocstat = LAU.int_list_to_array([])
    orstat = LAU.int_list_to_array([])
    status = CR.CPXXgetprestat(env, lp, prestat, pcstat, prstat,
                               ocstat, orstat)
    check_status(env, status)
    return LAU.array_to_list(prstat, nrows)


def getprestat_c(env, lp):
    redlp = CR.CPXLPptrPtr()
    status = CR.CPXXgetredlp(env, lp, redlp)
    check_status(env, status)
    if redlp.value() is None:
        raise CplexError("No presolved problem found")
    ncols = CR.CPXXgetnumcols(env, lp)
    prestat = CR.intPtr()
    pcstat = _safeIntArray(ncols)
    prstat = LAU.int_list_to_array([])
    ocstat = LAU.int_list_to_array([])
    orstat = LAU.int_list_to_array([])
    status = CR.CPXXgetprestat(env, lp, prestat, pcstat, prstat,
                               ocstat, orstat)
    check_status(env, status)
    return LAU.array_to_list(pcstat, ncols)


def getprestat_or(env, lp):
    redlp = CR.CPXLPptrPtr()
    status = CR.CPXXgetredlp(env, lp, redlp)
    check_status(env, status)
    if redlp.value() is None:
        raise CplexError("No presolved problem found")
    nprows = CR.CPXXgetnumrows(env, redlp.value())
    prestat = CR.intPtr()
    pcstat = LAU.int_list_to_array([])
    prstat = LAU.int_list_to_array([])
    ocstat = LAU.int_list_to_array([])
    orstat = _safeIntArray(nprows)
    status = CR.CPXXgetprestat(env, lp, prestat, pcstat, prstat,
                               ocstat, orstat)
    check_status(env, status)
    return LAU.array_to_list(orstat, nprows)


def getprestat_oc(env, lp):
    redlp = CR.CPXLPptrPtr()
    status = CR.CPXXgetredlp(env, lp, redlp)
    check_status(env, status)
    if redlp.value() is None:
        raise CplexError("No presolved problem found")
    npcols = CR.CPXXgetnumcols(env, redlp.value())
    prestat = CR.intPtr()
    pcstat = LAU.int_list_to_array([])
    prstat = LAU.int_list_to_array([])
    ocstat = _safeIntArray(npcols)
    orstat = LAU.int_list_to_array([])
    status = CR.CPXXgetprestat(env, lp, prestat, pcstat, prstat,
                               ocstat, orstat)
    check_status(env, status)
    return LAU.array_to_list(ocstat, npcols)


def prechgobj(env, lp, ind, val):
    status = CR.CPXXprechgobj(env, lp, len(ind),
                              LAU.int_list_to_array(ind),
                              LAU.double_list_to_array(val))
    check_status(env, status)


def preaddrows(env, lp, rhs, sense, rmatbeg, rmatind, rmatval, names):
    with LAU.double_c_array(rhs) as c_rhs, \
         LAU.long_c_array(rmatbeg) as c_rmatbeg, \
         LAU.int_c_array(rmatind) as c_rmatind, \
         LAU.double_c_array(rmatval) as c_rmatval:  # noqa: E126
        status = CR.CPXXpreaddrows(env, lp, len(rmatbeg), len(rmatind),
                                   c_rhs,
                                   sense,
                                   c_rmatbeg,
                                   c_rmatind,
                                   c_rmatval,
                                   names)
    check_status(env, status)

########################################################################
# MIP Starts API
########################################################################


def getnummipstarts(env, lp):
    return CR.CPXXgetnummipstarts(env, lp)


def chgmipstarts(env, lp, mipstartindices, beg, varindices, values,
                 effortlevel):
    with LAU.int_c_array(mipstartindices) as c_mipstartindices, \
            LAU.long_c_array(beg) as c_beg, \
            LAU.int_c_array(varindices) as c_varindices, \
            LAU.double_c_array(values) as c_values, \
            LAU.int_c_array(effortlevel) as c_effortlevel:
        status = CR.CPXXchgmipstarts(env, lp,
                                     len(mipstartindices),
                                     c_mipstartindices,
                                     len(varindices),
                                     c_beg,
                                     c_varindices,
                                     c_values,
                                     c_effortlevel)
    check_status(env, status)


def addmipstarts(env, lp, beg, varindices, values, effortlevel,
                 mipstartname):
    with LAU.long_c_array(beg) as c_beg, \
            LAU.int_c_array(varindices) as c_varindices, \
            LAU.double_c_array(values) as c_values, \
            LAU.int_c_array(effortlevel) as c_effortlevel:
        status = CR.CPXXaddmipstarts(
            env, lp, len(beg), len(varindices),
            c_beg, c_varindices, c_values, c_effortlevel,
            mipstartname)
    check_status(env, status)


def delmipstarts(env, lp, begin, end):
    delfn = CR.CPXXdelmipstarts
    _delbyrange(delfn, env, lp, begin, end)


def getmipstarts_size(env, lp, begin, end):
    beglen = _rangelen(begin, end)
    beg = LAU.long_list_to_array([])
    effortlevel = _safeIntArray(beglen)
    nzcnt = CR.cpxlongPtr()
    surplus = CR.cpxlongPtr()
    varindices = LAU.int_list_to_array([])
    values = LAU.double_list_to_array([])
    startspace = 0
    status = CR.CPXXgetmipstarts(env, lp, nzcnt, beg, varindices, values,
                                 effortlevel, startspace, surplus, begin,
                                 end)
    if status != CR.CPXERR_NEGATIVE_SURPLUS:
        check_status(env, status)
    return -surplus.value()


def getmipstarts_effort(env, lp, begin, end):
    beglen = _rangelen(begin, end)
    beg = LAU.long_list_to_array([])
    effortlevel = _safeIntArray(beglen)
    nzcnt = CR.cpxlongPtr()
    surplus = CR.cpxlongPtr()
    varindices = LAU.int_list_to_array([])
    values = LAU.double_list_to_array([])
    startspace = 0
    status = CR.CPXXgetmipstarts(env, lp, nzcnt, beg, varindices, values,
                                 effortlevel, startspace, surplus, begin,
                                 end)
    if status != CR.CPXERR_NEGATIVE_SURPLUS:
        check_status(env, status)
    if surplus.value() == 0:
        return ([0] * _rangelen(begin, end), [], [],
                [0] * _rangelen(begin, end))
    startspace = -surplus.value()
    beg = _safeLongArray(beglen)
    varindices = _safeIntArray(startspace)
    values = _safeDoubleArray(startspace)
    status = CR.CPXXgetmipstarts(env, lp, nzcnt, beg, varindices, values,
                                 effortlevel, startspace, surplus, begin,
                                 end)
    check_status(env, status)
    return LAU.array_to_list(effortlevel, beglen)


def getmipstarts(env, lp, begin, end):
    beglen = _rangelen(begin, end)
    beg = LAU.long_list_to_array([])
    effortlevel = _safeIntArray(beglen)
    nzcnt = CR.cpxlongPtr()
    surplus = CR.cpxlongPtr()
    varindices = LAU.int_list_to_array([])
    values = LAU.double_list_to_array([])
    startspace = 0
    status = CR.CPXXgetmipstarts(env, lp, nzcnt, beg, varindices, values,
                                 effortlevel, startspace, surplus, begin,
                                 end)
    if status != CR.CPXERR_NEGATIVE_SURPLUS:
        check_status(env, status)
    if surplus.value() == 0:
        return ([0] * _rangelen(begin, end), [], [],
                [0] * _rangelen(begin, end))
    beg = _safeLongArray(beglen)
    startspace = -surplus.value()
    varindices = _safeIntArray(startspace)
    values = _safeDoubleArray(startspace)
    status = CR.CPXXgetmipstarts(env, lp, nzcnt, beg, varindices, values,
                                 effortlevel, startspace, surplus, begin,
                                 end)
    check_status(env, status)
    return (LAU.array_to_list(beg, beglen),
            LAU.array_to_list(varindices, startspace),
            LAU.array_to_list(values, startspace),
            LAU.array_to_list(effortlevel, beglen))


def getmipstartname(env, lp, begin, end):
    namefn = CR.CPXXgetmipstartname
    return _getnamebyrange(env, lp, begin, end, namefn)


def getmipstartindex(env, lp, mipstartname):
    index = CR.intPtr()
    status = CR.CPXXgetmipstartindex(env, lp, mipstartname, index)
    check_status(env, status)
    return index.value()


def readcopymipstarts(env, lp, filename):
    status = CR.CPXXreadcopymipstarts(env, lp,
                                      filename)
    check_status(env, status)


def writemipstarts(env, lp, filename, begin, end):
    status = CR.CPXXwritemipstarts(env, lp, filename, begin, end)
    check_status(env, status)

# Optimizing Problems

# Progress


def getitcnt(env, lp):
    return CR.CPXXgetitcnt(env, lp)


def getphase1cnt(env, lp):
    return CR.CPXXgetphase1cnt(env, lp)


def getsiftitcnt(env, lp):
    return CR.CPXXgetsiftitcnt(env, lp)


def getsiftphase1cnt(env, lp):
    return CR.CPXXgetsiftphase1cnt(env, lp)


def getbaritcnt(env, lp):
    return CR.CPXXgetbaritcnt(env, lp)


def getcrossppushcnt(env, lp):
    return CR.CPXXgetcrossppushcnt(env, lp)


def getcrosspexchcnt(env, lp):
    return CR.CPXXgetcrosspexchcnt(env, lp)


def getcrossdpushcnt(env, lp):
    return CR.CPXXgetcrossdpushcnt(env, lp)


def getcrossdexchcnt(env, lp):
    return CR.CPXXgetcrossdexchcnt(env, lp)


def getmipitcnt(env, lp):
    return CR.CPXXgetmipitcnt(env, lp)


def getnodecnt(env, lp):
    return CR.CPXXgetnodecnt(env, lp)


def getnodeleftcnt(env, lp):
    return CR.CPXXgetnodeleftcnt(env, lp)


# MIP Only solution interface

def getbestobjval(env, lp):
    objval = CR.doublePtr()
    status = CR.CPXXgetbestobjval(env, lp, objval)
    check_status(env, status)
    return objval.value()


def getcutoff(env, lp):
    cutoff = CR.doublePtr()
    status = CR.CPXXgetcutoff(env, lp, cutoff)
    check_status(env, status)
    return cutoff.value()


def getmiprelgap(env, lp):
    relgap = CR.doublePtr()
    status = CR.CPXXgetmiprelgap(env, lp, relgap)
    check_status(env, status)
    return relgap.value()


def getnumcuts(env, lp, cuttype):
    num = CR.intPtr()
    status = CR.CPXXgetnumcuts(env, lp, cuttype, num)
    check_status(env, status)
    return num.value()


def getnodeint(env, lp):
    return CR.CPXXgetnodeint(env, lp)


def getsubstat(env, lp):
    return CR.CPXXgetsubstat(env, lp)

# for callback query methods


def get_wherefrom(cbstruct):
    return CR.get_wherefrom(cbstruct)


cpxlong_callback_node_info = [
    _const.CPX_CALLBACK_INFO_NODE_SEQNUM_LONG,
    _const.CPX_CALLBACK_INFO_NODE_NODENUM_LONG,
    _const.CPX_CALLBACK_INFO_NODE_DEPTH_LONG,
]

int_callback_node_info = [
    _const.CPX_CALLBACK_INFO_NODE_NIINF,
    _const.CPX_CALLBACK_INFO_NODE_VAR,
    _const.CPX_CALLBACK_INFO_NODE_SOS,
    _const.CPX_CALLBACK_INFO_LAZY_SOURCE,
]

double_callback_node_info = [
    _const.CPX_CALLBACK_INFO_NODE_SIINF,
    _const.CPX_CALLBACK_INFO_NODE_ESTIMATE,
    _const.CPX_CALLBACK_INFO_NODE_OBJVAL,
]

# NB: CPX_CALLBACK_INFO_NODE_TYPE not used in the Python API.

user_handle_callback_node_info = [
    _const.CPX_CALLBACK_INFO_NODE_USERHANDLE
]


def gettime(env):
    time = CR.doublePtr()
    status = CR.CPXXgettime(env, time)
    check_status(env, status)
    return time.value()


def getdettime(env):
    time = CR.doublePtr()
    status = CR.CPXXgetdettime(env, time)
    check_status(env, status)
    return time.value()


def getcallbackincumbent(cbstruct, begin, end):
    xlen = _rangelen(begin, end)
    x = _safeDoubleArray(xlen)
    status = CR.CPXXgetcallbackincumbent(cbstruct, x, begin, end)
    check_status(None, status)
    return LAU.array_to_list(x, xlen)


def getcallbackpseudocosts(cbstruct, begin, end):
    pclen = _rangelen(begin, end)
    uppc = _safeDoubleArray(pclen)
    dnpc = _safeDoubleArray(pclen)
    status = CR.CPXXgetcallbackpseudocosts(cbstruct, uppc, dnpc, begin, end)
    check_status(None, status)
    return (LAU.array_to_list(uppc, pclen),
            LAU.array_to_list(dnpc, pclen))


def getcallbacknodeintfeas(cbstruct, begin, end):
    feaslen = _rangelen(begin, end)
    feas = _safeIntArray(feaslen)
    status = CR.CPXXgetcallbacknodeintfeas(cbstruct, feas, begin, end)
    check_status(None, status)
    return LAU.array_to_list(feas, feaslen)


def getcallbacknodelb(cbstruct, begin, end):
    lblen = _rangelen(begin, end)
    lb = _safeDoubleArray(lblen)
    status = CR.CPXXgetcallbacknodelb(cbstruct, lb, begin, end)
    check_status(None, status)
    return LAU.array_to_list(lb, lblen)


def getcallbacknodeub(cbstruct, begin, end):
    ublen = _rangelen(begin, end)
    ub = _safeDoubleArray(ublen)
    status = CR.CPXXgetcallbacknodeub(cbstruct, ub, begin, end)
    check_status(None, status)
    return LAU.array_to_list(ub, ublen)


def getcallbacknodeobjval(cbstruct):
    x = CR.doublePtr()
    status = CR.CPXXgetcallbacknodeobjval(cbstruct, x)
    check_status(None, status)
    return x.value()


def getcallbacknodex(cbstruct, begin, end):
    xlen = _rangelen(begin, end)
    x = _safeDoubleArray(xlen)
    status = CR.CPXXgetcallbacknodex(cbstruct, x, begin, end)
    check_status(None, status)
    return LAU.array_to_list(x, xlen)


def getcallbacknodeinfo(cbstruct, node, which):
    if which in int_callback_node_info:
        data = CR.intPtr()
    elif which in cpxlong_callback_node_info:
        data = CR.cpxlongPtr()
    elif which in double_callback_node_info:
        data = CR.doublePtr()
    elif which in user_handle_callback_node_info:
        data = []
    else:
        raise CplexError(
            "invalid value for which in _internal._procedural.getcallbacknodeinfo")
    status = CR.CPXXgetcallbacknodeinfo(cbstruct, [node, which, data])
    check_status(None, status)
    if (which in int_callback_node_info or
            which in double_callback_node_info or
            which in cpxlong_callback_node_info):
        return data.value()
    assert which in user_handle_callback_node_info
    return data[0]


def callbacksetuserhandle(cbstruct, userhandle):
    data = []
    status = CR.CPXXcallbacksetuserhandle(cbstruct, [userhandle, data])
    check_status(None, status)
    return data[0]


def callbacksetnodeuserhandle(cbstruct, nodeindex, userhandle):
    data = []
    status = CR.CPXXcallbacksetnodeuserhandle(
        cbstruct, [nodeindex, userhandle, data])
    check_status(None, status)
    return data[0]


def getcallbackseqinfo(cbstruct, node, which):
    if which in int_callback_node_info:
        data = CR.intPtr()
    elif which in cpxlong_callback_node_info:
        data = CR.cpxlongPtr()
    elif which in double_callback_node_info:
        data = CR.doublePtr()
    elif which in user_handle_callback_node_info:
        data = []
    else:
        raise CplexError(
            "invalid value for which in _internal._procedural.getcallbackseqinfo")
    status = CR.CPXXgetcallbackseqinfo(cbstruct, [node, which, data])
    check_status(None, status)
    if (which in int_callback_node_info or
            which in double_callback_node_info or
            which in cpxlong_callback_node_info):
        return data.value()
    assert which in user_handle_callback_node_info
    return data[0]


int_sos_info = [
    _const.CPX_CALLBACK_INFO_SOS_NUM,
    _const.CPX_CALLBACK_INFO_SOS_SIZE,
    _const.CPX_CALLBACK_INFO_SOS_IS_FEASIBLE,
    _const.CPX_CALLBACK_INFO_SOS_MEMBER_INDEX,
]

double_sos_info = [
    _const.CPX_CALLBACK_INFO_SOS_MEMBER_REFVAL,
]

# NB: CPX_CALLBACK_INFO_SOS_TYPE not used in the Python API.


def getcallbacksosinfo(cbstruct, sosindex, member, which):
    if which in int_sos_info:
        data = CR.intPtr()
    elif which in double_sos_info:
        data = CR.doublePtr()
    else:
        raise CplexError(
            "invalid value for which in _internal._procedural.getcallbacksosinfo")
    status = CR.CPXXgetcallbacksosinfo(cbstruct, sosindex, member, which, data)
    check_status(None, status)
    return data.value()


def cutcallbackadd(cbstruct, rhs, sense, ind, val, purgeable):
    status = CR.CPXXcutcallbackadd(cbstruct, len(ind), rhs,
                                   sense,
                                   LAU.int_list_to_array(ind),
                                   LAU.double_list_to_array(val),
                                   purgeable)
    check_status(None, status)


def cutcallbackaddlocal(cbstruct, rhs, sense, ind, val):
    status = CR.CPXXcutcallbackaddlocal(cbstruct, len(ind), rhs,
                                        sense,
                                        LAU.int_list_to_array(ind),
                                        LAU.double_list_to_array(val))
    check_status(None, status)


def branchcallbackbranchgeneral(cbstruct, ind, lu, bd, rhs, sense, matbeg,
                                matind, matval, nodeest, userhandle):
    seqnum = CR.cpxlongPtr()
    status = CR.CPXXbranchcallbackbranchgeneral(
        cbstruct, len(ind),
        LAU.int_list_to_array(ind),
        lu,
        LAU.double_list_to_array(bd),
        len(matbeg), len(matind),
        LAU.double_list_to_array(rhs),
        sense,
        LAU.long_list_to_array(matbeg),
        LAU.int_list_to_array(matind),
        LAU.double_list_to_array(matval),
        nodeest, userhandle, seqnum)
    check_status(None, status)
    return seqnum.value()


def branchcallbackbranchasCPLEX(cbstruct, n, userhandle):
    seqnum = CR.cpxlongPtr()
    status = CR.CPXXbranchcallbackbranchasCPLEX(
        cbstruct, n, userhandle, seqnum)
    check_status(None, status)
    return seqnum.value()


def setpydel(env):
    status = CR.setpydel(env)
    check_status(env, status)


def delpydel(env):
    status = CR.delpydel(env)
    check_status(env, status)

# Solution pool


def addsolnpooldivfilter(env, lp, lb, ub, ind, wts, ref, name):
    status = CR.CPXXaddsolnpooldivfilter(env, lp, lb, ub, len(ind),
                                         LAU.int_list_to_array(ind),
                                         LAU.double_list_to_array(wts),
                                         LAU.double_list_to_array(ref),
                                         name)
    check_status(env, status)


def addsolnpoolrngfilter(env, lp, lb, ub, ind, val, name):
    status = CR.CPXXaddsolnpoolrngfilter(env, lp, lb, ub, len(ind),
                                         LAU.int_list_to_array(ind),
                                         LAU.double_list_to_array(val),
                                         name)
    check_status(env, status)


def getsolnpooldivfilter_constant(env, lp, which):
    lb = CR.doublePtr()
    ub = CR.doublePtr()
    nzcnt = CR.intPtr()
    space = 0
    surplus = CR.intPtr()
    ind = LAU.int_list_to_array([])
    wts = LAU.double_list_to_array([])
    ref = LAU.double_list_to_array([])
    status = CR.CPXXgetsolnpooldivfilter(env, lp, lb, ub, nzcnt, ind,
                                         wts, ref, space, surplus, which)
    if status != CR.CPXERR_NEGATIVE_SURPLUS:
        check_status(env, status)
    return (lb.value(), ub.value(), -surplus.value())


def getsolnpooldivfilter(env, lp, which):
    lb = CR.doublePtr()
    ub = CR.doublePtr()
    nzcnt = CR.intPtr()
    space = 0
    surplus = CR.intPtr()
    ind = LAU.int_list_to_array([])
    wts = LAU.double_list_to_array([])
    ref = LAU.double_list_to_array([])
    status = CR.CPXXgetsolnpooldivfilter(env, lp, lb, ub, nzcnt, ind,
                                         wts, ref, space, surplus, which)
    if status != CR.CPXERR_NEGATIVE_SURPLUS:
        check_status(env, status)
    space = -surplus.value()
    ind = _safeIntArray(space)
    wts = _safeDoubleArray(space)
    ref = _safeDoubleArray(space)
    status = CR.CPXXgetsolnpooldivfilter(env, lp, lb, ub, nzcnt, ind,
                                         wts, ref, space, surplus, which)
    check_status(env, status)
    return (lb.value(),
            ub.value(),
            LAU.array_to_list(ind, space),
            LAU.array_to_list(wts, space),
            LAU.array_to_list(ref, space))


def getsolnpoolrngfilter_constant(env, lp, which):
    lb = CR.doublePtr()
    ub = CR.doublePtr()
    nzcnt = CR.intPtr()
    space = 0
    surplus = CR.intPtr()
    ind = LAU.int_list_to_array([])
    val = LAU.double_list_to_array([])
    status = CR.CPXXgetsolnpoolrngfilter(env, lp, lb, ub, nzcnt, ind, val,
                                         space, surplus, which)
    if status != CR.CPXERR_NEGATIVE_SURPLUS:
        check_status(env, status)
    return (lb.value(), ub.value(), -surplus.value())


def getsolnpoolrngfilter(env, lp, which):
    lb = CR.doublePtr()
    ub = CR.doublePtr()
    nzcnt = CR.intPtr()
    space = 0
    surplus = CR.intPtr()
    ind = LAU.int_list_to_array([])
    val = LAU.double_list_to_array([])
    status = CR.CPXXgetsolnpoolrngfilter(env, lp, lb, ub, nzcnt, ind, val,
                                         space, surplus, which)
    if status != CR.CPXERR_NEGATIVE_SURPLUS:
        check_status(env, status)
    space = -surplus.value()
    ind = _safeIntArray(space)
    val = _safeDoubleArray(space)
    status = CR.CPXXgetsolnpoolrngfilter(env, lp, lb, ub, nzcnt, ind, val,
                                         space, surplus, which)
    check_status(env, status)
    return (lb.value(), ub.value(), LAU.array_to_list(ind, space),
            LAU.array_to_list(val, space))


def delsolnpoolfilters(env, lp, begin, end):
    delfn = CR.CPXXdelsolnpoolfilters
    _delbyrange(delfn, env, lp, begin, end)


def getsolnpoolfiltername(env, lp, which):
    namefn = CR.CPXXgetsolnpoolfiltername
    return _getname(env, lp, which, namefn, index_first=False)


def getsolnpoolnumfilters(env, lp):
    return CR.CPXXgetsolnpoolnumfilters(env, lp)


def fltwrite(env, lp, filename):
    status = CR.CPXXfltwrite(env, lp, filename)
    check_status(env, status)


def readcopysolnpoolfilters(env, lp, filename):
    status = CR.CPXXreadcopysolnpoolfilters(env, lp,
                                            filename)
    check_status(env, status)


def getsolnpoolfilterindex(env, lp, colname):
    index = CR.intPtr()
    status = CR.CPXXgetsolnpoolfilterindex(env, lp, colname, index)
    check_status(env, status)
    return index.value()


def getsolnpoolfiltertype(env, lp, index):
    type_ = CR.intPtr()
    status = CR.CPXXgetsolnpoolfiltertype(env, lp, type_, index)
    check_status(env, status)
    return type_.value()


def delsolnpoolsolns(env, lp, begin, end):
    delfn = CR.CPXXdelsolnpoolsolns
    _delbyrange(delfn, env, lp, begin, end)


def getsolnpoolnumsolns(env, lp):
    return CR.CPXXgetsolnpoolnumsolns(env, lp)


def getsolnpoolnumreplaced(env, lp):
    return CR.CPXXgetsolnpoolnumreplaced(env, lp)


def getsolnpoolsolnindex(env, lp, colname):
    index = CR.intPtr()
    status = CR.CPXXgetsolnpoolsolnindex(env, lp, colname, index)
    check_status(env, status)
    return index.value()


def getsolnpoolmeanobjval(env, lp):
    objval = CR.doublePtr()
    status = CR.CPXXgetsolnpoolmeanobjval(env, lp, objval)
    check_status(env, status)
    return objval.value()


def getsolnpoolsolnname(env, lp, which):
    namefn = CR.CPXXgetsolnpoolsolnname
    return _getname(env, lp, which, namefn, index_first=False)


def solwritesolnpool(env, lp, soln, filename):
    status = CR.CPXXsolwritesolnpool(env, lp, soln, filename)
    check_status(env, status)


def solwritesolnpoolall(env, lp, filename):
    status = CR.CPXXsolwritesolnpoolall(env, lp, filename)
    check_status(env, status)


def getsolnpoolobjval(env, lp, soln):
    obj = CR.doublePtr()
    status = CR.CPXXgetsolnpoolobjval(env, lp, soln, obj)
    check_status(env, status)
    return obj.value()


def getsolnpoolx(env, lp, soln, begin, end):
    xlen = _rangelen(begin, end)
    x = _safeDoubleArray(xlen)
    status = CR.CPXXgetsolnpoolx(env, lp, soln, x, begin, end)
    check_status(env, status)
    return LAU.array_to_list(x, xlen)


def getsolnpoolslack(env, lp, soln, begin, end):
    slacklen = _rangelen(begin, end)
    slack = _safeDoubleArray(slacklen)
    status = CR.CPXXgetsolnpoolslack(env, lp, soln, slack, begin, end)
    check_status(env, status)
    return LAU.array_to_list(slack, slacklen)


def getsolnpoolqconstrslack(env, lp, soln, begin, end):
    qlen = _rangelen(begin, end)
    q = _safeDoubleArray(qlen)
    status = CR.CPXXgetsolnpoolqconstrslack(env, lp, soln, q, begin, end)
    check_status(env, status)
    return LAU.array_to_list(q, qlen)


def getsolnpoolintquality(env, lp, soln, what):
    quality = CR.intPtr()
    status = CR.CPXXgetsolnpoolintquality(env, lp, soln, quality, what)
    check_status(env, status)
    return quality.value()


def getsolnpooldblquality(env, lp, soln, what):
    quality = CR.doublePtr()
    status = CR.CPXXgetsolnpooldblquality(env, lp, soln, quality, what)
    check_status(env, status)
    return quality.value()


# Initial data


def copystart(env, lp, cstat, rstat, cprim, rprim, cdual, rdual):
    status = CR.CPXXcopystart(env, lp,
                              LAU.int_list_to_array(cstat),
                              LAU.int_list_to_array(rstat),
                              LAU.double_list_to_array(cprim),
                              LAU.double_list_to_array(rprim),
                              LAU.double_list_to_array(cdual),
                              LAU.double_list_to_array(rdual))
    check_status(env, status)


def readcopybase(env, lp, filename):
    status = CR.CPXXreadcopybase(env, lp, filename)
    check_status(env, status)


def getorder(env, lp):
    count = CR.intPtr()
    surplus = CR.intPtr()
    space = 0
    ind = LAU.int_list_to_array([])
    pri = LAU.int_list_to_array([])
    dir_ = LAU.int_list_to_array([])
    status = CR.CPXXgetorder(env, lp, count, ind, pri, dir_, space, surplus)
    if status != CR.CPXERR_NEGATIVE_SURPLUS:
        check_status(env, status)
    space = -surplus.value()
    ind = _safeIntArray(space)
    pri = _safeIntArray(space)
    dir_ = _safeIntArray(space)
    status = CR.CPXXgetorder(env, lp, count, ind, pri, dir_, space, surplus)
    check_status(env, status)
    return (LAU.array_to_list(ind, space), LAU.array_to_list(pri, space),
            LAU.array_to_list(dir_, space))


def copyorder(env, lp, indices, priority, direction):
    status = CR.CPXXcopyorder(env, lp, len(indices),
                              LAU.int_list_to_array(indices),
                              LAU.int_list_to_array(priority),
                              LAU.int_list_to_array(direction))
    check_status(env, status)


def readcopyorder(env, lp, filename):
    status = CR.CPXXreadcopyorder(env, lp, filename)
    check_status(env, status)


def ordwrite(env, lp, filename):
    status = CR.CPXXordwrite(env, lp, filename)
    check_status(env, status)


def readcopystartinfo(env, lp, filename):
    status = CR.CPXXreadcopystartinfo(env, lp, filename)
    check_status(env, status)

# handle the lock for parallel callbacks


def initlock():
    return CR.init_callback_lock()


def finitlock(lock):
    CR.finit_callback_lock(lock)


# get problem statistics

def getprobstats(env, lp):
    ProbStats = namedtuple(
        'ProbStats',
        ['objs',  # 0
         'rows',  # 1
         'cols',  # 2
         'objcnt',  # 3
         'rhscnt',  # 4
         'nzcnt',  # 5
         'ecnt',  # 6
         'gcnt',  # 7
         'lcnt',  # 8
         'rngcnt',  # 9
         'ncnt',  # 10
         'fcnt',  # 11
         'xcnt',  # 12
         'bcnt',  # 13
         'ocnt',  # 14
         'bicnt',  # 15
         'icnt',  # 16
         'scnt',  # 17
         'sicnt',  # 18
         'qpcnt',  # 19
         'qpnzcnt',  # 20
         'nqconstr',  # 21
         'qrhscnt',  # 22
         'qlcnt',  # 23
         'qgcnt',  # 24
         'quadnzcnt',  # 25
         'linnzcnt',  # 26
         'nindconstr',  # 27
         'indrhscnt',  # 28
         'indnzcnt',  # 29
         'indcompcnt',  # 30
         'indlcnt',  # 31
         'indecnt',  # 32
         'indgcnt',  # 33
         'maxcoef',  # 34
         'mincoef',  # 35
         'minrhs',  # 36
         'maxrhs',  # 37
         'minrng',  # 38
         'maxrng',  # 39
         'minobj',  # 40
         'maxobj',  # 41
         'minlb',  # 42
         'maxub',  # 43
         'minqcoef',  # 44
         'maxqcoef',  # 45
         'minqcq',  # 46
         'maxqcq',  # 47
         'minqcl',  # 48
         'maxqcl',  # 49
         'minqcr',  # 50
         'maxqcr',  # 51
         'minind',  # 52
         'maxind',  # 53
         'minindrhs',  # 54
         'maxindrhs',  # 55
         'minlazy',  # 56
         'maxlazy',  # 57
         'minlazyrhs',  # 58
         'maxlazyrhs',  # 59
         'minucut',  # 60
         'maxucut',  # 61
         'minucutrhs',  # 62
         'maxucutrhs',  # 63
         'nsos',  # 64
         'nsos1',  # 65
         'sos1nmem',  # 66
         'sos1type',  # 67
         'nsos2',  # 68
         'sos2nmem',  # 69
         'sos2type',  # 70
         'lazyrhscnt',  # 71
         'lazygcnt',  # 72
         'lazylcnt',  # 73
         'lazyecnt',  # 74
         'lazycnt',  # 75
         'lazynzcnt',  # 76
         'ucutrhscnt',  # 77
         'ucutgcnt',  # 78
         'ucutlcnt',  # 79
         'ucutecnt',  # 80
         'ucutcnt',  # 81
         'ucutnzcnt',  # 82
         'npwl',  # 83
         'npwlbreaks'])   # 84

    objs_p = CR.intPtr()
    rows_p = CR.intPtr()
    cols_p = CR.intPtr()
    objcnt_p = CR.intPtr()
    rhscnt_p = CR.intPtr()
    nzcnt_p = CR.intPtr()
    ecnt_p = CR.intPtr()
    gcnt_p = CR.intPtr()
    lcnt_p = CR.intPtr()
    rngcnt_p = CR.intPtr()
    ncnt_p = CR.intPtr()
    fcnt_p = CR.intPtr()
    xcnt_p = CR.intPtr()
    bcnt_p = CR.intPtr()
    ocnt_p = CR.intPtr()
    bicnt_p = CR.intPtr()
    icnt_p = CR.intPtr()
    scnt_p = CR.intPtr()
    sicnt_p = CR.intPtr()
    qpcnt_p = CR.intPtr()
    qpnzcnt_p = CR.intPtr()
    nqconstr_p = CR.intPtr()
    qrhscnt_p = CR.intPtr()
    qlcnt_p = CR.intPtr()
    qgcnt_p = CR.intPtr()
    quadnzcnt_p = CR.intPtr()
    linnzcnt_p = CR.intPtr()
    nindconstr_p = CR.intPtr()
    indrhscnt_p = CR.intPtr()
    indnzcnt_p = CR.intPtr()
    indcompcnt_p = CR.intPtr()
    indlcnt_p = CR.intPtr()
    indecnt_p = CR.intPtr()
    indgcnt_p = CR.intPtr()
    maxcoef_p = CR.doublePtr()
    mincoef_p = CR.doublePtr()
    minrhs_p = CR.doublePtr()
    maxrhs_p = CR.doublePtr()
    minrng_p = CR.doublePtr()
    maxrng_p = CR.doublePtr()
    minobj_p = CR.doublePtr()
    maxobj_p = CR.doublePtr()
    minlb_p = CR.doublePtr()
    maxub_p = CR.doublePtr()
    minqcoef_p = CR.doublePtr()
    maxqcoef_p = CR.doublePtr()
    minqcq_p = CR.doublePtr()
    maxqcq_p = CR.doublePtr()
    minqcl_p = CR.doublePtr()
    maxqcl_p = CR.doublePtr()
    minqcr_p = CR.doublePtr()
    maxqcr_p = CR.doublePtr()
    minind_p = CR.doublePtr()
    maxind_p = CR.doublePtr()
    minindrhs_p = CR.doublePtr()
    maxindrhs_p = CR.doublePtr()
    minlazy_p = CR.doublePtr()
    maxlazy_p = CR.doublePtr()
    minlazyrhs_p = CR.doublePtr()
    maxlazyrhs_p = CR.doublePtr()
    minucut_p = CR.doublePtr()
    maxucut_p = CR.doublePtr()
    minucutrhs_p = CR.doublePtr()
    maxucutrhs_p = CR.doublePtr()
    nsos_p = CR.intPtr()
    nsos1_p = CR.intPtr()
    sos1nmem_p = CR.intPtr()
    sos1type_p = CR.intPtr()
    nsos2_p = CR.intPtr()
    sos2nmem_p = CR.intPtr()
    sos2type_p = CR.intPtr()
    lazyrhscnt_p = CR.intPtr()
    lazygcnt_p = CR.intPtr()
    lazylcnt_p = CR.intPtr()
    lazyecnt_p = CR.intPtr()
    lazycnt_p = CR.intPtr()
    lazynzcnt_p = CR.intPtr()
    ucutrhscnt_p = CR.intPtr()
    ucutgcnt_p = CR.intPtr()
    ucutlcnt_p = CR.intPtr()
    ucutecnt_p = CR.intPtr()
    ucutcnt_p = CR.intPtr()
    ucutnzcnt_p = CR.intPtr()
    npwl_p = CR.intPtr()
    npwlbreaks_p = CR.intPtr()
    status = CR.CPXEgetprobstats(env, lp,
                                 objs_p,
                                 rows_p,
                                 cols_p,
                                 objcnt_p,
                                 rhscnt_p,
                                 nzcnt_p,
                                 ecnt_p,
                                 gcnt_p,
                                 lcnt_p,
                                 rngcnt_p,
                                 ncnt_p,
                                 fcnt_p,
                                 xcnt_p,
                                 bcnt_p,
                                 ocnt_p,
                                 bicnt_p,
                                 icnt_p,
                                 scnt_p,
                                 sicnt_p,
                                 qpcnt_p,
                                 qpnzcnt_p,
                                 nqconstr_p,
                                 qrhscnt_p,
                                 qlcnt_p,
                                 qgcnt_p,
                                 quadnzcnt_p,
                                 linnzcnt_p,
                                 nindconstr_p,
                                 indrhscnt_p,
                                 indnzcnt_p,
                                 indcompcnt_p,
                                 indlcnt_p,
                                 indecnt_p,
                                 indgcnt_p,
                                 maxcoef_p,
                                 mincoef_p,
                                 minrhs_p,
                                 maxrhs_p,
                                 minrng_p,
                                 maxrng_p,
                                 minobj_p,
                                 maxobj_p,
                                 minlb_p,
                                 maxub_p,
                                 minqcoef_p,
                                 maxqcoef_p,
                                 minqcq_p,
                                 maxqcq_p,
                                 minqcl_p,
                                 maxqcl_p,
                                 minqcr_p,
                                 maxqcr_p,
                                 minind_p,
                                 maxind_p,
                                 minindrhs_p,
                                 maxindrhs_p,
                                 minlazy_p,
                                 maxlazy_p,
                                 minlazyrhs_p,
                                 maxlazyrhs_p,
                                 minucut_p,
                                 maxucut_p,
                                 minucutrhs_p,
                                 maxucutrhs_p,
                                 nsos_p,
                                 nsos1_p,
                                 sos1nmem_p,
                                 sos1type_p,
                                 nsos2_p,
                                 sos2nmem_p,
                                 sos2type_p,
                                 lazyrhscnt_p,
                                 lazygcnt_p,
                                 lazylcnt_p,
                                 lazyecnt_p,
                                 lazycnt_p,
                                 lazynzcnt_p,
                                 ucutrhscnt_p,
                                 ucutgcnt_p,
                                 ucutlcnt_p,
                                 ucutecnt_p,
                                 ucutcnt_p,
                                 ucutnzcnt_p,
                                 npwl_p,
                                 npwlbreaks_p)
    check_status(env, status)
    return ProbStats(
        objs_p.value(),
        rows_p.value(),
        cols_p.value(),
        objcnt_p.value(),
        rhscnt_p.value(),
        nzcnt_p.value(),
        ecnt_p.value(),
        gcnt_p.value(),
        lcnt_p.value(),
        rngcnt_p.value(),
        ncnt_p.value(),
        fcnt_p.value(),
        xcnt_p.value(),
        bcnt_p.value(),
        ocnt_p.value(),
        bicnt_p.value(),
        icnt_p.value(),
        scnt_p.value(),
        sicnt_p.value(),
        qpcnt_p.value(),
        qpnzcnt_p.value(),
        nqconstr_p.value(),
        qrhscnt_p.value(),
        qlcnt_p.value(),
        qgcnt_p.value(),
        quadnzcnt_p.value(),
        linnzcnt_p.value(),
        nindconstr_p.value(),
        indrhscnt_p.value(),
        indnzcnt_p.value(),
        indcompcnt_p.value(),
        indlcnt_p.value(),
        indecnt_p.value(),
        indgcnt_p.value(),
        maxcoef_p.value(),
        mincoef_p.value(),
        minrhs_p.value(),
        maxrhs_p.value(),
        minrng_p.value(),
        maxrng_p.value(),
        minobj_p.value(),
        maxobj_p.value(),
        minlb_p.value(),
        maxub_p.value(),
        minqcoef_p.value(),
        maxqcoef_p.value(),
        minqcq_p.value(),
        maxqcq_p.value(),
        minqcl_p.value(),
        maxqcl_p.value(),
        minqcr_p.value(),
        maxqcr_p.value(),
        minind_p.value(),
        maxind_p.value(),
        minindrhs_p.value(),
        maxindrhs_p.value(),
        minlazy_p.value(),
        maxlazy_p.value(),
        minlazyrhs_p.value(),
        maxlazyrhs_p.value(),
        minucut_p.value(),
        maxucut_p.value(),
        minucutrhs_p.value(),
        maxucutrhs_p.value(),
        nsos_p.value(),
        nsos1_p.value(),
        sos1nmem_p.value(),
        sos1type_p.value(),
        nsos2_p.value(),
        sos2nmem_p.value(),
        sos2type_p.value(),
        lazyrhscnt_p.value(),
        lazygcnt_p.value(),
        lazylcnt_p.value(),
        lazyecnt_p.value(),
        lazycnt_p.value(),
        lazynzcnt_p.value(),
        ucutrhscnt_p.value(),
        ucutgcnt_p.value(),
        ucutlcnt_p.value(),
        ucutecnt_p.value(),
        ucutcnt_p.value(),
        ucutnzcnt_p.value(),
        npwl_p.value(),
        npwlbreaks_p.value())

# get histogram of non-zero row/column counts


def gethist(env, lp, key):
    if key == 'r':
        space = CR.CPXXgetnumcols(env, lp) + 1
    else:
        key = 'c'
        space = CR.CPXXgetnumrows(env, lp) + 1
    hist = _safeIntArray(space)
    status = CR.CPXEgethist(env, lp, key, hist)
    check_status(env, status)
    return LAU.array_to_list(hist, space)

# get solution quality metrics


def getqualitymetrics(env, lp, soln):
    space = 26
    data = _safeDoubleArray(space)
    ispace = 10
    idata = _safeIntArray(ispace)
    status = CR.CPXEgetqualitymetrics(env, lp, soln, data, idata)
    check_status(env, status)
    return [LAU.array_to_list(idata, ispace),
            LAU.array_to_list(data, space)]

def showquality(env, lp, soln):
    status = CR.CPXEshowquality(env, lp, soln)
    check_status(env, status)

# ########## Expert Callback BEGIN ########################################


def setgenericcallbackfunc(env, lp, contextmask, cbhandle):
    # NOTE: The cbhandle that is passed in here, is the Cplex instance that
    #       installs the callback. We do not increment the reference count
    #       for this on purpose: First of all, it is not necessary since the
    #       lifetime of env/lp is controled by the lifetime of this instance.
    #       Hence any reference the callable library stores is valid as long
    #       as it may be used.
    #       Second, in the destructor of the Cplex class we attempt to set
    #       the callback to (0, None). This may fail with
    #       CPXERR_NOT_ONE_PROBLEM. If we had incremented the reference count
    #       we would have to figure out whether the attempt to unset got as
    #       far as decrementing the reference count or failed earlier.
    status = CR.CPXXcallbacksetfunc(env, lp, contextmask, cbhandle)
    check_status(env, status)


def callbackgetinfoint(contextptr, which):
    data = CR.intPtr()
    status = CR.CPXXcallbackgetinfoint(contextptr, [which, data])
    check_status(None, status)
    return data.value()


def callbackgetinfolong(contextptr, which):
    data = CR.cpxlongPtr()
    status = CR.CPXXcallbackgetinfolong(contextptr, [which, data])
    check_status(None, status)
    return data.value()


def callbackgetinfodbl(contextptr, which):
    data = CR.doublePtr()
    status = CR.CPXXcallbackgetinfodbl(contextptr, [which, data])
    check_status(None, status)
    return data.value()


def callbackabort(contextptr):
    CR.CPXXcallbackabort(contextptr)

def callbackcandidateispoint(contextptr):
    bounded = CR.intPtr()
    status = CR.CPXXcallbackcandidateispoint(contextptr, bounded)
    check_status(None, status)
    return bounded.value() != 0

def callbackgetcandidatepoint(contextptr, begin, end):
    xlen = _rangelen(begin, end)
    x = _safeDoubleArray(xlen)
    status = CR.CPXXcallbackgetcandidatepoint(contextptr, x, begin, end, None)
    check_status(None, status)
    return LAU.array_to_list(x, xlen)

def callbackcandidateisray(contextptr):
    ray = CR.intPtr()
    status = CR.CPXXcallbackcandidateisray(contextptr, ray)
    check_status(None, status)
    return ray.value() != 0

def callbackgetcandidateray(contextptr, begin, end):
    raylen = _rangelen(begin, end)
    ray = _safeDoubleArray(raylen)
    status = CR.CPXXcallbackgetcandidateray(contextptr, ray, begin, end)
    check_status(None, status)
    return LAU.array_to_list(ray, raylen)

def callbackgetcandidateobj(contextptr):
    obj_p = CR.doublePtr()
    status = CR.CPXXcallbackgetcandidatepoint(contextptr, None, 0, -1, obj_p)
    check_status(None, status)
    return obj_p.value()


def callbackgetrelaxationpoint(contextptr, begin, end):
    xlen = _rangelen(begin, end)
    x = _safeDoubleArray(xlen)
    status = CR.CPXXcallbackgetrelaxationpoint(contextptr, x, begin, end, None)
    check_status(None, status)
    return LAU.array_to_list(x, xlen)

def callbackgetrelaxationpointobj(contextptr):
    obj_p = CR.doublePtr()
    status = CR.CPXXcallbackgetrelaxationpoint(contextptr, None, 0, -1, obj_p)
    check_status(None, status)
    return obj_p.value()

def callbackgetrelaxationstatus(contextptr, flags):
    nodelpstat = CR.intPtr()
    status = CR.CPXXcallbackgetrelaxationstatus(contextptr, nodelpstat, flags)
    check_status(None, status)
    return nodelpstat.value()

def callbackmakebranch(contextptr, ind, lu, bd, rhs, sense, matbeg,
                       matind, matval, nodeest):
    seqnum = CR.cpxlongPtr()
    with LAU.int_c_array(ind) as c_ind,\
            LAU.double_c_array(bd) as c_bd, \
            LAU.double_c_array(rhs) as c_rhs, \
            LAU.long_c_array(matbeg) as c_matbeg, \
            LAU.int_c_array(matind) as c_matind, \
            LAU.double_c_array(matval) as c_matval:
        status = CR.CPXXcallbackmakebranch(contextptr, len(ind),
                                           c_ind, lu, c_bd,
                                           len(matbeg), len(matind),
                                           c_rhs, sense,
                                           c_matbeg, c_matind, c_matval,
                                           nodeest, seqnum)
    check_status(None, status)
    return seqnum.value()

def callbackprunenode(contextptr):
    status = CR.CPXXcallbackprunenode(contextptr)
    check_status(None, status)

def callbackexitcutloop(contextptr):
    status = CR.CPXXcallbackexitcutloop(contextptr)
    check_status(None, status)

def callbackgetincumbent(contextptr, begin, end):
    xlen = _rangelen(begin, end)
    x = _safeDoubleArray(xlen)
    status = CR.CPXXcallbackgetincumbent(contextptr, x, begin, end, None)
    check_status(None, status)
    return LAU.array_to_list(x, xlen)


def callbackgetincumbentobj(contextptr):
    obj_p = CR.doublePtr()
    status = CR.CPXXcallbackgetincumbent(contextptr, None, 0, -1, obj_p)
    check_status(None, status)
    return obj_p.value()


def callbackgetlocallb(contextptr, begin, end):
    xlen = _rangelen(begin, end)
    x = _safeDoubleArray(xlen)
    status = CR.CPXXcallbackgetlocallb(contextptr, x, begin, end)
    check_status(None, status)
    return LAU.array_to_list(x, xlen)


def callbackgetlocalub(contextptr, begin, end):
    xlen = _rangelen(begin, end)
    x = _safeDoubleArray(xlen)
    status = CR.CPXXcallbackgetlocalub(contextptr, x, begin, end)
    check_status(None, status)
    return LAU.array_to_list(x, xlen)

def callbackgetgloballb(contextptr, begin, end):
    xlen = _rangelen(begin, end)
    x = _safeDoubleArray(xlen)
    status = CR.CPXXcallbackgetgloballb(contextptr, x, begin, end)
    check_status(None, status)
    return LAU.array_to_list(x, xlen)


def callbackgetglobalub(contextptr, begin, end):
    xlen = _rangelen(begin, end)
    x = _safeDoubleArray(xlen)
    status = CR.CPXXcallbackgetglobalub(contextptr, x, begin, end)
    check_status(None, status)
    return LAU.array_to_list(x, xlen)

def callbackpostheursoln(contextptr, cnt, ind, val, obj, strategy):
    status = CR.CPXXcallbackpostheursoln(contextptr, cnt,
                                         LAU.int_list_to_array(ind),
                                         LAU.double_list_to_array(val),
                                         obj, strategy)
    check_status(None, status)


def callbackaddusercuts(contextptr, rcnt, nzcnt, rhs, sense, rmat,
                        cutmanagement, local):
    with LAU.double_c_array(rhs) as c_rhs, \
            LAU.int_c_array(cutmanagement) as c_cutmanagement, \
            LAU.int_c_array(local) as c_local:
        status = CR.CPXXcallbackaddusercuts(contextptr, rcnt, nzcnt, c_rhs,
                                            sense, rmat,
                                            c_cutmanagement, c_local)
    check_status(None, status)


def callbackrejectcandidate(contextptr, rcnt, nzcnt, rhs, sense, rmat):
    with LAU.double_c_array(rhs) as c_rhs:
        status = CR.CPXXcallbackrejectcandidate(contextptr, rcnt, nzcnt, c_rhs,
                                                sense, rmat)
    check_status(None, status)

def callbackrejectcandidatelocal(contextptr, rcnt, nzcnt, rhs, sense, rmat):
    with LAU.double_c_array(rhs) as c_rhs:
        status = CR.CPXXcallbackrejectcandidatelocal(contextptr, rcnt, nzcnt,
                                                     c_rhs, sense, rmat)
    check_status(None, status)

# ########## Expert Callback END ##########################################

# ########## Modeling Assistance Callback BEGIN ###########################

def modelasstcallbacksetfunc(env, lp, cbhandle):
    # See note in setgenericcallbackfunc (the same applies here).
    status = CR.CPXXmodelasstcallbacksetfunc(env, lp, cbhandle)
    check_status(env, status)

# ########## Modeling Assistance Callback END #############################
