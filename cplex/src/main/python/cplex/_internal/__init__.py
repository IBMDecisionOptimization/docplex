# --------------------------------------------------------------------------
# File: __init__.py
# ---------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2008, 2026. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# ------------------------------------------------------------------------
"""

   :undocumented: Environment, _aux_functions, _list_array_utils, _ostream

"""
import sys

# Priority order for native backends: lower value = higher priority.
# If both cplex_ce and cplex_pe are installed, full wins.
_BACKEND_PRIORITY = {"full": 0, "ce": 1}

# Load _pycplex FIRST before any modules that depend on it
def _load_native_backend():
    """Discover and load the highest-priority installed native backend."""
    import importlib.metadata

    eps = sorted(
        importlib.metadata.entry_points(group="cplex.native"),
        key=lambda ep: _BACKEND_PRIORITY.get(ep.name, 99),
    )
    if not eps:
        raise RuntimeError(
            "No CPLEX native backend installed. "
            "Install either 'cplex_ce' or 'cplex_pe'."
        )
    return eps[0].load()

_pycplex = _load_native_backend()

# Now import modules that depend on _pycplex
from . import _aux_functions
from . import _baseinterface
from . import _list_array_utils
from . import _ostream
from . import _procedural
from . import _constants
from . import _matrices
from . import _multiobj
from . import _multiobjsoln
from . import _parameter_classes
from . import _parameter_hierarchy
from . import _subinterfaces
from . import _parameters_auto
from . import _anno
from . import _pwl
from . import _constantsenum
from . import _callbackinfoenum
from . import _solutionstrategyenum
from ..exceptions import CplexError
from ..constant_class import ConstantClass

__all__ = ["Environment", "_aux_functions", "_baseinterface",
           "_list_array_utils", "_ostream", "_procedural",
           "_constants", "_matrices", "_multiobj", "_multiobjsoln",
           "_parameter_classes", "_subinterfaces", "_pycplex",
           "_parameters_auto", "_anno", "_pwl", "ProblemType",
           "_constantsenum", "_constants", "_callbackinfoenum",
           "_solutionstrategyenum"]


class ProblemType(ConstantClass):
    """Types of problems the Cplex object can encapsulate.

    For explanations of the problem types, see those topics in the
    CPLEX User's Manual in the topic titled Continuous Optimization
    for LP, QP, and QCP or the topic titled Discrete Optimization
    for MILP, FIXEDMILP, NODELP, NODEQP, MIQCP, NODEQCP.
    """
    LP = _constants.CPXPROB_LP
    """See CPXPROB_LP in the C API."""

    MILP = _constants.CPXPROB_MILP
    """See CPXPROB_MILP in the C API."""

    fixed_MILP = _constants.CPXPROB_FIXEDMILP
    """See CPXPROB_FIXEDMILP in the C API."""

    node_LP = _constants.CPXPROB_NODELP
    """See CPXPROB_NODELP in the C API."""

    QP = _constants.CPXPROB_QP
    """See CPXPROB_QP in the C API."""

    MIQP = _constants.CPXPROB_MIQP
    """See CPXPROB_MIQP in the C API."""

    fixed_MIQP = _constants.CPXPROB_FIXEDMIQP
    """See CPXPROB_MIQP in the C API."""

    node_QP = _constants.CPXPROB_NODEQP
    """See CPXPROB_NODEQP in the C API."""

    QCP = _constants.CPXPROB_QCP
    """See CPXPROB_QCP in the C API."""

    MIQCP = _constants.CPXPROB_MIQCP
    """See CPXPROB_MIQCP in the C API."""

    node_QCP = _constants.CPXPROB_NODEQCP
    """See CPXPROB_QCP in the C API."""


def _needs_delete_callback(callback_instance):
    # If the user has registered any callback that may change
    # the user data at a node then we need to register the
    # delete callback.
    # The Control, Node, and Incumbent callbacks have the set_node_data
    # method (and all who inherit from these).
    return hasattr(callback_instance, "set_node_data")


def _getcbattrname(cb_type_string):
    """Returns the attribute name to be used to store the callback
    instance.
    """
    return "_{0}_callback".format(cb_type_string)


def _checkcbcls(obj):
    """Checks callback class instance for expected attribute.

    Raises a CplexError if it is not found.
    """
    if getattr(obj, "_cb_type_string", None) is None:
        raise CplexError(str(obj) +
                         " is not a subclass of a subclassable Callback class.")


class Environment():
    """non-public"""
    RESULTS_CHNL_IDX = 0
    WARNING_CHNL_IDX = 1
    ERROR_CHNL_IDX = 2
    LOG_CHNL_IDX = 3

    def __init__(self):
        """non-public"""
        # Declare and initialize attributes
        self._e = None
        self._lock = None
        self._streams = {self.RESULTS_CHNL_IDX: None,
                         self.WARNING_CHNL_IDX: None,
                         self.ERROR_CHNL_IDX: None,
                         self.LOG_CHNL_IDX: None}
        self._callback_exception = None
        self._callbacks = []
        self._disposed = False
        # Initialize data strucutures associated with CPLEX
        self._e = _procedural.openCPLEX()
        self.parameters = _parameter_classes.RootParameterGroup(
            self, _parameter_hierarchy.root_members)
        _procedural.set_status_checker()
        self._lock = _procedural.initlock()
        self.set_results_stream(sys.stdout)
        self.set_warning_stream(sys.stderr)
        self.set_error_stream(sys.stderr)
        self.set_log_stream(sys.stdout)

    def _end(self):
        """Frees all of the data structures associated with CPLEX."""
        if self._disposed:
            return
        self._disposed = True
        for chnl_idx in self._streams:
            self._delete_stream(chnl_idx)
        if self._lock and self._e:
            _procedural.finitlock(self._lock)
        if self._e:
            _procedural.closeCPLEX(self._e)
            self._e = None

    def __del__(self):
        """non-public"""
        self._end()

    def _get_num_delete(self):
        """Count the callbacks that are installed and require a delete
        callback.
        """
        return sum(1 for c in self._callbacks
                   if _needs_delete_callback(c))

    def register_callback(self, callback_class):
        """Registers a callback for use when solving.

        callback_class must be a proper subclass of one of the
        callback classes defined in the module callbacks.  It must
        override the __call__ method with a method that has signature
        __call__(self) -> None.  If callback_class is a subclass of
        more than one callback class, it will only be called when its
        first superclass is called.  register_callback returns the
        instance of callback_class registered for use.  Any previously
        registered callback of the same class will no longer be
        registered.
        """
        cb = callback_class(self)
        _checkcbcls(cb)
        num_delete = self._get_num_delete()
        old_cb = getattr(
            self, _getcbattrname(cb._cb_type_string), None)
        if old_cb:
            self._callbacks.remove(old_cb)
        setattr(self, _getcbattrname(cb._cb_type_string), cb)
        if cb._cb_type_string == "MIP_info":
            # self._MIP_info_callback is set above with the call to
            # setattr. So, we are passing the callback instance as the
            # second argument here rather than the environment
            # (i.e., self).
            # pylint: disable=no-member
            cb._cb_set_function(self._e, self._MIP_info_callback)
        else:
            cb._cb_set_function(self._e, self)
        self._callbacks.append(cb)
        if _needs_delete_callback(cb) and num_delete < 1:
            # We need a delete callback and did not have one
            # before -> install it.
            _procedural.setpydel(self._e)
        return cb

    def unregister_callback(self, callback_class):
        """Unregisters a callback.

        callback_class must be one of the callback classes defined in
        the module callback or a subclass of one of them.  This method
        unregisters any previously registered callback of the same
        class.  If callback_class is a subclass of more than one
        callback class, this method unregisters only the callback of the
        same type as its first superclass.  unregister_callback
        returns the instance of callback_class just unregistered.

        """
        cb = callback_class(self)
        _checkcbcls(cb)
        current_cb = getattr(
            self, _getcbattrname(cb._cb_type_string), None)
        if current_cb:
            if (_needs_delete_callback(current_cb) and
                    self._get_num_delete() < 2):
                # We are about to remove the last callback that requires
                # a delete callback.
                _procedural.delpydel(self._e)
            current_cb._cb_set_function(self._e, None)
            self._callbacks.remove(current_cb)
            delattr(self, _getcbattrname(current_cb._cb_type_string))
        return current_cb

    def _add_stream(self, which_channel):
        """non-public"""
        channel = _procedural.getchannels(self._e)[which_channel]
        _procedural.addfuncdest(self._e, channel,
                                self._streams[which_channel])

    def _delete_stream(self, which_channel):
        """non-public"""
        if self._streams[which_channel] is None:
            return
        channel = _procedural.getchannels(self._e)[which_channel]
        _procedural.delfuncdest(self._e, channel,
                                self._streams[which_channel])
        self._streams[which_channel]._end()

    def _set_stream(self, which, outputfile, func=None, initerrstr=False):
        self._delete_stream(which)
        self._streams[which] = _ostream.OutputStream(
            outputfile, self, fn=func, initerrorstr=initerrstr)
        self._add_stream(which)
        return self._streams[which]

    def set_results_stream(self, results_file, fn=None):
        """Specifies where results will be printed.

        The first argument must be either a file-like object (that is, an
        object with a write method and a flush method) or the name of
        a file to be written to (the later is deprecated since V12.9.0).
        Use None as the first argument to suppress output.

        The second optional argument is a function that takes a string
        as input and returns a string.  If specified, strings sent to
        this stream will be processed by this function before being
        written.

        Returns the stream to which results will be written.  To write
        to this stream, use the write() method of this object.
        """
        return self._set_stream(which=self.RESULTS_CHNL_IDX,
                                outputfile=results_file,
                                func=fn,
                                initerrstr=False)

    def set_warning_stream(self, warning_file, fn=None):
        """Specifies where warnings will be printed.

        The first argument must be either a file-like object (that is, an
        object with a write method and a flush method) or the name of
        a file to be written to (the later is deprecated since V12.9.0).
        Use None as the first argument to suppress output.

        The second optional argument is a function that takes a string
        as input and returns a string.  If specified, strings sent to
        this stream will be processed by this function before being
        written.

        Returns the stream to which warnings will be written.  To write
        to this stream, use the write() method of this object.
        """
        return self._set_stream(which=self.WARNING_CHNL_IDX,
                                outputfile=warning_file,
                                func=fn,
                                initerrstr=False)

    def set_error_stream(self, error_file, fn=None):
        """Specifies where errors will be printed.

        The first argument must be either a file-like object (that is, an
        object with a write method and a flush method) or the name of
        a file to be written to (the later is deprecated since V12.9.0).
        Use None as the first argument to suppress output.

        The second optional argument is a function that takes a string
        as input and returns a string.  If specified, strings sent to
        this stream will be processed by this function before being
        written.

        Returns the stream to which errors will be written.  To write
        to this stream, use the write() method of this object.
        """
        return self._set_stream(which=self.ERROR_CHNL_IDX,
                                outputfile=error_file,
                                func=fn,
                                initerrstr=True)

    def set_log_stream(self, log_file, fn=None):
        """Specifies where the log will be printed.

        The first argument must be either a file-like object (that is, an
        object with a write method and a flush method) or the name of
        a file to be written to (the later is deprecated since V12.9.0).
        Use None as the first argument to suppress output.

        The second optional argument is a function that takes a string
        as input and returns a string.  If specified, strings sent to
        this stream will be processed by this function before being
        written.

        Returns the stream to which the log will be written.  To write
        to this stream, use this object's write() method.
        """
        return self._set_stream(which=self.LOG_CHNL_IDX,
                                outputfile=log_file,
                                func=fn,
                                initerrstr=False)

    def _get_results_stream(self):
        """non-public.  Nice for unit tests."""
        return self._streams[self.RESULTS_CHNL_IDX]

    def _get_warning_stream(self):
        """non-public.  Nice for unit tests."""
        return self._streams[self.WARNING_CHNL_IDX]

    def _get_error_stream(self):
        """non-public.  Nice for unit tests."""
        return self._streams[self.ERROR_CHNL_IDX]

    def _get_log_stream(self):
        """non-public.  Nice for unit tests."""
        return self._streams[self.LOG_CHNL_IDX]

    def get_version(self):
        """Returns a string specifying the version of CPLEX."""
        return _procedural.version(self._e)

    def get_versionnumber(self):
        """Returns an integer specifying the version of CPLEX.

        The version of CPLEX is in the format vvrrmmff, where vv is
        the version, rr is the release, mm is the modification, and ff
        is the fixpack number. For example, for CPLEX version 12.5.0.1
        the returned value is 12050001.
        """
        return _procedural.versionnumber(self._e)

    def get_num_cores(self):
        """Returns the number of cores on this machine."""
        return _procedural.getnumcores(self._e)

    def get_time(self):
        """Returns a timestamp in CPU or wallclock seconds from CPLEX."""
        return _procedural.gettime(self._e)

    def get_dettime(self):
        """Returns the current deterministic time in ticks."""
        return _procedural.getdettime(self._e)
