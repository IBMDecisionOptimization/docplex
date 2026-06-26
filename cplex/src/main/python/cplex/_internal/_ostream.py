# --------------------------------------------------------------------------
# File: _ostream.py
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
"""
import weakref

from ._aux_functions import identity
from ._procedural import check_status
from ..exceptions import ErrorChannelMessage


class _NoOpStream():
    """Simple no-op file-like object."""

    def write(self, ignored):
        """No-op write method."""

    def flush(self):
        """No-op flush method."""


class OutputStream():
    """Class to parse and write strings to a file object."""

    def __init__(self, outputfile, env, fn=None, initerrorstr=False):
        """OutputStream constructor.

        outputfile must be a file-like object. At a minimum, it must
        provide methods write(self, str) and flush(self). Can be None to
        suppress output.

        If fn is specified, it must be a fuction with signature
        fn(str) -> str. Strings sent to this stream will be processed by
        this function before being written.

        This constructor is not meant to be used externally.
        """
        self._env = weakref.proxy(env)
        if fn:
            self._fn = fn
        else:
            self._fn = identity
        self._is_valid = False
        self._disposed = False
        # We only create this attribute for the error channel.
        if initerrorstr:
            self._error_string = None
        if outputfile:
            self._file = outputfile
        else:
            self._file = _NoOpStream()
        try:
            tst = callable(self._file.write)
        except AttributeError:
            tst = False
        if not tst:
            raise TypeError("outputfile must have a write method")
        try:
            tst = callable(self._file.flush)
        except AttributeError:
            tst = False
        if not tst:
            raise TypeError("outputfile must have a flush method")
        self._is_valid = True

    def _end(self):
        """Flush and free any open file.

        If the user passes in a filename string, we open it.  In that case,
        we need to clean it up.
        """
        if self._disposed:
            return
        self._disposed = True
        # If something bad happened in the constructor, then don't
        # attempt to flush or close.
        if self._is_valid:
            try:
                self.flush()
            except ValueError:
                # If the file is already closed, then ignore the error
                # and continue.
                pass

    def __del__(self):
        """OutputStream destructor."""
        self._end()

    def _write_wrap(self, str_):
        """Used when anything is written to the message channels.

        The _error_string attribute should only be present on the error
        channel.  If we detect that something was printed on the error
        channel, then we raise an ErrorChannelMessage along with this
        message.  The message can contain more information than what
        we'd get by calling CPXgeterrorstring.  For example, we may see
        format string specifiers rather having them filled in.

        See SWIG_callback.c:messagewrap.
        """
        try:
            self.write(str_)
            self.flush()
            # Check to see if something was written to the error
            # channel.
            try:
                # Remove trailing newlines.
                msg = self._error_string.strip()
            except AttributeError:
                msg = None
            if msg is not None:
                # Errors raised from callbacks are handled in
                # SWIG_callback.c:cpx_handle_pyerr.  If we do not have a
                # callback error (CPXERR_CALLBACK = 1006), then raise a
                # ErrorChannelMessage containing the last error string.
                # This gets special treatment in
                # _procedural.py:StatusChecker.
                if not msg.startswith("CPLEX Error  1006"):
                    self._error_string = None
                    # NB: This will be caught immediately below and stored
                    #     in self._env._callback_exception.
                    raise ErrorChannelMessage(msg)
        except Exception as exc:
            self._env._callback_exception = exc
            check_status._pyenv = self._env

    def write(self, msg):
        """Parses and writes a string.

        If self._fn is not None, self._fn(msg) is passed to
        self._file.write. Otherwise, msg is passed to self._file.write.
        """
        if msg is None:
            msg = ""
        self._file.write(self._fn(msg))

    def flush(self):
        """Flushes the buffer."""
        self._file.flush()
