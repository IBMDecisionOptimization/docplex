# ---------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2009, 2026. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# ---------------------------------------------------------------------------
"""
Contains a base test case class and tests.

No command line arguments are required.
"""
import unittest
import difflib
import os
import pathlib
import platform
import errno
import tempfile
import uuid
import cplex
from contextlib import contextmanager

# Root of the cplex/ project directory (two levels up from this file:
# cplex/tests/cplextestcase.py → cplex/tests/ → cplex/)
CPLEX_SOURCE_DIR = pathlib.Path(__file__).resolve().parent.parent

@contextmanager
def getTempLPFile(model):
    try:
        with CplexTestCase._getTempFileName(ext='.lp', delete=False) as tmp:
            with open(tmp, mode='w') as tmpf:
                tmpf.write(model)
        yield tmp
    finally:
        CplexTestCase._failSafeDelete(tmp)


def enumerate_reversed(seq):
    """Enumerate a sequence in reverse."""
    for idx in reversed(range(len(seq))):
        yield idx, seq[idx]


class CplexTestCase(unittest.TestCase):
    """
    CplexTestCase is a base test case class that contains useful helper methods.

    New test cases can inherit from CplexTestCase to have access to these.
    """

    # Allow for showing the entire diff when we fail a test.
    maxDiff = None

    @staticmethod
    def _newCplex(source=None):
        """
        Create a CPLEX instance that has output disabled.

        We need this to avoid error messages printed from the callable library.
        If errors are expected then we test for them explicitly.
        """
        if source:
            cpx = cplex.Cplex(source)
        else:
            cpx = cplex.Cplex()
        CplexTestCase._setAllStreams(cpx, None)
        return cpx

    @staticmethod
    def _setAllStreams(cpx, stream):
        """Set all streams (i.e., results, warning, error, log)."""
        # If the name of a file is passed in, this would result in four
        # different file-like-objects being opened at once. The last
        # one will overwrite the others!
        if isinstance(stream, str):
            raise TypeError
        return (cpx.set_results_stream(stream),
                cpx.set_warning_stream(stream),
                cpx.set_error_stream(stream),
                cpx.set_log_stream(stream))

    @staticmethod
    def _failSafeDelete(path):
        """
        Delete a file.

        If the file doesn't exist then silently continue.  Uses Python's
        easier to ask for forgiveness than permission (EAFP) idiom.
        """
        # If path is a unicode string, then encode it before passing to
        # os.remove.
        if isinstance(path, str):
            path = path.encode('utf-8')
        try:
            os.remove(path)
        except OSError as e:
            # errno.ENOENT = no such file or directory
            if e.errno != errno.ENOENT:
                # re-raise exception if a different error occured
                raise

    @staticmethod
    def _touch(path):
        """
        Create an empty file at path.
        """
        # This could be extended to create any directories that are not in path
        # by doing something like the following:
        # basedir = os.path.dirname(path)
        # if not os.path.isdir(basedir):
        #     os.makedirs(basedir)
        with open(path, 'a'):
            os.utime(path, None)

    @classmethod
    def _getResource(cls, path) -> str:
        """
        Return the absolute path to a resource file under CPLEX_SOURCE_DIR.

        Raises FileNotFoundError if the resource does not exist, so callers
        get a clear diagnostic rather than a cryptic error later.
        """
        resource = CPLEX_SOURCE_DIR / path
        if not resource.exists():
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), str(resource)
            )
        return resource._str

    @staticmethod
    def _readFile(path):
        """
        Read a file and return the contents as a string.
        """
        with open(path) as f:
            return f.read()

    @staticmethod
    @contextmanager
    def _getTempFileName(ext='', delete=True):
        """Returns a safe temporary file name.

        The file will be created, but has been closed.  By default, the
        file will be deleted when this context manager goes out of
        scope.  If an extension is specified, it will be added as a
        suffix (you need to include the '.' if you want it).
        """
        (fd, filename) = tempfile.mkstemp(suffix=ext)
        try:
            os.close(fd)
            assert os.path.getsize(filename) == 0
            yield filename
        finally:
            if delete:
                CplexTestCase._failSafeDelete(filename)

    def assertListsAlmostEqual(self, lhs, rhs, delta=None):
        size = len(lhs)
        self.assertEqual(size, len(rhs))
        for i in range(size):
            if delta is not None:
                self.assertAlmostEqual(lhs[i], rhs[i], delta=delta)
            else:
                self.assertAlmostEqual(lhs[i], rhs[i])

    def compareLists(self, actual, expected):
        """Compare lists of strings (one line per list entry).

        If we fail to match a unified diff will be printed.
        """
        udiff = difflib.unified_diff(expected, actual)
        isempty = True
        for line in udiff:
            isempty = False
            print(line.strip())
        self.assertTrue(isempty)

    @staticmethod
    def iswindows():
        return platform.system() in ('Windows', 'Microsoft')

    @staticmethod
    def affectsParamTesting(cpx):
        return ((cpx.parameters.record.get() ==
                 cpx.parameters.record.values.on) or
                (cpx.parameters.parallel.get() ==
                 cpx.parameters.parallel.values.opportunistic))

    def skipIfParamTesting(self, cpx):
        if CplexTestCase.affectsParamTesting(cpx):
            self.skipTest("testing recording or opportunistic mode")


class CplexTestCaseTests(CplexTestCase):
    """
    Tests CplexTestCase.
    """

    def testNewCplex(self):
        def _checkForNullOutputStream(ostream):
            self.assertFalse(ostream is None)
            self.assertTrue(type(ostream) ==
                            cplex._internal._ostream.OutputStream)
            self.assertIsInstance(ostream._file,
                                  cplex._internal._ostream._NoOpStream)
            self.assertEqual(ostream._fn,
                             cplex._internal._aux_functions.identity)
        with self._newCplex() as cpx:
            _checkForNullOutputStream(cpx._env._get_results_stream())
            _checkForNullOutputStream(cpx._env._get_warning_stream())
            _checkForNullOutputStream(cpx._env._get_error_stream())
            _checkForNullOutputStream(cpx._env._get_log_stream())

    def testReadFileEmpty(self):
        filename = 'foo.txt'
        self._failSafeDelete(filename)
        self._touch(filename)
        actual = self._readFile(filename)
        self.assertEqual(actual, '')
        self._failSafeDelete(filename)

    def testReadFile(self):
        filename = 'foo.txt'
        self._failSafeDelete(filename)
        with open(filename, 'w') as tmp:
            tmp.write('aaa')
        actual = self._readFile(filename)
        self.assertEqual(actual, 'aaa')
        self._failSafeDelete(filename)

    def testFailSafeDeleteFile(self):
        filename = 'foo.txt'
        self._failSafeDelete(filename)
        self.assertFalse(os.path.isfile(filename))
        self._touch(filename)
        self.assertTrue(os.path.isfile(filename))
        self._failSafeDelete(filename)
        self.assertFalse(os.path.isfile(filename))

    def testGetTempFileName(self):
        with self._getTempFileName() as tmp:
            self.assertFalse(tmp is None)
            self.assertFalse(len(tmp) == 0)
        self.assertFalse(os.path.exists(tmp))

    def testGetTempFileNameWithExtension(self):
        with self._getTempFileName('.sav') as tmp:
            self.assertTrue(tmp.endswith('.sav'))
        self.assertFalse(os.path.exists(tmp))

    def testDeletesByDefault(self):
        with self._getTempFileName() as tmp:
            self.assertTrue(os.path.exists(tmp))
        self.assertFalse(os.path.exists(tmp))

    def testOptOutOfDelete(self):
        with self._getTempFileName(delete=False) as tmp:
            self.assertTrue(os.path.exists(tmp))
        self.assertTrue(os.path.exists(tmp))
        self._failSafeDelete(tmp)
        self.assertFalse(os.path.exists(tmp))

    def testWithoutWith(self):
        cntxtmgr = self._getTempFileName()
        tmp = cntxtmgr.__enter__()
        self.assertTrue(os.path.exists(tmp))
        cntxtmgr.__exit__(None, None, None)
        self.assertFalse(os.path.exists(tmp))


def main():
    unittest.main()


if __name__ == '__main__':
    main()
