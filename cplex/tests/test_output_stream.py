# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2013, 2026. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# ---------------------------------------------------------------------------
"""
Tests the cplex._internal._ostream.OutputStream class.

No command line arguments are required.
"""
import unittest
import cplex
from cplextestcase import CplexTestCase
from cplex.exceptions import CplexError
from cplex._internal._ostream import OutputStream

MUST_HAVE_WRITE_MSG = 'outputfile must have a write method'
MUST_HAVE_FLUSH_MSG = 'outputfile must have a flush method'


class Dummy():
    """A test dummy object."""
    pass


class FileDummy():
    """A test dummy file-like object."""

    def __init__(self):
        # File-like objects should implement the closed attribute.
        self.closed = False
        self.closed_count = 0
        self.content = ''
        self.was_flushed = False

    def write(self, str_):
        self.content += str_

    def flush(self):
        self.was_flushed = True

    def close(self):
        self.closed = True
        self.closed_count += 1


class OutputStreamTests(CplexTestCase):

    def testHasWrite(self):
        file_dummy = Dummy()
        self._testInit(file_dummy, MUST_HAVE_WRITE_MSG)

    def testHasCallableWrite(self):
        file_dummy = Dummy()
        file_dummy.write = True
        self._testInit(file_dummy, MUST_HAVE_WRITE_MSG)

    def testHasFlush(self):
        file_dummy = Dummy()
        file_dummy.write = lambda x : None
        self._testInit(file_dummy, MUST_HAVE_FLUSH_MSG)

    def testHasCallableFlush(self):
        file_dummy = Dummy()
        file_dummy.write = lambda x : None
        file_dummy.flush = True
        self._testInit(file_dummy, MUST_HAVE_FLUSH_MSG)

    def testWrite(self):
        file_dummy = FileDummy()
        self.assertEqual(file_dummy.content, '')
        env = cplex._internal.Environment()
        ostrm = OutputStream(file_dummy, env, fn=None)
        ostrm.write(None)
        self.assertEqual(file_dummy.content, '')
        ostrm.write('foo')
        self.assertEqual(file_dummy.content, 'foo')

    def testWriteWithFunction(self):
        def modify(str_):
            return str_ + '|'
        file_dummy = FileDummy()
        self.assertEqual(file_dummy.content, '')
        env = cplex._internal.Environment()
        ostrm = OutputStream(file_dummy, env, fn=modify)
        ostrm.write(None)
        self.assertEqual(file_dummy.content, '|')
        ostrm.write('foo')
        self.assertEqual(file_dummy.content, '|foo|')

    def testFlush(self):
        file_dummy = FileDummy()
        self.assertFalse(file_dummy.was_flushed)
        env = cplex._internal.Environment()
        ostrm = OutputStream(file_dummy, env, fn=None)
        ostrm.flush()
        self.assertTrue(file_dummy.was_flushed)

    def testFlushedAfterDel(self):
        file_dummy = FileDummy()
        self.assertFalse(file_dummy.was_flushed)
        env = cplex._internal.Environment()
        ostrm = OutputStream(file_dummy, env, fn=None)
        del ostrm
        self.assertTrue(file_dummy.was_flushed)

    def testWithNone(self):
        env = cplex._internal.Environment()
        ostrm = OutputStream(None, env, fn=None)
        ostrm.write(None)
        ostrm.flush()
        # As long as an exception isn't thrown we're happy.

    def testWithString(self):
        env = cplex._internal.Environment()
        with self._getTempFileName(delete=False) as tmp:
            with self.assertRaises(TypeError) as cm:
                ostrm = OutputStream(tmp, env, fn=None)
            self.assertEqual(str(cm.exception), MUST_HAVE_WRITE_MSG)

    def testDelWithClosedFile(self):
        env = cplex._internal.Environment()
        with self._getTempFileName(delete=False) as tmp:
            with open(tmp, 'w') as f:
                ostrm = OutputStream(f, env, fn=None)
            del ostrm
            # As long as an exception isn't thrown we're happy.

    def testFlushClosedDummy(self):
        """Test that we attempt to flush the stream when it is destroyed."""
        file_dummy = FileDummy()
        self.assertEqual(file_dummy.closed_count, 0)
        env = cplex._internal.Environment()
        ostrm = OutputStream(file_dummy, env, fn=None)
        file_dummy.close()
        self.assertEqual(file_dummy.closed_count, 1)
        del ostrm
        self.assertTrue(file_dummy.was_flushed)

    def testDontCloseFileObject(self):
        file_dummy = FileDummy()
        self.assertEqual(file_dummy.closed_count, 0)
        env = cplex._internal.Environment()
        ostrm = OutputStream(file_dummy, env, fn=None)
        del ostrm
        self.assertEqual(
            file_dummy.closed_count, 0,
            "We shouldn't close file-like objects!")

    def _testInit(self, file_object, msg):
        env = cplex._internal.Environment()
        with self.assertRaises(TypeError) as cm:
            ostrm = OutputStream(file_object, env, fn=None)
        self.assertEqual(str(cm.exception), msg)


def main():
    unittest.main()

if __name__ == '__main__':
    main()
