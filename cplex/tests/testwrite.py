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
Tests the Cplex.write() method.

No command line arguments are required.
"""
import unittest
import os.path
import io
from cplex.exceptions import CplexError, CplexSolverError
from cplex.exceptions import error_codes
from cplextestcase import CplexTestCase

LP_EXAMPLE_FILE = '../../../examples/data/lpprog.lp'
MIP_EXAMPLE_FILE = '../../../examples/data/case1.lp'
QP_EXAMPLE_FILE = '../../../examples/data/qpex.lp'
QCP_EXAMPLE_FILE = '../../../examples/data/qcp.lp'


class NoOpStream():
    """Simple no-op file-like object."""

    def __init__(self):
        self.was_called = False

    def write(self, data):
        self.was_called = True

    def flush(self):
        pass


# TODO: It would be nice if the supported file types were available in the
#       Cplex object so that it would be self-documented what file types are
#       available.

class WriteTests(CplexTestCase):

    def testWriteSav(self):
        self._testWriteFileTypeComplete('.sav')

    def testWriteMps(self):
        self._testWriteFileTypeComplete('.mps')

    def testWriteLP(self):
        self._testWriteFileTypeComplete('.lp')

    def testWriteRew(self):
        self._testWriteFileTypeComplete('.rew')

    def testWriteRlp(self):
        self._testWriteFileTypeComplete('.rlp')

    def testWriteAlp(self):
        self._testWriteFileTypeComplete('.alp')

    def testWriteDua(self):
        self._testWriteFileTypeComplete('.dua')

    def testWriteEmb(self):
        self._testWriteFileTypeComplete('.emb')

    def testWriteDpe(self):
        self._testWriteFileTypeComplete('.dpe')

    def testWritePpe(self):
        self._testWriteFileTypeComplete('.ppe')

    def testWriteToUnknownFileType(self):
        try:
            self._testWriteFileTypeComplete('.bogus')
            self.fail()
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2], error_codes.CPXERR_BAD_FILETYPE)

    def testWriteUnspecified(self):
        try:
            self._testWriteFileType('')
            self.fail()
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2], error_codes.CPXERR_BAD_FILETYPE)

    def testWriteEmpty(self):
        try:
            self._testWriteFileType('', startfile=None)
            self.fail()
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2], error_codes.CPXERR_BAD_FILETYPE)

    def testWriteToExisting(self):
        cpx = self._newCplex()
        cpx.read(LP_EXAMPLE_FILE)
        with self._getTempFileName('.sav') as tmp:
            cpx.write(tmp)
            self.assertTrue(os.path.exists(tmp))
            # Now, do it again.  Expecting that it will overwrite the file
            # silently.
            cpx.write(tmp)
            self.assertTrue(os.path.exists(tmp))

    def testWriteToNone(self):
        try:
            cpx = self._newCplex()
            cpx.write(None)
            self.fail()
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2], error_codes.CPXERR_NO_FILENAME)

    def _testWriteFileTypeComplete(self, ext):
        self.assertFalse(ext is None)
        self.assertFalse(len(ext) == 0)
        # test implied file type
        self._testWriteFileType(ext)
        self._testWriteFileType(ext, filetype=None)
        # test bogus file extension with specified file type
        self._testWriteFileType('.bogus', filetype=ext[1:])
        # test explicit file type
        self._testWriteFileType(ext, filetype=ext[1:])
        # test with different compression types
        for compext in ('.gz', '.bz2'):
            self._testWriteFileType(ext + compext)
            # compression extensions are not allowed in the filetype
            try:
                self._testWriteFileType(ext + compext, filetype=ext[1:] + compext)
                self.fail()
            except CplexSolverError as cse:
                self.assertEqual(cse.args[2], error_codes.CPXERR_BAD_FILETYPE)
        # test with different problem types
        try:
            self._testWriteFileType(ext, startfile=MIP_EXAMPLE_FILE)
            self.assertFalse(ext in ['.dua', '.emb'], 'Should only work with LP problems!')
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2], error_codes.CPXERR_NOT_FOR_MIP)
        try:
            self._testWriteFileType(ext, startfile=QP_EXAMPLE_FILE)
            self.assertFalse(ext in ['.dua', '.emb'], 'Should only work with LP problems!')
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2], error_codes.CPXERR_NOT_FOR_QP)
        try:
            self._testWriteFileType(ext, startfile=QCP_EXAMPLE_FILE)
            self.assertFalse(ext in ['.dua', '.emb'], 'Should only work with LP problems!')
        except CplexSolverError as cse:
            self.assertEqual(cse.args[2], error_codes.CPXERR_NOT_FOR_QCP)

    def _testWriteFileType(self, ext, filetype='', startfile=LP_EXAMPLE_FILE):
        cpx = self._newCplex()
        if startfile:
            cpx.read(startfile)
        with self._getTempFileName(ext) as tmp:
            # by default, filetype is implied via the filename extension.
            cpx.write(tmp, filetype)
            self.assertTrue(os.path.exists(tmp))
            # We can verify that some of the file types were written correctly
            # by making sure that we can read them back in.
            readable = ('.lp', '.mps', '.sav')
            compext = ('', '.gz', '.bz2')
            #if ext not in ('.bogus', '.rlp', '.rew', '.ppe', '.dua', '.emb', '.dpe', '.alp'):
            if ext in (r + c for r in readable for c in compext):
                #print "EXTENSION:", ext, "FILETYPE:", filetype
                cpx.read(tmp, filetype=filetype)

    def testWriteUnicodeFileName(self):
        try:
            with self._newCplex() as cpx:
                # Chinese for "empty problem"
                filename = u"\u7A7A\u554F\u984C.lp"
                cpx.write(filename)
                # Attempt to read the file back in.
                cpx.read(filename)
        finally:
            self._failSafeDelete(filename)

    def testWriteAsStringLP(self):
        cpx = self._newCplex()
        self.skipIfParamTesting(cpx)
        cpx.read(LP_EXAMPLE_FILE)
        actual = cpx.write_as_string()  # "lp" is the default
        expected = """\
\\ENCODING=ISO-8859-1
\\Problem name: ../../../examples/data/lpprog.lp

Minimize
 obj: - x1 - 2 x2 - 3 x3 + 4 x4 + 5 x5
Subject To
 row1: 8 x1 + x2 + 6 x3 <= 200
 row2: 3 x1 + 5 x2 + 7 x3 <= 300
 row3: 4 x1 + 9 x2 + 2 x3 <= 400
Bounds
 1 <= x1 <= 100
 2 <= x2 <= 200
 3 <= x3 <= 150
 1 <= x4 <= 100
 2 <= x5 <= 200
End
"""
        self.assertEqual(actual, expected)

    def testWriteAsStringLPUnicode(self):
        cpx = self._newCplex()
        self.skipIfParamTesting(cpx)
        # The magic comment at the top of this file means that Python
        # will use UTF-8 as the file encoding. Thus, the names below use
        # UTF-8. We have to set the fileencoding parameter so that the
        # names are "written" properly to the string.
        cpx.parameters.read.fileencoding.set("utf-8")
        cpx.read(LP_EXAMPLE_FILE)

        nameset = ["motörhead", "Ørsted", "Gauß", "Nuñoz"]
        self.assertGreater(cpx.variables.get_num(), len(nameset))
        cpx.variables.set_names([(idx, name) for idx, name in enumerate(nameset)])

        for idx, name in enumerate(nameset):
            self.assertEqual(cpx.variables.get_names(idx), name)

        actual = cpx.write_as_string()  # "lp" is the default

        with self._getTempFileName('.lp', delete=False) as tmp:
            with io.open(tmp, mode="w", encoding="utf-8") as out:
                out.write(actual)
            with self._newCplex() as cpx2:
                # See comment above about UTF-8.
                cpx2.parameters.read.fileencoding.set("utf-8")
                cpx2.read(tmp)
                for idx, name in enumerate(nameset):
                    self.assertEqual(cpx2.variables.get_names(idx), name)

        expected = """\
\\ENCODING=UTF-8
\\Problem name: ../../../examples/data/lpprog.lp

Minimize
 obj: - motörhead - 2 Ørsted - 3 Gauß + 4 Nuñoz + 5 x5
Subject To
 row1: 8 motörhead + Ørsted + 6 Gauß <= 200
 row2: 3 motörhead + 5 Ørsted + 7 Gauß <= 300
 row3: 4 motörhead + 9 Ørsted + 2 Gauß <= 400
Bounds
 1 <= motörhead <= 100
 2 <= Ørsted <= 200
 3 <= Gauß <= 150
 1 <= Nuñoz <= 100
 2 <= x5 <= 200
End
"""
        self.assertEqual(actual, expected)


    def testWriteAsStringMPS(self):
        cpx = self._newCplex()
        self.skipIfParamTesting(cpx)
        cpx.read(LP_EXAMPLE_FILE)
        actual = cpx.write_as_string("mps")
        expected = """\
* ENCODING=ISO-8859-1
NAME          ../../../examples/data/lpprog.lp
ROWS
 N  obj     
 L  row1    
 L  row2    
 L  row3    
COLUMNS
    x1        obj                            -1
    x1        row1                            8
    x1        row2                            3
    x1        row3                            4
    x2        obj                            -2
    x2        row1                            1
    x2        row2                            5
    x2        row3                            9
    x3        obj                            -3
    x3        row1                            6
    x3        row2                            7
    x3        row3                            2
    x4        obj                             4
    x5        obj                             5
RHS
    rhs       row1                          200
    rhs       row2                          300
    rhs       row3                          400
BOUNDS
 LO bnd       x1                              1
 UP bnd       x1                            100
 LO bnd       x2                              2
 UP bnd       x2                            200
 LO bnd       x3                              3
 UP bnd       x3                            150
 LO bnd       x4                              1
 UP bnd       x4                            100
 LO bnd       x5                              2
 UP bnd       x5                            200
ENDATA
"""
        self.assertEqual(actual, expected)

    def testWriteAsStringSAV(self):
        with self._newCplex() as cpx1, \
             self._getTempFileName(".sav") as tmp:
            self.skipIfParamTesting(cpx1)
            cpx1.read(LP_EXAMPLE_FILE)
            modelstring = cpx1.write_as_string("sav")
            with io.open(tmp, "wb") as stream:
                stream.write(modelstring)
            with self._newCplex() as cpx2:
                cpx2.read(tmp)
                self.assertEqual(cpx1.variables.get_num(),
                                 cpx2.variables.get_num())
                self.assertEqual(cpx1.linear_constraints.get_num(),
                                 cpx2.linear_constraints.get_num())

    @staticmethod
    def get_file_types():
        return ["lp", "mps", "rew", "rlp", "alp", "sav"]

    @staticmethod
    def get_compression_types():
        # The empty string is a place holder for compression type "none"
        # (i.e., so that we also test without compression).
        return ["", "gz", "bz2"]

    def testWriteAsString(self):
        cpx = self._newCplex()
        self.skipIfParamTesting(cpx)
        cpx.read(LP_EXAMPLE_FILE)
        for filetype in WriteTests.get_file_types():
            for comptype in WriteTests.get_compression_types():
                modelstring = cpx.write_as_string(filetype, comptype)
                if (filetype.lower().startswith("sav") or comptype):
                    self.assertIsInstance(modelstring, bytes)
                else:
                    self.assertIsInstance(modelstring, str)
                self.assertGreater(len(modelstring), 0)

    def testWriteAsStringBadFileType(self):
        cpx = self._newCplex()
        self.skipIfParamTesting(cpx)
        cpx.read(LP_EXAMPLE_FILE)
        with self.assertRaises(CplexSolverError) as cse:
            cpx.write_as_string(filetype="bogus")
        self.assertEqual(cse.exception.args[2],
                         error_codes.CPXERR_BAD_FILETYPE)

    def testWriteAsStringBadCompType(self):
        cpx = self._newCplex()
        self.skipIfParamTesting(cpx)
        cpx.read(LP_EXAMPLE_FILE)
        with self.assertRaises(ValueError):
            cpx.write_as_string(comptype="bogus")

    def testWriteToStreamNoOp(self):
        cpx = self._newCplex()
        self.skipIfParamTesting(cpx)
        cpx.read(LP_EXAMPLE_FILE)
        for filetype in WriteTests.get_file_types():
            for comptype in WriteTests.get_compression_types():
                stream = NoOpStream()
                cpx.write_to_stream(stream, filetype=filetype,
                                    comptype=comptype)
                self.assertTrue(stream.was_called)

    def testWriteToStreamToDisk(self):
        cpx1 = self._newCplex()
        self.skipIfParamTesting(cpx1)
        cpx1.read(LP_EXAMPLE_FILE)
        for filetype in WriteTests.get_file_types():
            for comptype in WriteTests.get_compression_types():
                ext = ".{0}".format(filetype)
                if comptype:
                    ext += ".{0}".format(comptype)
                with self._getTempFileName(ext) as tmp:
                    with io.open(tmp, "wb") as stream:
                        cpx1.write_to_stream(stream, filetype=filetype,
                                             comptype=comptype)
                    with self._newCplex() as cpx2:
                        cpx2.read(tmp)
                        self.assertEqual(cpx1.variables.get_num(),
                                         cpx2.variables.get_num())
                        self.assertEqual(cpx1.linear_constraints.get_num(),
                                         cpx2.linear_constraints.get_num())

    def testWriteToNoFlushStream(self):
        class NoFlushStream():
            # Missing flush() method.
            def write(self, data):
                pass
        with self._newCplex() as cpx:
            self.skipIfParamTesting(cpx)
            stream = NoFlushStream()
            with self.assertRaises(CplexError):
                cpx.write_to_stream(stream)

    def testWriteToNoWriteStream(self):
        class NoWriteStream():
            # Missing write() method.
            def flush(self):
                pass
        with self._newCplex() as cpx:
            self.skipIfParamTesting(cpx)
            stream = NoWriteStream()
            with self.assertRaises(CplexError):
                cpx.write_to_stream(stream)

    def testWriteToStringIO(self):
        """Tests writing a model to a StringIO instance.

        This test is a bit silly, because in most cases the user does not
        need to do this (i.e., the user can use the string result from
        Cplex.write_as_string() directly). Cplex.write_as_string() is a
        convenience method and takes care of the ugly details. Namely, we
        have to use a BytesIO and decode the byte strings using
        Cplex.parameters.read.fileencoding. We also have to be aware of
        the file type requested (e.g., the SAV, GZ, and BZ2, extensions
        will always need to be byte strings).

        Thus, this test is here for documentation purposes and to check
        that no unexpected exception is thrown.
        """
        class FooStream:
            def __init__(self, stream, encoding):
                self.stream = stream
                self.encoding = encoding
            def write(self, data):
                self.stream.write(data.decode(encoding))
            def flush(self):
                pass
        with self._newCplex() as cpx:
            self.skipIfParamTesting(cpx)
            cpx.read(LP_EXAMPLE_FILE)
            encoding = cpx.parameters.read.fileencoding.get()
            with io.StringIO() as stream:
                foo = FooStream(stream, encoding)
                cpx.write_to_stream(foo, filetype='LP')
                result = stream.getvalue()
                self.assertGreater(len(result), 0)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
