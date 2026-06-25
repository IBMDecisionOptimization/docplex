'''
Created on Jan 12, 2016

Testing distrib bits.

@author: kong
'''
import fnmatch
import os
import unittest
import re

from docplex.mp.utils import open_universal_newline


class DistribTest(unittest.TestCase):
    def check_in_file(self, regexp, filename):
        """
        """
        pattern = re.compile(regexp)
        with open_universal_newline(filename, "r") as f:
            for i, line in enumerate(f):
                match = re.match(pattern, line)
                if match:
                    return i, match
        return None, None

    def match(self, regexp, s):
        pattern = re.compile(regexp)
        return re.match(pattern, s)

    def extract_copyright(self, filename):
        pattern = re.compile('^# Source file provided under Apache License, Version 2.0, January 2004, *$')
        with open_universal_newline(filename, "r") as f:
            for i, line in enumerate(f):
                match = re.match(pattern, line)
                if match:
                    l1 = None
                    l2 = None
                    try:
                        l1 = next(f)
                        l2 = next(f)
                    except:
                        pass
                    return i, [line, l1, l2]
        return None, None

    def reset_error_list(self):
        self.errors = []

    def expect(self, cond, msg):
        if not cond:
            self.errors.append("%s: %s" % (self.filename, msg))

    def set_filename(self, filename):
        self.filename = filename

    def verify_copyright(self, filename):
        self.reset_error_list()
        self.set_filename(filename)
        lineno, copyright_statement = self.extract_copyright(filename)
        self.expect(copyright_statement is not None,
                    "No copyright statement found")
        if copyright_statement:
            self.expect(len(copyright_statement) == 3,
                        "line %s: Expecting a 3 line copyright statement" % lineno)
            self.expect(copyright_statement[1] is not None,
                        "line %s: Incomplete copyright statement, missing lines" % lineno)
            self.expect(copyright_statement[2] is not None,
                        "line %s: Incomplete copyright statement, missing lines" % lineno)
            self.expect(self.match("^# http://www.apache.org/licenses/ *$",
                                   copyright_statement[1]),
                        "line %s: Expecting # http://www.apache.org/licenses/ here" % (lineno+1))
            m = self.match("^# \(c\) Copyright IBM Corp. ([0-9]+)(, ([0-9]+))?.*$",
                           copyright_statement[2])
            l2 = lineno + 2
            year1 = None
            year2 = None
            if m:
                year1 = m.groups()[0]
                year2 = m.groups()[2]
            if year2 == None:
                year2 = year1

            self.expect(m,
                        "line %s: Expecting # (c) Copyright IBM Corp. year1[, year2], got: %s" % (l2,copyright_statement[2]))
            if year1:
                self.expect(int(year1) >= 2015,
                            "line %s: Year1 is supposed to be >= 2015" % l2)
                self.expect(int(year2) >= 2022,
                            "line %s: Year2 is supposed to be >= 2022" % l2)

        if len(self.errors) == 0:
            return None
        return self.errors

    def test_copyright_headers(self):
        """This test assume that:
        - test run in docplex_tests/
        - distribution files have been extracted to target/libs/python
        - distribution files contain:
           - lib extracted as docplex (contains docplex/mp and docplex/cp)
           - examples extracted (in examples/mp and examples/cp)

        The test will make sure that *.py files contain the following header:
        # Source file provided under Apache License, Version 2.0, January 2004,
        # http://www.apache.org/licenses/
        # (c) Copyright IBM Corp. 2015, 2022

        It will also test that years for the copyright has 2015 as 1st and
        2022 as 2nd
        """
        pattern = "*.py"
        dirs_to_explore = {"target/libs/python/docplex",
                           "target/libs/python/examples/mp",
                           "target/libs/python/examples/cp"
                           }
        files = []
        for d in dirs_to_explore:
            for (dirpath, dirnames, filenames) in os.walk(d):

                for f in filenames:
                    if (fnmatch.fnmatch(f, pattern)):
                        files.append(os.path.join(dirpath, f))

        print("parsing %s files" % len(files))

        all_errors = []
        for i in files:
            errors = self.verify_copyright(i)
            if errors:
                # create error report
                for e in errors:
                    all_errors.append("   %s" % e)

        if len(all_errors) != 0:
            all_errors.append("%s error found" % len(all_errors))
            self.fail('\n'.join(all_errors))


if __name__ == "__main__":
    unittest.main()
