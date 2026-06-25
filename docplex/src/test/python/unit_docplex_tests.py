'''
Created on Jul 16, 2018

@author: kong
'''
import compileall
import shutil
from shutil import ignore_patterns
import unittest

import docplex.mp
from os.path import abspath, dirname, join

from testutils import temporary_directory


class UnitDocplexTests(unittest.TestCase):
    def test_compile_lib(self):
        '''RTC 37306: This unittest just compile everything in lib and check
        that the compilation does not fail (no syntax error even in not
        used or imported files.
        '''
        lib_root = dirname(dirname(abspath(docplex.mp.__file__)))
        print('library root = %s' % lib_root)
        with temporary_directory(delete=True) as temp:
            dest = join(temp, 'docplex')
            print('temp dir: %s' % temp)
            shutil.copytree(lib_root, dest,
                            ignore=ignore_patterns('*.pyc', '__pycache__'))
            print('compiling %s' % dest)
            success = compileall.compile_dir(dest)
            self.assertTrue(success != 0,
                            'an error happened when compiling the lib')

if __name__ == "__main__":
    unittest.main()
