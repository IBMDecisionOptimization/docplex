'''This test suite validates the DOcplex documentation

@author: vblanchard
'''
import unittest
import sys
import os

class DocTests(unittest.TestCase):
    doc_path = ''
    def setUp(self):
        sys.stderr.flush()
        sys.stdout.flush()
        dn = os.path.dirname
        self.docplex_tests_home = dn(dn(dn(dn(__file__))))
        self.doc_path = self.docplex_tests_home + os.sep + "target" + os.sep + "doc"


    def tearDown(self):
        sys.stderr.flush()
        sys.stdout.flush()

    def test_RTC_32190(self):
        if not os.path.isdir(self.doc_path):
            print("Test SKIPPED ! %s does not exists" % self.doc_path)
            return

        doc_path = self.doc_path + os.sep + "docs"
        examples_path = self.docplex_tests_home + os.sep + "target" + os.sep + "docplex-mp-all-samples" + os.sep + "examples"
        for path, subdirs, files in (os.walk(doc_path) or os.walk(examples_path)):
            for name in files:
                if os.path.isfile(os.path.join(path, name)):
                    f = open(os.path.join(path, name), 'rb')
                    lines = f.read().decode('ISO-8859-1')
                    f.seek(0)
                    self.assertNotIn('datascientistworkbench', lines,
                                     msg='file {} refers to datascientistworkbench instead of datascience.ibm'.format(
                                         os.path.join(path, name)))
                    f.close()

    def test_doc_files(self):
        print("Testing presence of some doc files")
        doc_path = self.doc_path
        if not os.path.isdir(self.doc_path):
            print("Test SKIPPED ! %s does not exists" % self.doc_path)
            return
        for f in ['LICENSE.txt','README.md']:
            self.assertTrue(os.path.isfile(doc_path+os.sep+f),msg="Missing doc file: {}".format(f))
        self.assertTrue(os.path.isdir(doc_path + os.sep + "docs"),msg="Missing folder docs")
        for f in ["cp.html","index.html","mp_vs_cp.html","genindex.html","mp.html","search.html"]:
            self.assertTrue(os.path.isfile(doc_path + os.sep + "docs" + os.sep + f),msg="Missing doc file: {}".format(f))
        for d in ["mp","cp"]:
            self.assertTrue(os.path.isdir(doc_path + os.sep + "docs" + os.sep + d),msg="Missing doc folder: {}".format(d))

if __name__ == "__main__":
    unittest.main()
