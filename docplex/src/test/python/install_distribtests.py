
import importlib
import os
from os import walk
import pip
import shutil
from shutil import copyfileobj
import sys
import tempfile
import unittest
import zipfile

try:
    import urllib.request as urllib
except ImportError:
    import urllib


def install(package):
    try:
        pip.main(['install',package])
    except:
        print("Installation of docplex failed")

import imp


class TestList():
    @unittest.skip("Don't execute installation test automatically for now")
    def test_install(self):
        print("*** Test installed modules ***")
        install('docplex')
        pip.utils.pkg_resources = imp.reload(pip.utils.pkg_resources)
        installed_packages = [p.key for p in pip.get_installed_distributions()]
        # Check docplex and dependencies are installed
        modules = ['docplex', 'requests', 'six']
        for m in modules:
            print('check {} was installed'.format(m))
            assert m in installed_packages, "{} is missing".format(m)
        try:
            import docplex.mp
            import_mp = True
        except ImportError:
            print('could not import docplex.mp')
            import_mp = False
        assert import_mp is True
        try:
            import docplex.cp
            import_cpo = True
        except ImportError:
            print('could not import docplex.cp')
            import_cpo = False
        assert import_cpo is True

    # @unittest.skip("Don't execute installation test automatically for now")
    def test_installed_examples(self):


        print("*** Test installed examples ***")
        extract_path = tempfile.gettempdir() + os.sep + "examples"
        # download zip file from github https://github.com/IBMDecisionOptimization/docplex-examples/archive/master.zip
        url = 'https://github.com/IBMDecisionOptimization/docplex-examples/archive/master.zip'

        if sys.version_info[0] == 2:
            # on python 2, we'll use a no cert context since the pythons in jenkins
            # don't have the proper cert. Use our own cert file
            # but the then we cannot use urlretrieve because urlretrieve
            # does not allow customized certs
            cert = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                'data', 'requests_290_cacert.pem')
            print("using certificate %s" % cert)
            try:
                import urllib2
                in_stream = urllib2.urlopen(url, cafile=cert)
                with open('examples.zip', 'wb') as out_file:
                    copyfileobj(in_stream, out_file)
            finally:
                pass
        else:
            urllib.urlretrieve(url, 'examples.zip')
        # try:
        #     urllib.urlretrieve(url, 'examples.zip') # Python 2
        # except :
        #     urllib.request.urlretrieve(url, 'examples.zip') # Python 3

        try:
            examples = zipfile.ZipFile('examples.zip')
            examples.extractall(path=extract_path)
        except zipfile.error as e:
            print("Bad zipfile (from {}): {}".format(url, e))
        assert os.path.isdir(extract_path + os.sep + "docplex-examples-master") is True

        examples_root = os.path.join(extract_path, "docplex-examples-master", "examples")

        # Disable visu and possible extra traces
        from docplex.cp.config import context as cp_context
        cp_context.visu_enabled = False
        cp_context.solver.docloud.trace_response = False
        cp_context.model.trace_cpo = False

        modules = ['mp','cp']
        for m in modules:
            # examples_dirs = []
            # for (dirpath, dirnames, filenames) in walk(examples_root+os.sep+m):
            #     examples_dirs.extend(dirnames)
            examples_dirs = os.listdir(examples_root + os.sep + m)
            for d in examples_dirs:
                if d == 'visu':
                    sys.path.append(os.path.join(examples_root, m, d))
                examples = os.listdir(os.path.join(examples_root, m, d))
                for e in examples:
                    code = ""
                    if e.endswith(".py") and e != "__init__.py" and e != "_utils_visu.py":
                        example = os.path.join(examples_root, m, d, os.path.basename(e))
                        print("\n\t*** Test sample {} ***".format(example))
                        # print("sys.path= %s" % sys.path)
                        orig_stdout = sys.stdout
                        f = open('output.txt','w')
                        sys.stdout = f
                        try:
                            with open(example) as f:
                                code = compile(f.read(), example, 'exec')
                                exec(code, {'__name__': '__main__',
                                            "__file__": example})
                                f.close()
                        except RuntimeError:
                            print('sample {} failed to run'.format(example))
                        sys.stdout = orig_stdout
                        f.close()
                        if 'visu' not in e:
                            of = open('output.txt', 'r')
                            print(of)
                            max_len = len(max(of, key=len))
                            self.assertLess(max_len, 81, 'output is too long for DropSolve:  \nsee RTC-29931')
                            of.close()
                        # os.remove('output.txt')
                        # Clean up
        try:
            os.remove("examples.zip")
            shutil.rmtree(extract_path)
            print("environment cleaned")
        except OSError:
            print("Could not remove examples zip file and/or examples folder, please delete it manually")


class SampleTests(unittest.TestCase, TestList):
    def setUp(self):
        self.context = None
        self.use_cloud = False
        sys.stderr.flush()
        sys.stdout.flush()

    def tearDown(self):
        sys.stderr.flush()
        sys.stdout.flush()


def run_all_tests(user_verbosity=2):
    unittest.main(verbosity=user_verbosity)


if __name__ == "__main__":
    run_all_tests(3)
