import os
import re
from distutils.core import setup
import sys

required = ['six']

HERE = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    try:
        with open(os.path.join(HERE, *parts)) as f:
            return f.read()
    except:
        return None

readme = read('README.rst')
if readme is None:
    readme = 'DOcplex 2.24'

changelog = str(read('CHANGELOG.rst'))
if changelog is None:
    changelog = ''

ss = str(readme) + str(changelog)

packages=['docplex',
          'docplex.cp',
          'docplex.cp.solver',
          'docplex.cp.cpo',
          'docplex.cp.fzn',
          'docplex.cp.lp',
          'docplex.mp',
          'docplex.mp.callbacks',
          'docplex.mp.internal',
          'docplex.mp.params',
          'docplex.mp.sktrans',
          'docplex.mp.sparktrans',
          'docplex.mp.worker',
          'docplex.util',
          'docplex.util.dods',
          'docplex.util.ws']

if os.path.isdir(os.path.join("docplex", "worker")):
    packages.append('docplex.worker')

setup(
    name='docplex',
    packages=packages,
    version = '2.24.231',  # replaced at build time
    description = 'The IBM Decision Optimization CPLEX Modeling for Python',
    author = 'The IBM Decision Optimization on Cloud team',
    author_email = 'dofeedback@wwpdl.vnet.ibm.com',
    long_description='%s\n' % ss,
    long_description_content_type='text/x-rst',
    url = 'https://www.ibm.com/cloud/decision-optimization-for-watson-studio',
    keywords = ['optimization', 'cplex', 'cpo'],
    license = 'Apache 2.0',
    install_requires=required,
    classifiers = ["Development Status :: 5 - Production/Stable",
                   "Intended Audience :: Developers",
                   "Intended Audience :: Information Technology",
                   "Intended Audience :: Science/Research",
                   "Operating System :: Unix",
                   "Operating System :: MacOS",
                   "Operating System :: Microsoft",
                   "Operating System :: OS Independent",
                   "Topic :: Scientific/Engineering",
                   "Topic :: Scientific/Engineering :: Mathematics",
                   "Topic :: Software Development :: Libraries",
                   "Topic :: System",
                   "Topic :: Other/Nonlisted Topic",
                   "License :: OSI Approved :: Apache Software License",
                   "Programming Language :: Python",
                   "Programming Language :: Python :: 3.6",
                   "Programming Language :: Python :: 3.7",
                   "Programming Language :: Python :: 3.8",
                   "Programming Language :: Python :: 3.9",
                   "Programming Language :: Python :: 3.10"
                   ],
)

print("** The documentation can be found here: http://ibmdecisionoptimization.github.io/docplex-doc/")
print("** The examples can be found here: https://github.com/IBMDecisionOptimization/docplex-examples")
