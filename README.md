# IBM&reg; Decision Optimization Modeling for Python (DOcplex)

Welcome to the IBM&reg; Decision Optimization Modeling for Python.
Licensed under the Apache License v2.0.

With this library, you can quickly and easily add the power of optimization to
your application. You need IBM ILOG CPLEX Optimization Studio to solve the models.

This library is composed of 2 modules:

* IBM&reg; Decision Optimization CPLEX Optimizer Modeling for Python - with namespace docplex.mp
* IBM&reg; Decision Optimization CP Optimizer Modeling for Python - with namespace docplex.cp

This library is numpy friendly.

## Prerequisites

- **Python**: 3.10 or (higher)

## Installation

- [DOcplex](https://pypi.org/project/docplex/) (Decision Optimization CPLEX Modeling for Python):

```bash
pip install docplex
```

- [Cplex](https://pypi.org/project/cplex/)

```bash
pip install cplex
```

The Community Edition is free and suitable for learning and small-to-medium problems. However, it has limitations on problem size (1000 variables, 1000 constraints).

**Note**: For larger problems or commercial use, you'll need the full version of CPLEX Optimization Studio or an academic license.

### Full Version Installation

If you have IBM ILOG CPLEX Optimization Studio installed, you can install the Python API from the installation directory:

```bash
# Navigate to your CPLEX installation directory
cd <CPLEX_INSTALL_DIR>/python

# Install the package
python setup.py install
```

If CPLEX is installed separately, configure DOcplex:

```
docplex config --upgrade <COS_INSTALL_DIR>
```

This ensures DOcplex correctly detects your CPLEX installation.


DOcplex provides a higher-level API and works with both CPLEX and IBM Decision Optimization on Cloud. See the [DOcplex examples repository](https://github.com/IBMDecisionOptimization/docplex/tree/master/examples) for more information.

## Building a Combined docplex + cplex Wheel

This section explains how to create a single Python wheel (.whl) that bundles both docplex and cplex packages, enabling simplified distribution and configuration.
This approach is intended for advanced users who need to package an existing Python environment (for example, a virtual environment containing cplex) into a single wheel.

By configuring [setup.cfg](https://github.com/IBMDecisionOptimization/docplex/blob/master/setup.cfg ), you can generate a single wheel that contains both packages.

```
 [options]
-       packages = find:
-       install_requires = six
+package_dir =
+       = <myPythonEnvironmentPath>/lib/python<myPythonEnvironmentVersion>/site-packages
+packages = find:
+python_requires = >=3.7
+install_requires = 
+       six
+
+[options.packages.find]
+where = <myPythonEnvironmentPath>/lib/python<myPythonEnvironmentVersion>/site-packages
+include = cplex*, docplex*
 
 [options.entry_points]
 console_scripts = 
        docplex = docplex.util.cli:main
```
### Building the Wheel

From the directory containing setup.cfg, run:
```
pip wheel .
```

The resulting wheel is fully compatible with:
```
docplex --config
```
which correctly detects and configures the bundled cplex.

## Get the documentation and examples

* [Latest documentation](http://ibmdecisionoptimization.github.io/docplex-doc/)
* Documentation archives:
   * [2.23.222](http://ibmdecisionoptimization.github.io/docplex-doc/2.23.222)
   * [2.22.213](http://ibmdecisionoptimization.github.io/docplex-doc/2.22.213)
   * [2.21.207](http://ibmdecisionoptimization.github.io/docplex-doc/2.21.207)
   * [2.20.204](http://ibmdecisionoptimization.github.io/docplex-doc/2.20.204)
   * [2.19.202](http://ibmdecisionoptimization.github.io/docplex-doc/2.19.202)
   * [2.18.200](http://ibmdecisionoptimization.github.io/docplex-doc/2.18.200)
   * [2.16.195](http://ibmdecisionoptimization.github.io/docplex-doc/2.16.195)
* [Examples](https://github.com/IBMDecisionOptimization/docplex-examples)

## Get your IBM&reg; ILOG CPLEX Optimization Studio edition

- You can get a free [Community Edition](https://www.ibm.com/account/reg/us-en/signup?formid=urx-20028)
 of CPLEX Optimization Studio, with limited solving capabilities in term of problem size.

- Faculty members, research professionals at accredited institutions can get access to an unlimited version of CPLEX through the
 [IBM&reg; Academic Initiative](http://ibm.biz/cplex-free-for-students).

## Dependencies

These third-party dependencies are automatically installed with ``pip``

- [futures](https://pypi.python.org/pypi/futures)
- [requests](https://pypi.python.org/pypi/requests)
- [six](https://pypi.python.org/pypi/six)
- [certifi](https://pypi.python.org/pypi/certifi)



## License

This library is delivered under the  Apache License Version 2.0, January 2004 (see LICENSE.txt).
