Changelog
---------

Changed in 2.30.251:
````````````````````
* Fix for SyntaxWarning: invalid escape sequence when running with Python version > 3.12.
* Replace the deprecated usage of 'strtobool' with Python >= 3.12
* modeller.constant(x) function creates an unhandled constant(x) in .cpo file
* docplex config --upgrade requires setuptools with Python >= 3.12
* docplex config --upgrade retrieves python executable
* Conda SafetyError on reinstalling with conda install cplex after a docplex config --upgrade
* C-style comments not allowed inside """ ... """ in some Python 3.12 versions
* Replace the deprecated usage of strtobool
* Fix export crash when an incorrect format is passed
* Report best bound in solve details even if no integer solution has been found

Changed in 2.29.245 (2025.04):
``````````````````````````````

* each record of a solve history use a unique timestamp for its fields


Changed in 2.28.240 (2024.08):
``````````````````````````````

* In ``docplex.cp``: robustify the handling of callback messages coming from CP Optimizer
* Allows to use numpy 2.0

Changed in 2.27.239 (2024.04):
``````````````````````````````

* Allows to use numpy 2.0

Changed in 2.27.239 (2024.04):
``````````````````````````````

* PEP517, PEP518 build
* Faster PWLs from `PR#6 <https://github.com/IBMDecisionOptimization/docplex/pull/6>`__.

Changed in 2.26.237 (2023.07):
``````````````````````````````

* Support for transactional output attachments

Changed in 2.25.236 (2023.01):
``````````````````````````````

* Support of .ops files for parameter handling.
* Improvement on the Nurse example.

Changed in 2.24.232 (2022.11):
``````````````````````````````

* Optional CPLEX import for qiskit
* Fix in cut getter

Changed in 2.24.231 (2022.11):
``````````````````````````````

* Support for CPLEX Optimization Studio 22.1.1 runtimes.
* Cleanup of the code related to Watson Machine Learning.
* In ``docplex.mp``:
   * Support for sensitivity analysis in Solution.
   * Fixes in Solution handling.
* In ``docplex.cp``:
  * Support for 'inferred' statement.
  * Support for 'sub_circuit' constraint.
  * Fixed conditional module importation in sched_jobshop_blackbox.py 

Changed in 2.23.222 (2022.03):
``````````````````````````````

* Support for CPLEX Optimization Studio 22.1 runtimes.
* In ``docplex.mp``:
    * Support of parameter sets for multi objective optimization.
    * Bug fix for multi objective optimization.
    * Correct documentation urls to current CPLEX offering.
    * Removal of the deprecated ``docloud_context``.
    * Performance improvements for model building with 22.1
* In ``docplex.cp``:
    * Added a new module blackbox to allow the new blackbox function feature
    * Wrote full documentation on blackbox functions
    * Added a new SearchType 'Neighborhood'
    * Fixed a bug in modeler.same_common_subsequence
    * Fixed problem of order of computation of actual solving parameters.


Changed in 2.22.213 (2021.09):
``````````````````````````````

* Removed zeppelin examples
* In ``docplex.mp``:
    * Added a variant of Model.sum() with variable number of arguments: Model.sums()
    * Removed all Python2 compatibility code
    * Added deterministic time in solve details.
    * Linear relaxer now relaxes SOS variable sets (linear sum relaxation)
    * Fixed a bug on Model.solve_with_goals() with quadratic sub-objectives
    * Fixed a bug in SolveSolution.kpi_value_by_name
    * Fixed a bug in SolveSolution.get_value_dict() about precision filtering
    * Fixed a bug when updating a constraint rhs with a NaN value now raises exception.

* In ``docplex.cp``:
    * Add new methods get_objective_value(), get_objective_bound() and get_objective_gap() on CpoModelSolution 
	  and CpoSolveResult to access directly the first objective, bound or gap.
    * Support real call to abort_search() instead of killing the solver.
    * Add a method get_parameters() to CpoSolver to retrieve actual solving parameters.
    * Fix problem of order of computation of actual solving parameters.


Changed in 2.21.207 (2021.06):
``````````````````````````````

* In ``docplex.mp``:
    * Add support for importing solution from a different model, provided variables
        can be matched between both models (Model.import_solution)
    * Added contextual function to temporarily modify aspects of the model:
        `model_parameters` to change parameters in a block, `model_objective` to set a temporary objective in a block.
    * Naming expressions is now deprecated, use a temporary variable if needed.
    * Display of expressions in constraints is customizable: use a space separator (or not),
        set a maximum length for very long expression.
    * Fixed documentation references to class Var, moved from linear to dvar
    * Fixed a bug in LP export for multi-objective models with no priorities passed

* In ``docplex.cp``:
    * Replace context attribute solve_with_start_next by solve_with_search_next, keeping ascending compatibility.
    * Support solver parameter ModelAnonymizer to generate random names for all model elements (except KPIs) in the CPO file format.
    * Optimize generation of CPO expressions from Python expressions
    * Interval variable solutions tuples are now named tuples.
    * Method CpoModel.add() now accepts multiple expressions (or lists of expressions) to add to the model.

    
Changed in 2.20.204 (2021.02):
``````````````````````````````

* In ``docplex.mp``:
    * Add support for exporting solutions and solution pools to SOL format.
    * Add fast methods for changing batches of variable bounds:
        Model.change_var_lower_bounds, Model.change_var_upper_bounds
    * Reset random seed value for cplex 12.10 , was different from COS release value.
    * Improved performance of variable creation
    * Removed a warning about accessing a deprecated "solve_status" in solve.
    * docplex.mp.AdvModel now has checker enabled by default to avoid Python errors.
       Is is up to the user to disable type-checking to get maximum performance.
    * Fixed a bug about not printing updated variable bounds in MPS and SAV
    * Changed the default rounding behavior: solution values are not rounded by default.

* In ``docplex.cp``:
    * Fix problems on step function operations.
    * Add status in conflict refiner solution object.
    * Fix a bug to not send and receive conflict in CPO format if no conflict.
    * Enhance printing of model solution.
     
    * The method search_next(), returns only new solutions of the model.
      In particular, no new solution is returned if solve status just change from Feasible to Optimal.
    * Rework completely model statistics (CpoModelStatistics)
    * Add a parameter add_cpo to the write method of CpoRefineConflictResult


Changed in 2.19.202 (2020.12):
``````````````````````````````

* In ``docplex.mp``:
    * Exporting models to SAV.gz format is now supported.
    
* In ``docplex.cp``:
    * Add a new method remove_expressions() to CpoModel that removes a list of expressions in one shot.
    * Update documentation to describe how to set private solver parameters.
    * Add functions to export and import context as flat file.
    * Fix a bug on the string representation of calls to constant() modeling function.
    * Add JSON filter in case of double identical value for objective.
    * Add functions to export and import context as flat file.
    * Fix a bug on the string representation of calls to constant() modeling function.
    * Add JSON filter in case of double value for objective.

    
Changed in 2.18.200 (2020.11.#3):
`````````````````````````````````

* In ``docplex.mp``:
    * Latest supported CPLEX Optimization Studio is now 20.1


Changed in 2.17.196 (2020.11.#2):
`````````````````````````````````

* In ``docplex.mp``:
    * Fixed a bug with pickling: edition of a constraint in a pickled model raised an error
    * Fixed a bug with pickling: models with piecewise-linear constraints could not be pickled

* In ``docplex.cp``:
    * Add environment variable DOCPLEX_CP_CONTEXT to modify configuration
    * Add new module check_list that print a report on execution environment
    * Remove DOcloud from documentation (including code)
    * Rework customization of configuration and better support of default directory
    * Add new configuration parameter model.sort_names to drive sort of variable declarations in CPO file format.
    * Fix a problem that may crash Python in case of abort_search with local solve.

    
Changed in 2.16.196 (2020.11):
``````````````````````````````

* In ``docplex.mp``:
    * add `Model.add_quadratic_constraints()` to add a batch of quadratic constraints
    * add `Model.populate_solution_pool()` for a native support of solurtion pools
    * support of CPLEX 20.1
    * compatible with Python 3.8 (only with CPLEX 20.1)
    * enable changing absolute and relative tolerances for  multi-objectives
    * Optimization of `Model.if_then`: when condition is of the form `b==1` (or 0), no additional boolean
      variable is generated
    * For solving, `docplex.mp` now uses the `cplex` module if it has been installed. If not, `docplex.mp`
      checks for the latest installed version of CPLEX Optimization Studio (COS) (using the CPLEX_STUDIO_DIRXXX
      environment variables) and use the `cplex` if a COS is found, unless the configuration of the engine
      states otherwise.

* In ``docplex.cp``:
    * Add pngfile= parameter to visu.show() method to store in a PNG file instead of display on screen.
    * Make parameters and solver infos also available in refine_conflict solution.
    * Add a IntegerDomain class used to represent domain of integer variables, with a specific __str__ method
    * Add new functions ceil(), floor(), trunc(), round() and sgn()
    * Remove all warnings generated by Python 38
    * Optimize creation of add expressions with CumulExpr and zero
    * Implement configurable factorization of common model expressions when generating CPO format
    * Add checking of scal_prod() array sizes at modeling time
    * Add strict_lexicographic() and checking of strict_lexicographic() and lexicographic() array sizes at modeling time.
    * Add failure explanation as new method explain_failure() allowing to log failure tags or get details on one or several failures.
    * Enhance management of local solve sub-process timeout with detailed error and configurable timeout delay
    * For solving, `docplex.cp` now uses the `cpoptimizer` executable if it has been installed. If not, `docplex.cp`
      checks for the latest installed version of CPLEX Optimization Studio (COS) (using the CPLEX_STUDIO_DIRXXX
      environment variables) and use the `cpoptimizer` if a COS is found, unless the configuration of the engine
      states otherwise.
    * Support last optimal solution in search/next sequence
    * Support of solver parameters in all next() solutions
    * Add solver version in process info attached to a run result


Changed in 2.15.194 (2020.07):
``````````````````````````````

* In ``docplex.mp``:
    * add Model.quadratic_dual_slacks()
    * Fixed a bug in multi-objectives: objectives were incorrectly rounded
    * Fixed a bug in Model.report(): multiple objective values were not displayed()
* In ``docplex.cp``:
    * Add conflict in CPO format in refine conflict result
    * Fix problem when parsing KPIs section of a CPO model
    * Add method add_constraint() to model for compatibility with docplex.mp
    * Comment method get_fail_status() of SolveResult as deprecated.
    * Fix problem of wrong import of deque in collections.abc


Changed in 2.14.186 (2020.05):
``````````````````````````````

* Updated tracking events in Watson studio notebooks.
* In ``docplex.mp``:
    * Model.solve() will not use solve on cloud unless `agent` is specifically set to 'docloud`.

    
Changed in 2.13.184 (2020.03):
``````````````````````````````

* Removed dependency to the `docloud` package. Now you need to explicitely install the package using `pip install docloud` to use DOcplexcloud.
* In ``docplex.mp``:
    * added Model.export_as_mps_string(), Model.export_as_sav_string()
    * fixed a bug with dettime_limit: solving with a deterministic time limit
       was mis-interpreted as a solve failure, returning None.
    * fixed bug on cplexcloud solve: number of nodes processed was always zero.
    * repeated solves incorrectly restarted from start of search, now start from where the last solve stopped.
    * added keyword argument 'time_limit' to Model.solve() to set a temporary time limit.
    * added new method SolveSolution.is_valid_solution()
    * fixed a bug in ModelReader: ranged constraints bounds were inverted when reading from SAV or MPS.
    * fixed a bug in Model.set_lex_multiobj(): arguments abstols, reltols were ignored.
    * added proper type-checking for Model.add_indicator_constraints()
    * added docplex.mp.check_list/py to check local installation.
* In ``docplex.cp``:
    * Enable reading of #line directives when parsing a CPO file
    * Remove parameter LogSearchTags from public parameters
    * Fix a minor problem concerning compilation of KPI expressions in CPO format


Changed in 2.12.182 (2019.12):
``````````````````````````````
* In ``docplex.mp``:
    * Added a LinearRelaxer class to make a linearized copy of a MIP model (if possible).
      see class `docplex.mp.relax_linear.LinearRelaxer`
    * Conflict refiner default behavior is now identical to CPLEX interactive
      (the new behavior is much faster).
    * Bug fixed: expressions of the form k*x did not notify constraints when modified.
    * Fixed: message "ignored keyword argument" was incorrectly printed when setting
      `cts_by_name=True` in model constructor.


Changed in 2.11.176 (2019.11):
``````````````````````````````

* Added support for CPLEX 12.10
* In ``docplex.mp``:
   * Logical expressions, binary variables, and constraints can now be freely nested with logical operators.
   * Fixed a print of 'CPLEX Error 1217' in log for multi-objective problems.
   * Fixed a bug when setting log_output to a file name: file was created, but empty.
    

Changed in 2.10.155 (2019.08):
``````````````````````````````

* Fixed bug in logical_and() when result var is set to 0.

Changed in 2.10.154 (2019.07):
``````````````````````````````

* Fixed TypeError occuring in python 3.7 in progressData initialization.

Changed in 2.10.151 (2019.07):
``````````````````````````````

* In ``docplex.mp``:
   * fixed a bug in ModelReader when reading SAV files with no names
   * fixed a bug in mip starts, which prevented mip starts with piecewise
     functions to work properly.
   * fixed bug on Model.add_indicators() using comprehensions (len() was called).
   * Added support for the '!=' (not equals) operator in expressions.
   * Clarified four types of checker: on, off, numeric and full.
     Pass checker=<name> at model creation to specify which checker is used.
   * fixed a bug in solution JSON encoder for nonconvex QP problems.
   * Add direct support for lazy constraints, see Model.add_lazy_constraints()
   * Add direct support for user cuts, see Model.add_user_cut_constraints()
   * Get basis status of variables in LP problems, see `Var.basis_status`
   * Read MIP start files (MST format)
   * Allow to set the effort level for a MIP solution.
   * Read basis status files (in BAS format)
   * Read variable priority orders (in ORD format)
   * fixed bug in functional KPIs, solution argument was not passed on.
   * Enable constraint name dictionary at Model creation time: Model(cts_by_name=True)
   * Multi-objective is now pickled correctly
   * Multi-objective is now copied in Model.copy()
   * Wrote full documentation on progress listeners
   * Added `Model.set_lp_start_basis()` to provide an initial basis for LP problems.

* In ``docplex.cp``:
   * When trying to access a solution member that does not exists, an exception is thrown instead of returning None.
   * Add a new map_solution function that replace in a Python object all model expressions by their value in a solve result.
   * In CPO parser, fix a problem reading #line statements in startingPoint section.
   * In CPO parser, skip experimental section 'expressions' in 'startingPoint' section.
   * Simplify writing of interval variable domains reduced to a single value.
   * Adding a second objective function now raises an exception
   * Add new experimental local solve with a shared library.
   * Enable iterators to specify the domain of an integer variable
   * Add global methods get_version_info() and get_solver_verion() in docplex.cp.solver.
   * By default, generate CPO model without explicit format version.
   * Add a method reset() on CpoParameters object.
   * Modeling method allowed_assignments() and forbidden_assignments() can now accept an empty list of tuples.
   * On CpoModelSolution object, add a function map_solution() thar replace variables by their value in a python object.
   * Add parser for LP models
   * Add possibility to import CPO, MZN and LP models in gzip and zip format.
   * Enhance management of unexpected errors thrown by cpoptimizer.exe


Changed in 2.9.141 (2019.03):
`````````````````````````````

* In ``docplex.mp``:
   * Removed links to rawgit.com as this service is going end of life.
   * Model.solve_lexicographic() is deprecated. This method should be used
     to perform lexicographic solve with COS 12.8, but with COS 12.9,
     Model.set_multi_objective() should be used for solving problems
     with multiple objectives.

* In ``docplex.cp``:
   * Add KPIs supported by CPO Solver 12.9
   * Update CPO parser to read KPIs section for format 12.9
   * Add new examples with KPIs.


Changed in 2.8.125 (2018.10):
`````````````````````````````
* Solving with solver agent 'docloud' is deprecated. 
  Models are now preferably solved with local solver, or the python source can be submitted to DOcplexcloud solve service. 
  See https://ibm.biz/BdYhhK.

* In ``docplex.mp``:
   * solve_lexicographic is being deprecated. In a future version, a new api will be available to support multi-objectives.

* In ``docplex.cp``:
   * Fix problem with boolean indicators in no_overlap(), always_constant() and always_equal().
   * Allow model solution to be used directly as a starting point (ignores what is not integer or interval var).
   * Add methods domain_min(), domain_max(), domain_iterator() and domain_contains() on both CpoIntVar and CpoIntVarSolution.
   * Default solver agent is now 'local' instead of 'docloud'. All examples modified consequently.


Changed in 2.7.113 (2018.07):
`````````````````````````````

* In ``docplex.mp``:
   * Multiplying a constant expression by a quadratic expression raised an exception. Now returns the
     product of the quadratic expression and the constant value.
   * Model.solve_lexicographic() on cloud now send the previous pass solution as a MIP start (for MIP problems)
   * The slack of quadratic constraints always returned  zero. Now returns the correct value.
   * Accessing the dual (or slack) of a constraint that is not added to the model returned zero; now it raises an exception. A constraint must belong to a model to return a valid dual (or slack) value
   * Range constraints with infeasible domain (i.e. lb > ub) did not fail to solve. Now they raise a modeling exception.
   * Multiplying two absolute value expressions raised an exception. Now fixed.
   * When using tuples in variable dictionaries, the default name generation used to generate non-LP-compliant names,
     because of ( and ). Now the name generator formats the tuples with a "_" separator without parentheses.

* In ``docplex.cp``:
   * Split fzn stuff in a separate package docplex.cp.fzn
   * Optimize construction of arrays in FZN parser
   * Enhance FZN parser and save 30% time


Changed in 2.6.94 (2018.04):
````````````````````````````

* In ``docplex.cp``:
   * Allow CpoModel.add() to accept list of constraints.
   * Fix a bug in the conversion of an array of boolean constants into CPO expression.
   * Extend CpoModel method set_parameters() to accept a dictionary and/or optional list of updates using named arguments.
   * Method CpoModel.set_parameters() now clone the CpoParameters object given in arguments.
   * Add a new method CpoModel.add_parameters() that updates parameters associated to the model.
   * Fix wrong source location (not in real model source) when CpoModel.add() is called from another docplex.cp method.
   * When constraint auto-naming is on (in particular for refine_conflict(), searchPhases are no more included in the process.
   * Parameters mean_UB and mean_LB are now optional in standard_deviation()
   * CpoModel.add() checks that the added expression is limited to constraint, boolean, objective or search phase.
   * Add documented functions slope_piecewise_linear() and coordinate__piecewise_linear() in modeler.py.
   * Remove default configuration settings for parameters TimeLimit and Workers.


Changed in 2.5.92 (2018.03):
````````````````````````````

* ``docplex.cli`` gains new features:
   * option ``--details`` will display solve details as they are published on
     DOcplexcloud.
   * options ``--url`` and ``-key`` allow specification of credentials without
     using a config file.

* In ``docplex.cp``:
   * Fix problem with min() and max() that did not support optional key.
   * Add a Flatzinc parser capable of reading Minizinc Challenge problems.
   * Move expression dependencies analysis from model to compiler side.
   * No more constraint to have a unique name for model expressions. Compiler reallocate private names when needed.
   * Multiple variables or expressions with the same public name is now allowed.
   * Replace method CpoModel.get_expression() by CpoModel.get_named_expressions_dict().
   * Make SolverProgressPanelListener work properly with Python 2
   * Solve is automatically set to start/next loop when SolverProgressPanelListener is used.
   * In CpoModel, add a method that allows to substitute a function by another in the whole model.
   * Overwrite method __bool__ to avoid accidental use of CPO expressions as Python booleans.
   * Add special cases to search for the local CP Optimizer Interactive executable.
   * Allow methods min(), max(), min_of() and max_of() to support variable number of arguments.
   * Allow method all_diff() to support variable number of arguments.
   * Context parameter 'length_for_rename' is deprecated. Only length_for_alias is used.
   * Add a method add_var() in CpoModelSolution as a shortcut to add_integer_var_solution() and add_interval_var_solution()
   * Overwrite method __contains__() in CpoModelSolution to easily verify that a solution to a given variable is in the solution.
   * When called on a model, export_model() and get_cpo_string() disable all model optimization options.


Changed in 2.4.61 (2017.11):
````````````````````````````

* Both ``docplex.mp`` & ``docplex.cp``:
   * Support for CPLEX engines 12.8. Some features of docplex2.4 are available only with engines >= 12.8.
   * Adding new ports (AIX, plinux).
   * Examples are now available as Zeppelin notebooks.

* In ``docplex.mp``:
   * Express a linear problem as a scikit-learn transformer by providing a numpy, a pandas or scipy matrix.
   * Logical constraints: constraint equivalence, if-then & rshift operator.
   * Meta-constraints: allow the use of discrete
     linear constraints in expressions, using their truth value.
   * Solve hook to add a method to be called at each intermediate solution.
   * KPIS automatically published at each intermediate solution if running on docplexcloud python worker.
   * Support for scipy coo & csr matrixes.
   * Fixed a bug in Model.add_constraints() when passing a string instead of a list of strings.

* In ``docplex.cp``:
   * add new method run_seeds() to execute a model multiple times, available with local solver 12.8.
   * add support of new solver infos 'SearchStatus' and 'SearchStopCause'.
   * In method ``docplex.cp.model.CpoModel.propagate()``, add possibility to add an optional constraint to the model.
   * add domain iterator in integer variables and integer variables solutions, allowing to get domain
     as a list of individual integers.
   * add possibility to identify some model variables as KPIs of the model.
   * add abort_search() method on solver (not supported everywhere)
   * Rework code generation to enhance performances and remove unused variables that was pointed by removed expressions.
   * add possibility to add one or more CpoSolverListener to put some callback functions
     when solve is started, ended, or when a solution is found.
     Implementation is provided in new python module ``docplex.cp.solver.solver_listener`` that also contains sample
     listeners SolverProgressPanelListener and AutoStopListener.
   * Using parameter *context.solver.solve_with_start_next*, enable solve() method to execute a start/next loop instead
     of standard solve. This enables, for optimization problems, usage of SolveListeners with a greater progress accuracy.
   * Completely remove deprecated 'angel' to identify local solver.
   * Deprecate usage of methods ``minimize()`` and ``maximize()`` on ``docplex.cp.CpoModel``. 
   * Add methods ``get_objective_bounds()`` and ``get_objective_gaps()`` in solution objects.

  
  
Changed in 2.3.44 (2017.09):
````````````````````````````

* Module ``docplex.cp.model.solver_angel.py`` has been renamed ``solver_local.py``. 
  A shadow copy with previous name still exist to preserve ascending compatibility.
  Module ``docplex.cp.model.config.py`` is modified to refer this new module.
* Class ``docplex.cp.model.solver_local.SolverAngel`` has been renamed ``SolverLocal``. 
  A shadow copy with previous name still exist to preserve ascending compatibility.
* Class ``docplex.cp.model.solver_local.AngelException`` has been renamed ``LocalSolverException``. 
  A shadow copy with previous name still exist to preserve ascending compatibility.
* Functions logical_and() and logical_or() are able to accept a list of model boolean expressions.
* Fix defect on allowed_assignments() and forbiden_assignments() that was wrongly converting 
  list of tupes into tuple_set.
* Update all examples to add comments and split them in sections data / prepare / model / solve
* Add new sched_RCPSPMM_json.py example that reads data from JSON file instead of raw data file.
* Rename all visu examples with more explicit names.
* Remove the object class CpoTupleSet. Tuple sets can be constructed only by calling tuple_set() method, or more
  simply by passing directly a Python iterable of iterables when a tupleset is required 
  (in expressions allowed_assignments() and forbidden_assignments)
* Allow logical_and() and logical_or() to accept a list of boolean expressions.
* Add overloading of builtin functions all() and any() as other form of logical_and() and logical_or().
* In no_overlap() and state_function(), transition matrix can be passed directly as a Python iterable of iterables of integers, 
* Editable transition matrix, created with a size only, is deprecated. However it is still available for ascending compatibility.
* Add conditional() modeling function
* Parameter 'AutomaticReplay' is deprecated.
* Add get_search_status() and get_stop_cause() on object CpoSolveResult, available for solver COS12.8
* Improved performance of ``Var.reduced_cost()`` in ``docplex.mp``.

Changed in 2.2.34 (2017.07):
````````````````````````````

* Methods ``docplex.cp.model.export_model()`` and ``docplex.cp.model.import_model()``
  have been added to respectively generate or parse a model in CPO format.
* Methods ``docplex.cp.model.minimize()`` and ``docplex.cp.model.maximize()``
  have been added to directly indicate an objective at model level.
* Notebook example ``scheduling_tuto.ipynb`` contains an extensive tutorial
  to solve scheduling problems with CP.
* Modeling method sum() now supports sum of cumul expressions.
* Methods ``docplex.cp.model.start_search()`` allows to start a new 
  search sequence directly from the model object.
* When setting ``context.solver.auto_publish`` is set, and using the CPLEX
  engine, KPIs and current objective are automatically published when the
  script is run on DOcplexcloud Python worker.
* When setting ``context.solver.auto_publish`` is set, and using the CP
  engine, current objective is automatically published when the
  script is run on DOcplexcloud Python worker.
* ``docplex.util.environment.Environment.set_stop_callback`` and
  ``docplex.util.environment.Environment.get_stop_callback`` are added so that
  you can add a callback when the DOcplexcloud job is aborted.


Changed in 2.1.28:
``````````````````

* New methods ``Model.logical_or()`` and ``Model.logical_and()`` handle
  logical operations on binary variables.
* DOcplex now supports CPLEX 12.7.1 and Benders decomposition. Set annotations
  on constraints and variables using the ``benders_annotation`` property and use
  the proper CPLEX parameters governing Benders decomposition.
* CPLEX tutorials: in the documentation and as notebooks in the examples.
* Fixed a bug in ``docplex.mp.solution.SolveSolution.display()`` and in 
  ``docplex.mp.solution.Model.report_kpi()`` when using unicode variable names.
* There's now a simple command line interface for DOcplexcloud. It can be run
  in a terminal. ``python -m docplex.cli help`` for more info. That command
  line reads your DOcplexcloud credentials in your cplex_config.py file. It
  allows you to submit, list, delete jobs on DOcplexcloud. The cli is available
  in notebooks too, using the ``%docplex_cli`` magics. ``%docplex_cli help`` for
  some help. In a notebook, credentials can be passed using `%docplex_url` and
  `%docplex_key` magics.
* Removing constraints in 1 call
* Bug fixes when editing an existing model.
* Bug fix in the relaxation mechanism when using docplexcloud.


Changed in 2.0.15:
``````````````````

* Piecewise linear (PWL) functions are now supported. An API is now available
  on ``docplex.mp.model`` to create PWL functions and to create constraints using these PWL functions.
  PWL functions may be defined with breakpoints (default API) or by using slopes. Some simple arithmetic is
  also available to build new PWL functions by adding, subtracting, or scaling existing PWL functions.
* DOcplex has undergone a significant overhaul effort that has resulted in an average of 30-50% improvement
  of modeling run-time performance. All parts of the API benefit from the performance improvements: creation of variables and constraints, removal of constraints, computation of sums of variables, and so on.
* Constraints are now fully editable: 
  the expressions of a constraint can be modified.
  Similarly, the objective expression can also be modified. This allows for complex workflows in which the model is modified after a solve and then solved again. 
* docplex is now available on Anaconda cloud and can be installed via the conda installation packager.
  See the `IBM Anaconda home <https://anaconda.org/IBMDecisionOptimization>`_
  CPLEX Community Edition for Python is also provided on Anaconda Cloud to get free local solving capabilities with limitations.
* Support of ``~/.docplexrc`` configuration files for ``docplex.mp.context.Context`` is now dropped.
  This feature has been deprecated since 1.0.0.
* Known incompatibility: class ``docplex.mp.model.AbstractModel`` moved to ``docplex.mp.absmodel.AbstractModel``. 
  Samples using this class have been updated.


Changed in 1.0.630:
```````````````````

* Added support for CPLEX 12.7 and Python 3.5.
* Upgraded the DOcplexcloud client to version 1.0.202.
* Module ``docplex.mp.advmodel`` is now officially supported. This module
  provides support for efficient, specialized aggregator methods for large
  models.
* When solving on DOcplexcloud, proxies can now be specified with the
  ``context.solver.docloud.proxies`` property.
* When two constraints are defined with the same name, issue a warning instead of
  a fatal exception. The last constraint defined will take over the first one in the name directory.
* Fix ValueError when passing a pandas DataFrame as variable keys (using
  DataFrame indexes).
* Solution.get_values() returns a collection of variable values in one call.
* ``docplex.mp.model`` no longer imports ``docloud.status``. Any status
  previously initialized as ``JobSolveStatus.UNKNOWN`` is now initialized as
  ``None``.
* Minor improvements to notebooks and examples.
