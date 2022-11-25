# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2020, 2022
# --------------------------------------------------------------------------

from contextlib import contextmanager

from docplex.mp.constants import ObjectiveSense


@contextmanager
def model_parameters(mdl, temp_parameters):
    """ This contextual function is used to override a model's parameters.
    As a contextual function, it is intended to be used with the `with` construct, for example:

    >>> with model_parameters(mdl, {"timelimit": 30, "empahsis.mip": 4}) as mdl2:
    >>>     mdl2.solve()


    The new model returned from the `with` has temporary parameters overriding those of the initial model.

    when exiting the with block, initial parameters are restored.

    :param mdl: an instance of `:class:Model`.
    :param temp_parameters: accepts either a dictionary of qualified names to values, for example
        {"mip.tolernaces.mipgap": 0.03, "emphasis.mip": 4}, or a dictionary from parameter objects to values.
    :return: the same model, with overridden parameters.

    See Also:
        - :func:`docplex.mp.params.Parameter.qualified_name`

    *New in version 2.21*
    """
    if not temp_parameters:
        try:
            yield mdl
        finally:
            pass
    else:
        ctx = mdl.context
        saved_context = ctx
        temp_ctx = ctx.copy()
        try:
            temp_ctx.update_cplex_parameters(temp_parameters)
            mdl.context = temp_ctx
            yield mdl
        finally:
            mdl.context = saved_context
            return mdl


@contextmanager
def model_objective(mdl, temp_obj, temp_sense=None):
    """ This contextual function is used to temporarily override the objective of a model.
    As a contextual function, it is intended to be used with the `with` construct, for example:

    >>> with model_objective(mdl, x+y) as mdl2:
    >>>     mdl2.solve()


    The new model returned from the `with` has a temporary objective overriding the initial objective.

    when exiting the with block, the initial objective and sense are restored.

    :param mdl: an instance of `:class:Model`.
    :param temp_obj: an expression.
    :param temp_sense: an optional objective sense to override the model's. Default is None (keep same objective).
        Accepts either an instance of enumerated value `:class:docplex.mp.constants.ObjectiveSense` or a string
        'min' or 'max'.
    :return: the same model, with overridden objective.

    *New in version 2.21*
    """
    saved_obj = mdl.objective_expr
    saved_sense = mdl.objective_sense
    new_sense_ = ObjectiveSense.parse(temp_sense, mdl) if temp_sense is not None else None

    try:
        mdl.set_objective_expr(temp_obj)
        if new_sense_:
            mdl.set_objective_sense(new_sense_)

        yield mdl
    finally:
        mdl.set_objective_expr(saved_obj)
        if new_sense_:
            mdl.set_objective_sense(saved_sense)


@contextmanager
def model_solvefixed(mdl):
    """ This contextual function is used to temporarily change the type of the model
    to "solveFixed".
    As a contextual function, it is intended to be used with the `with` construct, for example:

    >>> with model_solvefixed(mdl) as mdl2:
    >>>     mdl2.solve()

    The  model returned from the `with` has a temporary problem type set to "solveFixex overriding the
    actual problem type.
    This function is useful for MIP models which have been successfully solved; the modified model
    can be solved as a LP, with all discrete values fixed to their solutions in the previous solve.

    when exiting the with block, the actual problem type is restored.

    :param mdl: an instance of `:class:Model`.

    :return: the same model, with overridden problem type.

    Note:
        - an exception is raised if the model has not been solved
        - LP models are returned unchanged, as this mfunction has no use.

    *New in version 2.22*
    """
    cpx = mdl._get_cplex(do_raise=True, msgfn=lambda: "model_solvefixed requires CPLEX runtime")

    # save initial problem type, to be restored.
    saved_problem_type = cpx.get_problem_type()
    if saved_problem_type == 0:
        mdl.warning("Model {0} is a LP model, solvefixed does nothing".format(mdl.name))
        return mdl

    if mdl.solution is None:
        # a solution is required.
        mdl.fatal(f"model_solvefixed requires that the model has been solved successfully")
    try:
        cpx.set_problem_type(3)  # 3 is constant fixed_MILP
        yield mdl
    finally:
        cpx.set_problem_type(saved_problem_type)
