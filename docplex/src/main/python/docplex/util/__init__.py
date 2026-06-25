# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2022
# --------------------------------------------------------------------------

# gendoc: ignore


def as_df(what, **kwargs):
    '''
    Returns a `pandas.DataFrame` representation of an object.

    Attributes:
        what: The object to represent as an object.
        **kwargs: Additional parameters for the conversion.
    Returns:
        A `pandas.DataFrame` representation of an object or None if a
        representation could not be found.
    '''
    try:
        return what.__as_df__(**kwargs)
    except AttributeError:
        return None


class LazyEvaluation(object):
    def __init__(self, func):
        self._func = func

    def __str__(self):
        return self._func()


def lazy(f):
    '''
    Sometimes, we want to use some `f(x)` where `f` can be anything from a function that
    does nothing to a function that prints things. However, `x` can be a time consuming
    operation, so we don't always want x to be evaluated. Typically, loggers:

       logger.info(f"Doing something to n: {json.dumps(n)}")

    In the example above, it would be nice that the expression is not evaluated if
    the logger is a dummy logger.

    `lazy()` allows this to be rewritten as:

       logger.info(lazy(lambda: f"Doing something to n: {json.dumps(n)}"))

    '''
    return LazyEvaluation(f)
