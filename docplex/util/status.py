# --------------------------------------------------------------------------
from enum import Enum
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2018
# --------------------------------------------------------------------------

# gendoc: ignore

'''
  class JobSolveStatus(Enum):
        """ Job solve status values.

        This `Enum` is used to convert job solve status string values into an
        enumeration::

            >>> job = client.get_job(jobid)
            >>> solveStatus = JobSolveStatus[job['solveStatus']]

        Attributes:
            UNKNOWN: The algorithm has no information about the solution.
            FEASIBLE_SOLUTION: The algorithm found a feasible solution.
            OPTIMAL_SOLUTION: The algorithm found an optimal solution.
            INFEASIBLE_SOLUTION: The algorithm proved that the model is infeasible.
            UNBOUNDED_SOLUTION: The algorithm proved the model unbounded.
            INFEASIBLE_OR_UNBOUNDED_SOLUTION: The model is infeasible or unbounded.
        """
        UNKNOWN = 0
        FEASIBLE_SOLUTION = 1
        OPTIMAL_SOLUTION = 2
        INFEASIBLE_SOLUTION = 3
        UNBOUNDED_SOLUTION = 4
        INFEASIBLE_OR_UNBOUNDED_SOLUTION = 5
'''
class JobSolveStatus(Enum):
    """ Job solve status values.

    This `Enum` is used to convert job solve status string values into an
    enumeration::

        >>> job = client.get_job(jobid)
        >>> solveStatus = JobSolveStatus[job['solveStatus']]

            Attributes:
                UNKNOWN: The algorithm has no information about the solution.
                FEASIBLE_SOLUTION: The algorithm found a feasible solution.
                OPTIMAL_SOLUTION: The algorithm found an optimal solution.
                INFEASIBLE_SOLUTION: The algorithm proved that the model is infeasible.
                UNBOUNDED_SOLUTION: The algorithm proved the model unbounded.
                INFEASIBLE_OR_UNBOUNDED_SOLUTION: The model is infeasible or unbounded.
            """
    UNKNOWN = 0
    FEASIBLE_SOLUTION = 1
    OPTIMAL_SOLUTION = 2
    INFEASIBLE_SOLUTION = 3
    UNBOUNDED_SOLUTION = 4
    INFEASIBLE_OR_UNBOUNDED_SOLUTION = 5
