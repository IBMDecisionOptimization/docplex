# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2020, 2022
# --------------------------------------------------------------------------
import json


def is_in_docplex_worker():
    try:
        import docplex.util.environment as runenv
        is_in_worker = isinstance(runenv.get_environment(), runenv.WorkerEnvironment)
    except:
        is_in_worker = False
    return is_in_worker


def make_new_kpis_dict(allkpis=None, int_vars=None, continuous_vars=None,
                       linear_constraints=None, bin_vars=None,
                       quadratic_constraints=None, total_constraints=None,
                       total_variables=None):
    # This is normally called once at the beginning of a solve
    # those are the details required for docplexcloud and DODS legacy
    kpis_name= [ kpi.name for kpi in allkpis ]
    kpis = {'MODEL_DETAIL_INTEGER_VARS': int_vars,
            'MODEL_DETAIL_CONTINUOUS_VARS': continuous_vars,
            'MODEL_DETAIL_CONSTRAINTS': linear_constraints,
            'MODEL_DETAIL_BOOLEAN_VARS': bin_vars,
            'MODEL_DETAIL_KPIS': json.dumps(kpis_name)}
    # those are the ones required per https://github.ibm.com/IBMDecisionOptimization/dd-planning/issues/2491
    new_details = {'STAT.cplex.size.integerVariables': int_vars,
                   'STAT.cplex.size.continousVariables': continuous_vars,
                   'STAT.cplex.size.linearConstraints': linear_constraints,
                   'STAT.cplex.size.booleanVariables': bin_vars,
                   'STAT.cplex.size.constraints': total_constraints,
                   'STAT.cplex.size.quadraticConstraints': quadratic_constraints,
                   'STAT.cplex.size.variables': total_variables,
                   }
    kpis.update(new_details)
    return kpis