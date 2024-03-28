# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2019, 2020
# --------------------------------------------------------------------------

# gendoc: ignore

from docplex.mp.publish import auto_publishing_kpis_table_names, \
    auto_publishing_result_output_names
from docplex.util.environment import get_environment, make_new_kpis_dict
from docplex.mp.context import is_auto_publishing_solve_details

from docplex.mp.publish import write_kpis_table, write_result_output

from docplex.mp.utils import apply_thread_limitations

try:
    import numpy
except ImportError:
    numpy = None


class SolveEnv(object):

    def __init__(self, mdl):
        self._model = mdl

    def before_solve(self, context):
        pass

    def after_solve(self, context, solve_res, engine):
        mdl = self._model
        mdl._set_solution(solve_res)

        solve_details = engine.get_solve_details()
        mdl._notify_solve_hit_limit(solve_details)
        mdl._solve_details = solve_details

        # call notify_end on listeners
        if solve_res:
            obj = solve_res.objective_value
            has_solution = True
        else:
            has_solution = False
            obj = None

        mdl._fire_end_solve_listeners(has_solution, obj)


class CplexLocalSolveEnv(SolveEnv):

    def __init__(self, m):
        super(CplexLocalSolveEnv, self).__init__(m)
        self._saved_params = {}
        self._transaction = get_environment().create_transaction()

    def before_solve(self, context):
        mdl = self._model

        # step 1 : notify start solve
        mdl.notify_start_solve()

        the_env = get_environment()
        auto_publish_details = is_auto_publishing_solve_details(context)
        if the_env.is_wmlworker and auto_publish_details:
            # do not use lambda here
            def env_kpi_hookfn(kpd):
                the_env.update_solve_details(kpd, transaction=self._transaction)

            from docplex_wml.worker.worker_utils import _KpiRecorder
            mdl.kpi_recorder = _KpiRecorder(mdl,
                                            clock=context.solver.kpi_reporting.filter_level,
                                            publish_hook=env_kpi_hookfn,
                                            transaction=self._transaction)
            mdl.add_progress_listener(mdl.kpi_recorder)

        # connect progress listeners (if any) if problem is mip
        mdl._connect_progress_listeners()

        # call notifyStart on progress listeners
        mdl._fire_start_solve_listeners()

        # The following block used to be published only if auto_publish_details.
        # It is now modified so that notify start is always performed,
        # then we update solve details only if they need to be published
        # [[[
        self_stats = mdl.statistics
        # implementation for https://github.ibm.com/IBMDecisionOptimization/dd-planning/issues/2491
        problem_type = mdl._get_cplex_problem_type()
        kpis = make_new_kpis_dict(allkpis=mdl._allkpis[:],
                                      sense=mdl._objective_sense.verb,
                                      model_type=problem_type,
                                      int_vars=self_stats.number_of_integer_variables,
                                      continuous_vars=self_stats.number_of_continuous_variables,
                                      semicontinuous_vars = self_stats._number_of_semicontinuous_variables,
                                      semiinteger_vars = self_stats._number_of_semiinteger_variables,
                                      linear_constraints=self_stats.number_of_linear_constraints,
                                      bin_vars=self_stats.number_of_binary_variables,
                                      quadratic_constraints=self_stats.number_of_quadratic_constraints,
                                      total_constraints=self_stats.number_of_constraints,
                                      total_variables=self_stats.number_of_variables)
        if the_env.is_wmlworker:
            from docplex_wml.worker.worker_utils import make_cplex_new_kpis_dict
            new_kpis = make_cplex_new_kpis_dict(mdl)
            kpis.update(new_kpis)

        the_env.notify_start_solve(kpis)
        if auto_publish_details:
            the_env.update_solve_details(kpis, transaction=self._transaction)

        # ---
        #  parameters override if necessary...
        # --
        self_params = mdl.context._get_raw_cplex_parameters()
        parameters = apply_thread_limitations(context)
        if self_params and parameters is not self_params:
            self._saved_params = {p: p.get() for p in self_params}
        else:
            self._saved_params = {}

        # returns the parameters group to use.
        return parameters

    def after_solve(self, context, solve_res, engine):
        mdl = self._model
        super(CplexLocalSolveEnv, self).after_solve(context, solve_res, engine)

        mdl._disconnect_progress_listeners()
        mdl._clear_qprogress_listeners()

        # --- specific to local
        the_env = get_environment()
        the_env.notify_end_solve(mdl.job_solve_status)

        if is_auto_publishing_solve_details(context):
            details = mdl.solve_details.as_worker_dict()
            if solve_res:
                new_solution = solve_res
                kpi_dict = mdl.kpis_as_dict(new_solution, use_names=True)

                def publish_name_fn(kn):
                    return 'KPI.%s' % kn
                
                def convert_from_numpy(o):
                    return o.item() if isinstance(o, numpy.generic) else o

                def identity(o):
                    return o
                
                publish_value = convert_from_numpy if numpy else identity
                # build a dict of kpi names (formatted) -> kpi values
                # kpi values are converted to python types if they are numpy types
                kpi_details_dict = {publish_name_fn(kn): publish_value(kv) for kn, kv in kpi_dict.items()}
                details.update(kpi_details_dict)
                # add objective with predefined key name
                details['PROGRESS_CURRENT_OBJECTIVE'] = new_solution.objective_value
            the_env.update_solve_details(details, transaction=self._transaction)

        if solve_res:
            auto_publish_solution = auto_publishing_result_output_names(context) is not None
            auto_publish_kpis_table = auto_publishing_kpis_table_names(context) is not None
            if auto_publish_solution:
                write_result_output(env=the_env,
                                    context=context,
                                    model=mdl,
                                    solution=solve_res,
                                    transaction=self._transaction)

            # save kpi
            if auto_publish_kpis_table:
                write_kpis_table(env=the_env,
                                 context=context,
                                 model=mdl,
                                 solution=solve_res,
                                 transaction=self._transaction)

        # restore cached params
        saved_params = self._saved_params
        if saved_params:
            self_engine = mdl.get_engine()
            for p, v in saved_params.items():
                self_engine.set_parameter(p, v)
                # clear saved
            self._saved_params = {}

        kpi_rec_attr_name = 'kpi_recorder'
        try:
            kpi_rec = getattr(mdl, kpi_rec_attr_name, None)
            if kpi_rec:
                delattr(mdl, kpi_rec_attr_name)
                mdl.remove_progress_listener(kpi_rec)

        except (AttributeError, ValueError):
            pass

        self._transaction.commit()
