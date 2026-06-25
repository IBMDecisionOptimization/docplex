# read mipstarts from cplex and upload them to solution

import unittest

from docplex.mp.constants import WriteLevel

def cpx_mipstarts_to_solutions(cpx, mdl):
    # arguments: cpx: a Cplex() instance
    # mdl: a model instance
    cpx_mip_starts = cpx.MIP_starts
    number_of_mipstarts = cpx_mip_starts.get_num()
    mipstarts = []
    if number_of_mipstarts:
        starts = cpx_mip_starts.get_starts()
        names = cpx_mip_starts.get_names()
        efforts = cpx_mip_starts.get_effort_levels()
        for effort, start, name in zip(efforts, starts, names):
            mipstart_sol = mdl.new_solution(name=name)
            # convert start to var values
            sp, _ = start
            for (index, value) in zip (sp.ind, sp.val):
                #
                dv = mdl.get_var_by_index(index)
                if dv:
                    mipstart_sol._set_var_value(dv, value)
            mipstarts.append((mipstart_sol, effort))
    return mipstarts


class MIPstartRoundripTests(unittest.TestCase):

    def get_number_of_expected_vars_in_mst(self, mdl, sol, level):
        if level == WriteLevel.AllVars:
            return mdl.number_of_user_variables
        elif level in {WriteLevel.Auto, WriteLevel.DiscreteVars}:
            return sum(1 for uv in mdl.generate_user_variables() if uv.is_discrete())
        elif level == WriteLevel.NonZeroVars:
            return sum(1 for uv in mdl.generate_user_variables() if sol[uv])
        elif level == WriteLevel.NonZeroDiscreteVars:
            return sum(1 for uv in mdl.generate_user_variables() if sol[uv] and uv.is_discrete())
        else:
            return 0

    def check_mipstart_roundtrip(self, mdl, level= WriteLevel.Auto, **solve_kwargs):
        s = mdl.solve(**solve_kwargs)
        self.assertIsNotNone(s)
        basename = "%s_%s" % (mdl.name.lower(), level.short_name)
        mst_path = s.export_as_mst(basename=basename, write_level=level)
        self.assertIsNotNone(mst_path)

        cpx = mdl.get_cplex()
        print("* reading MST file: {0}".format(mst_path))
        cpx.MIP_starts.read(mst_path)
        mipstart_sols = cpx_mipstarts_to_solutions(cpx, mdl)
        self.assertEqual(1, len(mipstart_sols))
        mst_sol = mipstart_sols[0][0]
        expected_nb_vars = self.get_number_of_expected_vars_in_mst(mdl, s,level)
        print("-- expecting {0} variables in MST".format(expected_nb_vars))
        self.assertEqual(self.get_number_of_expected_vars_in_mst(mdl, s,level), len(mst_sol))
        return mipstart_sols

    def test_warehouse_auto(self):
        from examples.modeling.warehouse import build_test_model
        wm = build_test_model()
        mss = self.check_mipstart_roundtrip(wm)

    def test_warehouse_all(self):
        from examples.modeling.warehouse import build_test_model
        wm = build_test_model()
        mss = self.check_mipstart_roundtrip(wm, level=WriteLevel.AllVars)

    def test_warehouse_nonzero_discrete(self):
        from examples.modeling.warehouse import build_test_model
        wm = build_test_model()
        mss = self.check_mipstart_roundtrip(wm, level=WriteLevel.NonZeroDiscreteVars)

    def test_mst_roundtrip_sports_auto(self):
        from examples.delivery.modeling.sport_scheduling import build_sports
        spm = build_sports()
        mss = self.check_mipstart_roundtrip(spm, level=WriteLevel.Auto)

    def test_mst_roundtrip_sports_nonzero_discrete(self):
        from examples.delivery.modeling.sport_scheduling import build_sports
        spm = build_sports()
        mss = self.check_mipstart_roundtrip(spm, level=WriteLevel.NonZeroDiscreteVars)


if __name__ == "__main__":
    unittest.main()