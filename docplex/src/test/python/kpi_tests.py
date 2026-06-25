#!/usr/bin/python
# -*- coding: utf-8 -*-
# import unittest

from testutils import DocplexAbstractTest, RedirectedOutputToStringContext, silent_remove, skipIfCplexCE
from docplex.mp.utils import DOcplexException

from docplex.mp.context import create_default_auto_publish_context

import os
import unittest
import six

the_hook = None
try:
    from docplex_wml.worker.solvehook import SolveHook
    from testutils import OverrideHook

    HOOK_AVAILABLE = True


    # set a custom solve hook to record things
    class MySolveHook(SolveHook):
        def __init__(self):
            super(MySolveHook, self).__init__()
            self.solve_details = []
            self.attachments = {}
            self.attachments_history = []

        def reset(self):
            self.solve_details = []
            self.attachments = {}

        def set_output_attachments(self, attachments):
            self.attachments.update(attachments)
            self.attachments_history.append(attachments)

        def get_parameter_value(self, name):
            return SolveHook.get_parameter_value(self, name)

        def update_solve_details(self, details):
            self.solve_details.append(details)

        def get_available_core_count(self):
            return 4  # arbitrary esoteric number


    the_hook = MySolveHook()
except ImportError:
    the_hook = None

try:
    import pandas
except ImportError:
    pandas = None

try:
    import docplex_wml
except ImportError:
    docplex_wml = None

class KPITestsBase(DocplexAbstractTest):

    def tearDown(self):
        DocplexAbstractTest.tearDown(self)
        silent_remove('kpis.csv')


class KPICustomSolutionTests(DocplexAbstractTest):

    def setUp(self):
        DocplexAbstractTest.setUp(self)
        m = self.model
        self.x = m.integer_var(name='ix', lb=5)
        self.y = m.integer_var(name='iy', lb=3)

        self.custom_s = m.new_solution(var_value_dict={self.x: 13, self.y: 17})

    def tearDown(self):
        DocplexAbstractTest.tearDown(self)
        silent_remove('kpis.csv')

    def test_custom_kpis_as_dict(self):
        m = self.model
        x = m.integer_var(name='ii', lb=5)
        y = m.integer_var(name='jj', lb=3)

        kpi1 = m.add_kpi(x, 'k1')
        kpi2 = m.add_kpi(x + 2 * y, 'k2')
        kpi3 = m.add_kpi(x ** 2 + y ** 2, 'k3')
        kpi4 = m.add_kpi(m.abs(x - y), publish_name='ze_abs')
        kpi5 = m.add_kpi(m.max(x, y), publish_name='ze_max')
        m.minimize(x + y)
        s = m.new_solution(var_value_dict={x: 13, y: 17})
        kpid = m.kpis_as_dict(solution=s)

        self.assertEqual(5, len(kpid))
        self.assertIn('k1', kpid)
        self.assertEqual(13, kpid['k1'])
        self.assertIn('k2', kpid)
        self.assertEqual((13 + 2 * 17), kpid['k2'])
        self.assertIn('k3', kpid)
        # 13^2 + 17^2 = 169 + 2 = 458
        self.assertEqual(458, kpid['k3'])
        self.assertIn('ze_abs', kpid)
        self.assertEqual(4, kpid[kpi4.name])
        self.assertIn('ze_max', kpid)
        self.assertEqual(17, kpid[kpi5.name])
        # use filtering
        kpd2 = m.kpis_as_dict(s, kpi_filter=lambda k: 'ze' in k.name)
        # we get 'ze_abs', 'ze_max'
        self.assertEqual(2, len(kpd2))
        self.assertIn('ze_abs', kpd2)
        self.assertIn('ze_max', kpd2)

    def test_custom_solution_kpi_compute_unsolved(self):
        m = self.model
        x = self.x
        y = self.y
        kpi1 = m.add_kpi(x, 'k1')
        kpi2 = m.add_kpi(x + 2 * y, 'k2')
        fkpi = m.add_kpi(lambda m1, s1: 3 + s1['ix'])
        my_s = self.custom_s

        self.assertEqual(13, kpi1.compute(my_s))
        self.assertEqual(13 + 2 * 17, kpi2.compute(my_s))
        self.assertEqual(16, fkpi.compute(my_s))

    def test_custom_solution_kpi_report_unsolved(self):
        m = self.model
        x = self.x
        y = self.y
        m.add_kpi(x, 'k1')
        m.add_kpi(x + 2 * y, 'k2')
        m.add_kpi(lambda m_, s_: 3 + s_['ix'], 'fk')
        my_s = self.custom_s

        with RedirectedOutputToStringContext() as oss:
            m.report_kpis(my_s)
        log = oss.get_str()
        self.assertIn('k1 = 13', log)
        self.assertIn('k2 = 47', log)
        self.assertIn('fk = 16', log)

    def test_custom_solution_kpi_value_by_name_unsolved(self):
        m = self.model
        x = self.x
        y = self.y
        m.add_kpi(x, 'k1')
        m.add_kpi(x + 2 * y, 'k2')
        m.add_kpi(lambda m_, s_: 3 + s_['ix'], 'fk')
        my_s = self.custom_s

        self.assertEqual(13, m.kpi_value_by_name('k1', my_s))
        self.assertEqual(47, m.kpi_value_by_name('k2', my_s))
        self.assertEqual(16, m.kpi_value_by_name('fk', my_s))

    def test_custom_solution_kpi_as_dict_unsolved(self):
        m = self.model
        x = self.x
        y = self.y
        m.add_kpi(x, 'k1')
        m.add_kpi(x + 2 * y, 'k2')
        m.add_kpi(lambda m_, s_: 3 + s_['ix'], 'fk')
        my_s = self.custom_s

        self.assertEqual({'k1': 13, 'k2': 47, 'fk':16}, m.kpis_as_dict(my_s))

    def test_custom_solution_kpi_compute_solved(self):
        m = self.model
        x = self.x
        y = self.y
        kpi1 = m.add_kpi(x, 'k1')
        kpi2 = m.add_kpi(x + 2 * y, 'k2')
        fkpi = m.add_kpi(lambda m1, s1: 3 + s1['ix'])
        m.solve()
        my_s = self.custom_s

        self.assertEqual(13, kpi1.compute(my_s))
        self.assertEqual(13 + 2 * 17, kpi2.compute(my_s))
        self.assertEqual(16, fkpi.compute(my_s))

    def test_custom_solution_kpi_report_solved(self):
        m = self.model
        x = self.x
        y = self.y
        m.add_kpi(x, 'k1')
        m.add_kpi(x + 2 * y, 'k2')
        m.add_kpi(lambda m_, s_: 3 + s_['ix'], 'fk')

        m.solve()
        my_s = self.custom_s

        with RedirectedOutputToStringContext() as oss:
            m.report_kpis(my_s)
        log = oss.get_str()
        self.assertIn('k1 = 13', log)
        self.assertIn('k2 = 47', log)
        self.assertIn('fk = 16', log)

    def test_custom_solution_kpi_value_by_name_solved(self):
        m = self.model
        x = self.x
        y = self.y
        m.add_kpi(x, 'k1')
        m.add_kpi(x + 2 * y, 'k2')
        m.add_kpi(lambda m_, s_: 3 + s_['ix'], 'fk')
        m.solve()
        my_s = self.custom_s

        # from solution of solve
        self.assertEqual(5, m.kpi_value_by_name('k1'))
        self.assertEqual(11, m.kpi_value_by_name('k2'))
        self.assertEqual(8, m.kpi_value_by_name('fk'))

        self.assertEqual(13, m.kpi_value_by_name('k1', my_s))
        self.assertEqual(47, m.kpi_value_by_name('k2', my_s))
        self.assertEqual(16, m.kpi_value_by_name('fk', my_s))

    def test_custom_solution_kpi_as_dict_solved(self):
        m = self.model
        x = self.x
        y = self.y
        m.add_kpi(x, 'k1')
        m.add_kpi(x + 2 * y, 'k2')
        m.add_kpi(lambda m_, s_: 3 + s_['ix'], 'fk')
        m.solve()
        # from solve
        self.assertEqual({'k1':5, 'k2': 11, 'fk': 8}, m.kpis_as_dict())
        # from custom
        self.assertEqual({'k1': 13, 'k2': 47, 'fk':16}, m.kpis_as_dict(self.custom_s))

class DocplexKpiTests(KPITestsBase):


    def test_kpi_var_unnamed(self):
        m = self.model
        ijk = m.integer_var(name='ijk')
        kpi = m.add_kpi(ijk, publish_name=None)
        self.assertEqual(kpi.name, 'ijk')

    def test_kpi_var_named(self):
        m = self.model
        ijk = m.integer_var(name='ijk')
        kpi = m.add_kpi(ijk, publish_name='foo')
        self.assertEqual(kpi.name, 'foo')

    def test_kpi_linexpr_unnamed(self):
        m = self.model
        ijks = m.integer_var_list(3, name='ijk')
        kpi = m.add_kpi(m.sum(ijks), publish_name=None)
        self.assertEqual(kpi.name, 'ijk_0+ijk_1+ijk_2')

    def test_kpi_linexpr_bad_name_type(self):
        m = self.model
        ijks = m.integer_var_list(3, name='ijk')
        six.assertRaisesRegex(self, DOcplexException, "Expecting a non-empty string",
                              lambda m1: m1.add_kpi(m1.sum(ijks), publish_name=3.14), m)

    def test_kpi_linexpr_bad_name_empty_string(self):
        m = self.model
        ijks = m.integer_var_list(3, name='ijk')
        six.assertRaisesRegex(self, DOcplexException, "non-empty string",
                              lambda m1: m1.add_kpi(m1.sum(ijks), publish_name=""), m)

    # def test_kpi_linexpr_named_no_publish_name(self):
    #     m = self.model
    #     ijks = m.integer_var_list(3, name='ijk')
    #     sum_ijks = m.sum(ijks)
    #     sum_ijks.name = 'the_big_sum'
    #     kpi = m.add_kpi(sum_ijks)
    #     self.assertEqual(kpi.name, 'the_big_sum')

    def test_kpi_match(self):
        m = self.model
        x = m.integer_var(name='x')
        m.add_kpi(x * x, 'x square')
        k = None
        try:
            k = m.kpi_by_name('square', try_match=False)
        except Exception as e:
            print(e)
        self.assertEqual(k, None)
        try:
            k = m.kpi_by_name('square', try_match=True)
        except Exception as e:
            print(e)
        self.assertEqual(k.name, 'x square')

    def test_remove_kpi_string(self):
        m = self.model
        x = m.integer_var(name='x')
        m.add_kpi(x ** 2, publish_name='square')
        self.assertEqual(1, m.number_of_kpis)
        m.remove_kpi('square')
        self.assertEqual(0, m.number_of_kpis)

    def test_remove_kpi_kpi(self):
        m = self.model
        x = m.integer_var(name='x')
        kpi1 = m.add_kpi(x ** 2, publish_name='square')
        self.assertEqual(1, m.number_of_kpis)
        m.remove_kpi(kpi1)
        self.assertEqual(0, m.number_of_kpis)

    def test_remove_kpi_other(self):
        m = self.model
        x = m.integer_var(name='x')
        m.add_kpi(x ** 2, publish_name='square')
        self.assertEqual(1, m.number_of_kpis)
        m.remove_kpi(x)

    def s(self):
        m = self.model
        x = m.integer_var(name='x')
        m.add_kpi(x * x, publish_name='x square')
        m.add_kpi(2 * x, publish_name='2x')
        self.assertEqual(2, m.number_of_kpis)
        m.clear_kpis()
        self.assertEqual(0, m.number_of_kpis)
        self.assertEqual([], list(m.iter_kpis()))

    def test_rename_kpi_named(self):
        m = self.model
        x = m.integer_var(name='x')
        k1 = m.add_kpi(x * x, publish_name='x square')
        self.assertEqual('x square', k1.name)
        k1.name = 'x2'
        self.assertEqual('x2', k1.name)

    def test_rename_kpi_bad_name1(self):
        m = self.model
        x = m.integer_var(name='x')
        k1 = m.add_kpi(x * x, 'x2')
        six.assertRaisesRegex(self, DOcplexException, 'KPI.name: Expecting a non-empty string',
                              lambda k: k.set_name(None), k1)

    def test_rename_kpi_bad_name2(self):
        m = self.model
        x = m.integer_var(name='x')
        k1 = m.add_kpi(x * x, 'x2')
        six.assertRaisesRegex(self, DOcplexException, 'KPI.name: Expecting a non-empty string',
                              lambda k: k.set_name(''), k1)

    def test_kpi_name_collision_ko(self):
        m = self.model
        x = m.integer_var(name='x')
        m.add_kpi(x * x, 'foo')
        six.assertRaisesRegex(self, DOcplexException, "Duplicate KPI name: \"foo\"",
                              lambda m_: m_.add_kpi(x + 1, 'foo'), m)

    def test_report_kpis_unsolved_ko(self):
        # report with no explicit solution fails
        m = self.model
        x = m.integer_var(name='i', lb=5)
        y = m.integer_var(name='j', lb=3)

        m.add_kpi(x, 'k1')
        m.add_kpi(lambda m1, s1: 3 + s1['i'], 'fk')

        # report_kpis() has neither solve sol nor explicit -> break
        six.assertRaisesRegex(self, DOcplexException, "model is not solved and no solution was passed",
                              lambda m_: m_.report_kpis(), m)

    def test_kpi_value_by_name_unsolved_ko(self):
        m = self.model
        x = m.integer_var(name='i', lb=5)
        y = m.integer_var(name='j', lb=3)
        m.add_kpi(x, 'k1')
        six.assertRaisesRegex(self, DOcplexException, "model is not solved and no solution was passed",
                             lambda m_: m_.kpi_value_by_name('k1'), m)

    def test_kpis_as_dict_unsolved_ko(self):
        # report with no explicit solution fails
        m = self.model
        x = m.integer_var(name='i', lb=5)
        y = m.integer_var(name='j', lb=3)

        m.add_kpi(x, 'k1')
        m.add_kpi(lambda m1, s1: 3 + s1['i'], 'fk')

        # report_kpis() has neither solve sol nor explicit -> break
        six.assertRaisesRegex(self, DOcplexException, "model is not solved and no solution was passed",
                              lambda m_: m_.kpis_as_dict(), m)



    def test_kpi_subscriber(self):
        m = self.model
        x = m.integer_var(name='ii', lb=5)
        y = m.integer_var(name='jj', lb=3)

        e = x + 2 * y + 3
        self.assertFalse(e.is_in_use())  # not used
        k1 = m.add_kpi(e, 'bla bla')
        self.assertTrue(e.is_in_use())
        e -= 3
        self.assertEqual(str(k1.as_expression()), 'ii+2jj')
        m.remove_kpi(k1)
        self.assertFalse(e.is_in_use())

    def test_kpi_values(self):
        m = self.model
        x = m.integer_var(name='ii', lb=5, ub=100)
        y = m.integer_var(name='jj', lb=3, ub=100)
        m.minimize(x + y)
        kp1 = m.add_kpi(x + 2 * y + 3, 'foo')
        s = m.solve()
        #m.export_as_lp(basename='bizarre')
        self.assertIsNotNone(s)
        xpected = 14  # 5 + 2 * 3 + 3
        self.assertEqual(xpected, kp1.compute())
        self.assertEqual(xpected, kp1.compute(s))
        self.assertEqual(xpected, kp1.solution_value)
        # also works with a dict!
        # dd = {x: 5, y: 3}
        # self.assertEqual(xpected, kp1.compute(dd))
        self.assertAlmostEqual(14, m.kpi_value_by_name('foo'), delta=1e-2)
        six.assertRaisesRegex(self, DOcplexException, "Model has no KPI with name matching", lambda m_: m_.kpi_value_by_name('bar'), m)

    def test_funkpis(self):
        m = self.model
        x = m.integer_var(name='ii', lb=5, ub=100)
        y = m.integer_var(name='jj', lb=3, ub=100)
        m.minimize(x + y)
        m.add_kpi(lambda m_, s_: 3.14, 'pi')
        m.add_kpi(lambda m_, s_: 'bingo', 'status')
        m.solve()
        dd = m.kpis_as_dict()
        self.assertEqual(2, len(dd))
        self.assertEqual(dd['pi'], 3.14)
        self.assertEqual(dd['status'], 'bingo')

    def test_funkpi_report(self):
        m = self.model
        x = m.integer_var(name='ii', lb=5, ub=100)
        y = m.integer_var(name='jj', lb=3, ub=100)
        m.solve()
        m.add_kpi(lambda m_, s_: s_.number_of_var_values, publish_name="nvv")
        with RedirectedOutputToStringContext() as oss:
            m.report_kpis()
        self.assertIn("*  KPI: nvv = 2.000", oss.get_str())

    def test_numkpis(self):
        m = self.model
        x = m.integer_var(name='ii', lb=5, ub=100)
        y = m.integer_var(name='jj', lb=3, ub=100)
        m.minimize(x + y)
        m.add_kpi(3.14, 'pi')
        m.add_kpi(9999, "BIG")
        m.add_kpi(0, "zero")
        m.add_kpi(-1, "minus")
        s = m.solve()
        self.assertIsNotNone(s)
        dd = m.kpis_as_dict()
        self.assertEqual(4, len(dd))
        self.assertEqual(dd['pi'], 3.14)
        self.assertEqual(dd['BIG'], 9999)
        self.assertEqual(dd['zero'], 0)
        self.assertEqual(dd['minus'], -1)

    def test_funkpis_report(self):
        m = self.model
        x = m.integer_var(name='ii', lb=5, ub=100)
        y = m.integer_var(name='jj', lb=3, ub=100)
        m.minimize(x + y)
        m.add_kpi(lambda m_, s_: 3.14, 'pi')
        m.add_kpi(lambda m_, s_: 'bingo', 'status')
        m.solve()
        with RedirectedOutputToStringContext() as oss:
            m.report_kpis()
        log = oss.get_str()
        self.assertIn('KPI: pi     = 3.140', log)
        self.assertIn('KPI: status = bingo', log)

    @unittest.skipIf(pandas is None or docplex_wml is None, "pandas/docplex_wml not available")
    def test_auto_publish_kpi_table(self):
        '''Test the auto publish kpi feature'''
        from docplex_wml.worker.environment import WorkerEnvironment

        with OverrideHook(the_hook, new_env=WorkerEnvironment(the_hook)):
            m = self.model
            m.context.solver.auto_publish = create_default_auto_publish_context()
            x = m.integer_var(name='ii', lb=5, ub=100)
            y = m.integer_var(name='jj', lb=3, ub=100)
            m.minimize(x + y)
            m.add_kpi(lambda m1, s1: 3.14, 'pi')
            m.add_kpi(lambda m1, s1: 'bingo', 'status')
            m.solve()
            m.report()
            name_field = m.context.solver.auto_publish.kpis_output_field_name
            value_field = m.context.solver.auto_publish.kpis_output_field_value

            # TODO: test KPI values in my_hook.solve_details
            # test KPI files
            kpi_file = the_hook.attachments['kpis.csv']
            with open(kpi_file, 'rb') as f:
                idf = pandas.read_csv(f)
                self.assertEqual(len(idf), m.number_of_kpis, 'wrong number of kpis')
                self.assertAlmostEqual(float(idf[idf[name_field] == 'pi'][value_field]),
                                       3.14, delta=1e-3)
                self.assertEqual(idf[idf[name_field] == 'status'][value_field].tolist()[0],
                                 'bingo')


    @unittest.skipIf(pandas is None or docplex_wml is None, "pandas/docplex_wml not available")
    def test_auto_publish_kpi_table_customized_fields(self):
        '''Test the auto publish kpi feature'''
        from docplex_wml.worker.environment import WorkerEnvironment

        with OverrideHook(the_hook, new_env=WorkerEnvironment(the_hook)):
            m = self.model
            m.context.solver.auto_publish = create_default_auto_publish_context()
            name_field = "TutuName"
            value_field = "TutuValue"
            m.context.solver.auto_publish.kpis_output_field_name = name_field
            m.context.solver.auto_publish.kpis_output_field_value = value_field
            x = m.integer_var(name='ii', lb=5, ub=100)
            y = m.integer_var(name='jj', lb=3, ub=100)
            m.minimize(x + y)
            m.add_kpi(lambda m1, s1: 3.14, 'pi')
            m.add_kpi(lambda m1, s1: 'bingo', 'status')
            m.solve()
            m.report()

            # TODO: test KPI values in my_hook.solve_details
            # test KPI files
            kpi_file = the_hook.attachments['kpis.csv']
            with open(kpi_file, 'rb') as f:
                idf = pandas.read_csv(f)
                self.assertEqual(len(idf), m.number_of_kpis, 'wrong number of kpis')
                self.assertAlmostEqual(float(idf[idf[name_field] == 'pi'][value_field]),
                                       3.14, delta=1e-3)
                self.assertEqual(idf[idf[name_field] == 'status'][value_field].tolist()[0],
                                 'bingo')

    @unittest.skipIf(docplex_wml is None, "docplex_wml not available")
    @skipIfCplexCE
    def test_auto_publish_internals(self):
        from docplex_wml.worker.environment import WorkerEnvironment
        with OverrideHook(the_hook, new_env=WorkerEnvironment(the_hook)):
            from examples.modeling.generics.ucp_new import make_default_ucp_model
            ucpm = make_default_ucp_model()
            ucpm.context.solver.auto_publish = True
            ucpm.solve()
            print(len(the_hook.solve_details))
            for i in the_hook.solve_details:
                print(i)
            att_with_kpi = [a for a in the_hook.attachments_history if 'kpis.csv' in a]
            print(len(att_with_kpi))
            for i in att_with_kpi:
                print(i)
            # solve details should have one more entry than attachments number
            # because of the final kpi publish in solve_local()
            self.assertGreater(len(the_hook.solve_details), len(att_with_kpi),
                             '# of solve details with kpi != number of attachments published')

    @unittest.skipIf(docplex_wml is None, "docplex_wml not available")
    def test_check_that_last_solution_kpi_published(self):
        '''see defect #36015: If there's no intermediate solutions,
        kpis are never published to solve details
        '''
        from docplex_wml.worker.environment import WorkerEnvironment
        with OverrideHook(the_hook, new_env=WorkerEnvironment(the_hook)):
            m = self.model
            m.context.solver.auto_publish = create_default_auto_publish_context()
            x = m.integer_var(name='ii', lb=5, ub=100)
            y = m.integer_var(name='jj', lb=3, ub=100)
            m.minimize(x + y)
            m.add_kpi(x, 'kpi1')
            m.solve()
            m.report()
            last_details = the_hook.solve_details[-1]
            self.assertIn('KPI.kpi1', last_details, 'Should have found a KPI.kpi1 entry')

    @unittest.skipIf(pandas is None or docplex_wml is None, "Skipping because pandas is not available")
    @skipIfCplexCE
    def test_defect_36015_objective_is_zero(self):
        from docplex_wml.worker.environment import WorkerEnvironment
        with OverrideHook(the_hook, new_env=WorkerEnvironment(the_hook)):
            import pandas as pd

            names = {
                139987: "Guadalupe J. Martinez", 140030: "Michelle M. Lopez", 140089: "Terry L. Ridgley",
                140097: "Miranda B. Roush", 139068: "Sandra J. Wynkoop", 139154: "Roland Gu�rette",
                139158: "Fabien Mailhot",
                139169: "Christian Austerlitz", 139220: "Steffen Meister", 139261: "Wolfgang Sanger",
                139416: "Lee Tsou", 139422: "Sanaa' Hikmah Hakimi", 139532: "Miroslav �karoupka",
                139549: "George Blomqvist", 139560: "Will Henderson", 139577: "Yuina Ohira", 139580: "Vlad Alekseeva",
                139636: "Cassio Lombardo", 139647: "Trinity Zelaya Miramontes", 139649: "Eldar Muravyov",
                139665: "Shu T'an",
                139667: "Jameel Abdul-Ghani Gerges", 139696: "Zeeb Longoria Marrero", 139752: "Matheus Azevedo Melo",
                139832: "Earl B. Wood", 139859: "Gabrielly Sousa Martins", 139881: "Franca Palermo"}

            data = [(139987, "Pension", 0.13221, "Mortgage", 0.10675), (140030, "Savings", 0.95678, "Pension", 0.84446),
                    (140089, "Savings", 0.95678, "Pension", 0.80233),
                    (140097, "Pension", 0.13221, "Mortgage", 0.10675), (139068, "Pension", 0.80506, "Savings", 0.28391),
                    (139154, "Pension", 0.13221, "Mortgage", 0.10675),
                    (139158, "Pension", 0.13221, "Mortgage", 0.10675),
                    (139169, "Pension", 0.13221, "Mortgage", 0.10675),
                    (139220, "Pension", 0.13221, "Mortgage", 0.10675),
                    (139261, "Pension", 0.13221, "Mortgage", 0.10675),
                    (139416, "Pension", 0.13221, "Mortgage", 0.10675),
                    (139422, "Pension", 0.13221, "Mortgage", 0.10675),
                    (139532, "Savings", 0.95676, "Mortgage", 0.82269), (139549, "Savings", 0.16428, "Pension", 0.13221),
                    (139560, "Savings", 0.95678, "Pension", 0.86779),
                    (139577, "Pension", 0.13225, "Mortgage", 0.10675),
                    (139580, "Pension", 0.13221, "Mortgage", 0.10675),
                    (139636, "Pension", 0.13221, "Mortgage", 0.10675),
                    (139647, "Savings", 0.28934, "Pension", 0.13221), (139649, "Pension", 0.13221, "Mortgage", 0.10675),
                    (139665, "Savings", 0.95675, "Pension", 0.27248),
                    (139667, "Pension", 0.13221, "Mortgage", 0.10675), (139696, "Savings", 0.16188, "Pension", 0.13221),
                    (139752, "Pension", 0.13221, "Mortgage", 0.10675),
                    (139832, "Savings", 0.95678, "Pension", 0.83426), (139859, "Savings", 0.95678, "Pension", 0.75925),
                    (139881, "Pension", 0.13221, "Mortgage", 0.10675)]

            products = ["Car loan", "Savings", "Mortgage", "Pension"]
            productValue = [100, 200, 300, 400]
            budgetShare = [0.6, 0.1, 0.2, 0.1]

            availableBudget = 500
            channels = pd.DataFrame(data=[("gift", 20.0, 0.20), ("newsletter", 15.0, 0.05), ("seminar", 23.0, 0.30)],
                                    columns=["name", "cost", "factor"])
            offers = pd.DataFrame(data=data, index=range(0, len(data)),
                                  columns=["customerid", "Product1", "Confidence1", "Product2", "Confidence2"])

            offers.insert(0, 'name', pd.Series(names[i[0]] for i in data))

            from docplex.mp.model import Model
            mdl = Model(name="marketing_campaign")
            mdl.context.solver.auto_publish = create_default_auto_publish_context()

            offersR = range(0, len(offers))
            productsR = range(0, len(products))
            channelsR = range(0, len(channels))

            channelVars = mdl.binary_var_cube(offersR, productsR, channelsR)
            totaloffers = mdl.integer_var(lb=0)
            budgetSpent = mdl.continuous_var()

            # Only 1 product is offered to each customer
            mdl.add_constraints(mdl.sum(channelVars[o, p, c] for p in productsR for c in channelsR) <= 1
                                for o in offersR)

            mdl.add_constraint(totaloffers == mdl.sum(channelVars[o, p, c]
                                                      for o in offersR
                                                      for p in productsR
                                                      for c in channelsR))

            mdl.add_constraint(budgetSpent == mdl.sum(channelVars[o, p, c] * channels.at[c, "cost"]
                                                      for o in offersR
                                                      for p in productsR
                                                      for c in channelsR))

            # Balance the offers among products
            for p in productsR:
                mdl.add_constraint(mdl.sum(channelVars[o, p, c] for o in offersR for c in channelsR)
                                   <= budgetShare[p] * totaloffers)

            # Do not exceed the budget
            mdl.add_constraint(mdl.sum(channelVars[o, p, c] * channels.at[c, "cost"]
                                       for o in offersR
                                       for p in productsR
                                       for c in channelsR) <= availableBudget)

            mdl.print_information()

            mdl.maximize(
                mdl.sum(channelVars[idx, p, idx2] * c.factor * productValue[p] * o.Confidence1
                        for p in productsR
                        for idx, o in offers[offers['Product1'] == products[p]].iterrows()
                        for idx2, c in channels.iterrows())
                +
                mdl.sum(channelVars[idx, p, idx2] * c.factor * productValue[p] * o.Confidence2
                        for p in productsR
                        for idx, o in offers[offers['Product2'] == products[p]].iterrows()
                        for idx2, c in channels.iterrows())
            )

            s = mdl.solve()

        # we should have in the last solve details
        last = the_hook.solve_details[-1]
        self.assertIn('PROGRESS_BEST_OBJECTIVE', last,
                      'could not find PROGRESS_BEST_OBJECTIVE')
        self.assertIn('PROGRESS_CURRENT_OBJECTIVE', last,
                      'could not find PROGRESS_CURRENT_OBJECTIVE')
        self.assertAlmostEqual(last['PROGRESS_BEST_OBJECTIVE'],
                               last['PROGRESS_CURRENT_OBJECTIVE'],
                               delta=1e-4,
                               msg='Should have a PROGRESS_CURRENT_OBJECTIVE value')

    @unittest.skipIf(pandas is None or docplex_wml is None, "Skipping because pandas/docplex_wml is not available")
    @skipIfCplexCE
    def test_defect_36015_objective_is_na(self):
        from docplex_wml.worker.environment import WorkerEnvironment
        from docplex.mp.model import Model
        with OverrideHook(the_hook, new_env=WorkerEnvironment(the_hook)):
            from os.path import dirname, join
            this_path = dirname(os.path.abspath(__file__))
            tests_path = dirname(this_path)
            data_path = join(tests_path, 'data', 'models', 'rtc36015_na')

            import pandas as pd

            inputs = {}
            inputs['resources'] = pd.read_csv(join(data_path, 'resources.csv'),
                                              index_col=None)
            inputs['demands'] = pd.read_csv(join(data_path, 'demands.csv'),
                                            index_col=None)

            N_DAYS = 2
            N_PERIODS_PER_DAY = 12 * 24
            N_PERIODS = N_DAYS * N_PERIODS_PER_DAY

            df_resources = inputs['resources']
            df_demands = inputs['demands']

            mdl = Model("planning")
            mdl.context.solver.auto_publish = create_default_auto_publish_context()

            resources = df_resources['id'].values.tolist()

            nb_periods = N_PERIODS

            # periods range from 0 to nb_periods excluded
            periods = range(0, nb_periods)

            # days range from 0 to N_DAYS excluded
            days = range(0, N_DAYS)

            # start[r,t] is number of resource r starting to work at period t
            start = mdl.integer_var_matrix(keys1=resources, keys2=periods, name="start")

            # work[r,t] is number of resource r working at period t
            work = mdl.integer_var_matrix(keys1=resources, keys2=periods, name="work")

            # nr[r] is number of resource r working in total
            nr = mdl.integer_var_dict(keys=resources, name="nr")

            # nr[r,d] is number of resource r working on day d
            nrd = mdl.integer_var_matrix(keys1=resources, keys2=days, name="nrd")

            # Organize all decision variables in a DataFrame indexed by 'resources' and 'periods'
            df_decision_vars = pd.DataFrame({'start': start, 'work': work})

            # Set index names
            df_decision_vars.index.names = ['resources', 'periods']

            # Organize resource decision variables in a DataFrame indexed by 'resources'
            df_decision_vars_res = pd.DataFrame({'nr': nr})

            # Set index names
            df_decision_vars_res.index.names = ['resources']

            # available per day
            for r in resources:
                min_avail = int(df_resources[df_resources.id == r].min_avail)
                max_avail = int(df_resources[df_resources.id == r].max_avail)
                for d in range(N_DAYS):
                    mdl.add(
                        mdl.sum(start[r, t] for t in range(d * N_PERIODS_PER_DAY, (d + 1) * N_PERIODS_PER_DAY)) == nrd[
                            r, d])
                    mdl.add(nrd[r, d] <= nr[r])
                mdl.add(min_avail <= nr[r])
                mdl.add(nr[r] <= max_avail)

            # working
            for r in resources:
                duration = int(df_resources[df_resources.id == r].duration)
                for t in periods:
                    mdl.add(mdl.sum(start[r, t2] for t2 in range(max(t - duration, 0), t)) == work[r, t])

            # work vs demand
            for t in periods:
                demand = int(df_demands[df_demands.period == t]['demand'])
                mdl.add(mdl.sum(work[r, t] for r in resources) >= demand)

            total_cost = mdl.sum(int(df_resources[df_resources.id == r].cost) * nr[r] for r in resources)
            n_fix_used = nr['fix']
            n_temp_used = nr['temp']

            mdl.add_kpi(total_cost, "Total Cost")
            mdl.add_kpi(n_fix_used, "Nb Fix Used")
            mdl.add_kpi(n_temp_used, "Nb Temp Used")

            mdl.minimize(total_cost)

            solution = mdl.solve()

        # check the last solve details
        last = the_hook.solve_details[-1]
        self.assertIn('PROGRESS_BEST_OBJECTIVE', last,
                      'could not find PROGRESS_BEST_OBJECTIVE')
        self.assertIn('PROGRESS_CURRENT_OBJECTIVE', last,
                      'could not find PROGRESS_CURRENT_OBJECTIVE')
        self.assertAlmostEqual(last['PROGRESS_BEST_OBJECTIVE'],
                               last['PROGRESS_CURRENT_OBJECTIVE'],
                               delta=1e-4,
                               msg='Should have a PROGRESS_CURRENT_OBJECTIVE value')
        # end of test_defect_36015_objective_is_na(self)


if __name__ == "__main__":
    unittest.main()
