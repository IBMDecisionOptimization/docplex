# __author__ = 'couronne'

from __future__ import print_function

from six import iteritems

from collections import defaultdict

from pandas import DataFrame

from private.absmodel import AbstractModel
from docplex.mp.model import Model

from docplex.mp.sktrans.transformers import CplexTransformer
from private.timing import MyTimer
from examples.client_models.assignments.assignments import *
from docplex.mp.cplex_engine import CplexEngine

try:
    import scipy.stats as sci_stats

    def scaling_mean(npa):
        return sci_stats.gmean(npa)
except ImportError:
    def scaling_mean(npa):
        return npa.mean()


class KeyRecord(object):
    def __init__(self, i):
        self.i = i

    def __str__(self):
        return "k_%d" % self.i

    def __repr__(self):
        return str(self)

    def coef(self):
        return 2 if self.i % 2 == 0 else 3

    @property
    def index(self):
        return self.i

    def next(self):
        return KeyRecord(self.i + 1)

    def __hash__(self):
        return self.i


class ScalingModel(AbstractModel):
    def _register_timer(self, timer):
        self.timers[timer.header] = int(math.ceil(timer.elapsed_msecs))

    def __init__(self, nb_vars, time_verbose=True):
        AbstractModel.__init__(self, name="scaling")
        self.nb_vars = nb_vars
        self.time_verbose = time_verbose
        self.keys = [KeyRecord(i) for i in range(1, nb_vars + 1)]
        # each key has a next (circular)
        self.nexts = {key: self.keys[(k + 1) % nb_vars] for k, key in enumerate(self.keys)}
        # map of timers by header
        self.timers = {}
        self.vars = {}
        self.bvars = {}

    def setup_variables(self):
        self_time_verbose = self.time_verbose
        with MyTimer("vars/nobounds", verbose=self_time_verbose, print_details=False) as std_var_timer:
            self.vars = self.binary_var_dict(self.keys, name='z')
        self._register_timer(std_var_timer)
        with MyTimer("vars/bounds", verbose=self_time_verbose) as custom_var_timer:
            self.bvars = self.continuous_var_dict(self.keys, ub=100, name="q")
        self._register_timer(custom_var_timer)

    def setup_constraints(self):
        mdl = self
        self_time_verbose = self.time_verbose

        # 1 sum with a sequence
        with MyTimer("sum/vars/seq", verbose=self_time_verbose) as sum_seq_timer:
            mdl.add_constraint(mdl.sum(mdl.vars) >= 0)
        self._register_timer(sum_seq_timer)

        # 2 sum with comprehension
        with MyTimer("sum/vars/comp", verbose=self_time_verbose) as sum_comp_timer:
            mdl.add_constraint(mdl.sum(mdl.vars[k] for k in self.keys) >= 2)
        self._register_timer(sum_comp_timer)

        # 3 sum of monomials
        with MyTimer("sum/AX/comp", verbose=self_time_verbose) as sum_ax_comp_timer:
            mdl.add_constraint(mdl.sum(mdl.vars[k] * k.coef() for k in self.keys) >= 1000)
        self._register_timer(sum_ax_comp_timer)

        # 3 sum of monomials from sequence
        # do not time the buil dof the sequence
        with MyTimer("sum/AX/seq", verbose=self_time_verbose) as sum_ax_seq_timer:
            mdl.add_constraint(mdl.sum([mdl.vars[k] * k.coef() for k in self.keys]) >= 1000)
        self._register_timer(sum_ax_seq_timer)

        # the sequence is built before the sum (creating the monomial terms)
        all_monomials = [mdl.vars[k] * k.coef() for k in self.keys]
        with MyTimer("sum/AX/mnm", verbose=self_time_verbose) as sum_true_monoms:
            mdl.add_constraint(mdl.sum(all_monomials) <= 1)
        self._register_timer(sum_true_monoms)

        # use python sum() on a comprehension
        # with MyTimer("pysum/AX/comp", verbose=self_time_verbose) as pysum_ax_comp_timer:
        # mdl.add_constraint(sum(mdl.vars[k] * k.coef() for k in self.keys) >= 1000)
        # self._register_timer(pysum_ax_comp_timer)

        # 3 sum of complex expressions
        self_vars = self.vars
        self_nexts = self.nexts
        self_keys = self.keys
        with MyTimer("sum/exprs/comp", verbose=self_time_verbose) as sum_expr_comp_timer:
            mdl.add_constraint(mdl.sum(
                (self_vars[k] + 2 * self_vars[self_nexts[k]]) for k in self_keys) >= 1000)
        self._register_timer(sum_expr_comp_timer)

        # keep precomputation of arrays out of timed code.
        koefs = (k.coef() for k in self.keys)
        allvars = (mdl.vars[k] for k in self.keys)
        with MyTimer("scal_prod", verbose=self_time_verbose) as scal_prod_timer:
            mdl.add_constraint(mdl.scal_prod(allvars, koefs) >= 13)
        self._register_timer(scal_prod_timer)

    def get_times(self):
        return [ms for _, ms in iteritems(self.timers)]

    def get_times_dict(self):
        # returns a dict of header -> times
        return {h: ms for h, ms in iteritems(self.timers)}

    def setup_expr_sum(self):
        self_vars = self.vars
        self.add_constraint(
            self.sum(self_vars[k] + 2 * self_vars[self.nexts[k]] for k in self.keys) >= 3)

    def setup_exprs_comp(self, details=False):
        # 3 sum of complex expressions
        self_vars = self.vars
        self_nexts = self.nexts
        self_keys = self.keys
        mdl = self
        mdl.add_constraint(mdl.sum(
            [(self_vars[k] + 2 * self_vars[self_nexts[k]]) for k in self_keys]) >= 1000)


def run_scaling_dot(size_in_kvars=100, multiplier=1000, details=False):
    size = multiplier * size_in_kvars
    keys = [KeyRecord(i) for i in range(1, size + 1)]
    mdl = Model("scal_prod", checker='off')
    dvars = mdl.binary_var_dict(keys=keys, name='z')
    seq_coefs = [i*i for i in range(1, size+1)]
    seq_vars = [dvars[k] for k in keys]
    with MyTimer("scal_prod/seq_%dk" % size_in_kvars,print_details=details) as tt:
        mdl.dot(seq_vars, seq_coefs)
    return tt.elapsed

def run_scaling_dotf(size_in_kvars=100, multiplier=1000, details=False):
    size = multiplier * size_in_kvars
    keys = [KeyRecord(i) for i in range(1, size + 1)]
    mdl = Model("scal_prod", checker='off')
    lvars = mdl.binary_var_list(keys=size, name='z')
    with MyTimer("scal_prod_f/seq_%dk" % size_in_kvars,print_details=details) as tt:
        mdl.dotf(lvars, lambda k_: k_* k_)
    return tt.elapsed


def test_sum_of_axs(size_in_kvars, nb_of_kcts=1):
    """ Test performance of posting R rows with a sum of A*X terms


    :param size_in_kvars: the number of vars in the expressions (identical)
    :param nb_of_kcts: the number of kilo-constraints we post
    :return: the time in seconds.
    """
    print("* starting sum of axs, #vars=%dk, #cts=%dk" % (size_in_kvars, nb_of_kcts))
    # test R times the post of s sum(A_i*X_i)
    size = 1000 * size_in_kvars
    keys = [KeyRecord(i) for i in range(1, size + 1)]
    mdl = Model("sum_axs_%dkV_%dkC" % (size_in_kvars, nb_of_kcts), solver_agent='fail')
    dvars = mdl.continuous_var_dict(keys=keys, name='z')
    with MyTimer("scale_sum_AX_{0}_{1}".format(size_in_kvars, nb_of_kcts), verbose=True) as sumaxs_tt:
        for _ in range(1000 * nb_of_kcts):
            mdl.add_constraint(mdl.sum(dvars[k] * k.coef() for k in keys) >= 0.0)
    print("* created %d instances of LinearExpr" % mdl.number_of_linear_expr_instances)
    return sumaxs_tt.secs

def run_scaling_sum_vars(size_in_kvars=1, nb_of_cts=10):
    """ Test performance of posting R rows with a sum of A*X terms


    :param size_in_kvars: the number of vars in the expressions (identical)
    :param nb_of_cts: the number of kilo-constraints we post
    :return: the time in seconds.
    """
    print("* starting sum of xs, #vars=%dk, #cts=%dk" % (size_in_kvars, nb_of_cts))
    # test R times the post of s sum(A_i*X_i)
    nb_vars = 1000 * size_in_kvars
    mdl = Model("sum_xs_%dkV_%dkC" % (size_in_kvars, nb_of_cts),keep_ordering=True)
    dvars = mdl.continuous_var_list(keys=nb_vars)
    with MyTimer("scale_sum_X_{0}k vars *{1}k".format(size_in_kvars, nb_of_cts), print_details=False) as tt:
        for _ in range(nb_of_cts):
            mdl.add_constraint(mdl.sum(dvars) >= 0.0)
    return tt.secs


def run_scaling_is_mip(size_in_kvars=1, detailed=False):

    print("* starting sum of xs, #vars=%dk" % size_in_kvars)
    # test R times the post of s sum(A_i*X_i)
    nb_vars = 1000 * size_in_kvars
    mdl = Model("sismip_xs_%dkV" % size_in_kvars)
    dvars = mdl.continuous_var_list(keys=nb_vars)
    mdl.integer_var(name="mipp")
    with MyTimer("is_mip_X_{0}k vars".format(size_in_kvars), print_details=detailed) as tt:
        for _ in range(nb_vars):
            mdl._solved_as_mip()
    return tt.secs

def nth_plus(seq, lim):
    cur = seq[0]
    for i in range(1, lim):
        cur = cur.plus(seq[i])
    return cur

def scale_nplus(plus_size=100, size_in_kvars=1, nb_of_kcts=1, **kwargs):
    print("* starting nplus_sum, #vars=%dk, #cts=%dk" % (size_in_kvars, nb_of_kcts))
    # test R times the post of s sum(A_i*X_i)
    size = 1000 * size_in_kvars
    keys = [KeyRecord(i) for i in range(1, size + 1)]
    mdl = Model(name="sum_axs_%dkV_%dkC" % (size_in_kvars, nb_of_kcts), **kwargs)
    dvars = mdl.continuous_var_list(keys=keys, name='z')
    with MyTimer("scale_nthplus_{0}_{1}".format(plus_size, nb_of_kcts), verbose=True) as sumnplus:
        for k in range(1000 * nb_of_kcts):
            mdl.add_constraint(nth_plus(dvars, plus_size) >= 0)
    print("* created %d instances of LinearExpr" % mdl.number_of_linear_expr_instances())
    return sumnplus.secs

def scale_nplus_mnm(plus_size=100, size_in_kvars=1, nb_of_kcts=1, **kwargs):
    print("* starting nplus_sum_monomials, #vars=%dk, #cts=%dk" % (size_in_kvars, nb_of_kcts))
    # test R times the post of s sum(A_i*X_i)
    size = 1000 * size_in_kvars
    keys = [KeyRecord(i) for i in range(1, size + 1)]
    mdl = Model(name="sum_axs_%dkV_%dkC" % (size_in_kvars, nb_of_kcts), **kwargs)
    dvars = mdl.continuous_var_list(keys=keys, name='z')
    mnms = [(i+3) * dvars[i] for i in range(size)]
    with MyTimer("scale_nthplus_{0}_{1}".format(plus_size, nb_of_kcts), verbose=True) as sumnplus:
        for k in range(1000 * nb_of_kcts):
            mdl.add_constraint(nth_plus(mnms, plus_size) >= 0)
    print("* created %d instances of LinearExpr" % mdl.number_of_linear_expr_instances)
    return sumnplus.secs


def run_scaling_create_default_vars(size_in_k=1, details=False, **kwargs):
    m = Model(**kwargs)
    with MyTimer("running scaling tests nb_vars=%dk" % size_in_k, print_details=details):
        m.binary_var_list(size_in_k*1000)

def run_scaling_create_named_vars(size_in_k=1, details=False, **kwargs):
    m = Model(**kwargs)
    with MyTimer("running scaling tests nb_vars=%dk" % size_in_k, print_details=details):
        m.continuous_var_list(size_in_k*1000, name='xxx')

def run_scaling_comp_exprs(size, multiplier=1000, details=False):
    scaler = ScalingModel(nb_vars=size * multiplier)
    scaler._keep_all_exprs = False
    scaler.setup_variables()
    with MyTimer("running scaling tests nb_vars = %d k" % size, print_details=details):
        scaler.setup_exprs_comp()

def run_scaling_expr_sum(size, multiplier=1000):
    scaler = ScalingModel(nb_vars=size * multiplier)
    instance_origin = scaler._linexpr_instance_counter
    cloned_origin = scaler._linexpr_clone_counter
    scaler.setup_variables()
    with MyTimer("running scaling tests sum(exprs) = %d k" % size, print_details=False) as tb:
        scaler.setup_expr_sum()
    scaler.end()
    del scaler
    print("-- #exprs: {}".format(scaler._linexpr_instance_counter - instance_origin))
    print("-- #cloned: {}".format(scaler._linexp_clone_counter - cloned_origin))
    return tb.elapsed_msecs


def run_scaling_lp_print(size, multiplier=1000):
    scaler = ScalingModel(nb_vars=size * multiplier)
    scaler.setup()
    with MyTimer("--> print lp") as lpt:
        scaler.export_as_lp()
    return lpt.msecs


def run_scaling(size, multiplier=1000, nb_repeat=3, verbose=True):
    # need to memorize the initial count value

    repeat_times = defaultdict(list)
    for r in range(1, nb_repeat + 1):
        scaler = ScalingModel(nb_vars=size * multiplier, time_verbose=verbose)
        with MyTimer("running scaling tests nb_vars=%dk" % size, verbose=verbose):
            scaler.setup()
            times_dict = scaler.get_times_dict()
        for h, tt in iteritems(times_dict):
            repeat_times[h].append(tt)
        scaler.end()
        del scaler

    res_times = {}
    for h, ttimes in iteritems(repeat_times):
        np_times = np.array(ttimes)
        np_mean_times = scaling_mean(np_times)
        res_times[h] = int(math.ceil(np_mean_times))

    return res_times


sorted_topics = ["vars/nobounds", "vars/bounds",
                 "sum/vars/seq", "sum/vars/comp",
                 "sum/AX/comp", "sum/AX/seq", "sum/AX/mnm",
                 "sum/exprs/comp",
                 "scal_prod",
                 "nb_exprs"]
time_topics = [c for c in sorted_topics if not c.startswith("nb")]


def run_iterated_scaling(ns, multiplier=1000, nb_repeat=3, run_verbose=False):
    """ Iterates scaling over a series of sizes.

    ArgsL
        ns: a sequence of integer sizes (in 1000s).
        multiplier: usually 1000, used to multiply size to get the number of artefacts
        nb_repeat: number of times each timed operation is repeated.
            final result is obtained by taking the geomertic mean of all times.
    Return:
        a pandas dataframe. Columns are timer headers (e.g. vars/bounds).
            Index is sequence of sizes (10, 30, ...)

    """
    all_res = defaultdict(dict)

    for n in ns:
        print("-- run scaling for size: {0}, repeat-factor={1}".format(n, nb_repeat))
        # returns a dict of {topic: time}
        n_times = run_scaling(size=n, multiplier=multiplier, nb_repeat=nb_repeat, verbose=run_verbose)
        all_res[n] = n_times

    # here we have a dict of {n: {h: tt}
    df_ref = DataFrame(all_res, index=sorted_topics)
    # transpose to get a df of topic -> size > time
    df_t = df_ref.T
    df_times = df_t[time_topics]
    # sum() of times, not of counts
    df_t["total"] = df_times.sum(axis=1)
    wpi = compute_wpi(df_t)
    print("* wpi for sizes={0} is={1}".format(ns, wpi))
    return df_t


def generic_scale(ns, scaling_fn, setup_fn=lambda m: m, is_verbose=False):
    res = {}
    for n in ns:
        m = Model()
        setup_fn(m)
        with MyTimer(verbose=is_verbose) as tt:
            scaling_fn(m, n)
        res[n] = tt.elapsed_msecs
    return res


def scale_iadd(ns):
    res = {}
    for n in ns:
        mdl = Model()
        xs = mdl.binary_var_list(n)

        with MyTimer(header="iadd_%d" % n) as tt:
            # repeat n times  e+= v[i]
            e = 0
            for var in xs:
                e += var
        res[n] = tt.elapsed_msecs
    return res


def scale_isub(ns):
    res = {}
    for n in ns:
        mdl = Model()
        xs = mdl.binary_var_list(n)

        with MyTimer(header="isub_%d" % n) as tt:
            # repeat n times  e+= v[i]
            for i in range(n):
                e = 0
                for var in xs:
                    e -= var
        res[n] = tt.elapsed_msecs
    return res


def scale_pysum(ns):
    res = {}
    for n in ns:
        mdl = Model()
        xs = mdl.binary_var_list(n)

        with MyTimer(".") as tt:
            e = sum(xs)
        mdl.end()
        del mdl

        res[n] = tt.elapsed_msecs
    return res


def scale_msum(ns):
    res = {}
    for n in ns:
        mdl = Model()
        xs = mdl.binary_var_list(n)

        with MyTimer(header="docplex_sum_%d" % n) as tt:
            mdl.sum(xs)
        mdl.end()
        del mdl
        res[n] = tt.elapsed_msecs
    return res

def scale_var_var_constraints(size_in_k=1, details=False, **kwargs):
    # fixed nb of vars
    with Model(**kwargs) as mdl:
        nb_vars = 1000
        xs = mdl.continuous_var_list(keys=nb_vars)
        with MyTimer('x_i <= x_i+1  repeats %dk times' % size_in_k, print_details=details) as tt:
            mdl.add_constraints( xs[ i % nb_vars] >= xs[ (i+1) % nb_vars] for i in range(1000 * size_in_k))
        return tt.elapsed_msecs

def scale_var_plus_var_constraints(size_in_k=1, details=False, **kwargs):
    # fixed nb of vars
    with Model(**kwargs) as mdl:
        nb_vars = 1000
        xs = mdl.continuous_var_list(keys=nb_vars)
        with MyTimer('x_i + x_i+1 <= 1  repeats %dk times' % size_in_k, print_details=details) as tt:
            mdl.add_constraints( (xs[ i % nb_vars] + xs[ (i+1) % nb_vars] <= 1) for i in range(1000 * size_in_k))
        return tt.elapsed_msecs

def scale_var_minus_var_constraints(size_in_k=1, details=False, **kwargs):
    # fixed nb of vars
    with Model(**kwargs) as mdl:
        nb_vars = 1000
        xs = mdl.continuous_var_list(keys=nb_vars)
        with MyTimer('x_i - x_i+1 <= 1  repeats %dk times' % size_in_k, print_details=details) as tt:
            mdl.add_constraints( (xs[ i % nb_vars] - xs[ (i+1) % nb_vars] <= 1) for i in range(1000 * size_in_k))
        return tt.elapsed_msecs


def scale_var_mn_constraints(size_in_k=1, details=False, **kwargs):
    # fixed nb of vars
    with Model(**kwargs) as mdl:
        nb_vars = 1000
        xs = mdl.continuous_var_list(keys=nb_vars)
        with MyTimer('x1 >= -x2 repeats %dk times' % size_in_k, print_details=details) as tt:
            mdl.add_constraints( xs[ i % nb_vars] >= -xs[ (i+1) % nb_vars] for i in range(1000 * size_in_k))
        return tt.elapsed

def scale_constant_linear_exprs(size_in_k=1, details=False, **kwargs):
    mdl = Model(name='constant_linexprs_%dk' % size_in_k)
    xs = mdl.continuous_var_list(10)
    with MyTimer('create %dk constant linear exprs', print_details=details) as tt:
        for k in range(size_in_k*1000):
            mdl.linear_expr(arg=k)
    return tt.elapsed

def scale_constant_exprs(size_in_k=1, details=False, **kwargs):
    mdl = Model(name='constant_linexprs_%dk' % size_in_k)
    xs = mdl.continuous_var_list(10)
    lfact = mdl._lfactory
    with MyTimer('create %dk constant linear exprs', print_details=details) as tt:
        for k in range(size_in_k*1000):
            lfact._new_constant_expr(k)
    return tt.elapsed

def scale_nurse_build(size=1, details=False):
    from examples.delivery.modeling.nurses import build

    with MyTimer('build_nurses_%d' % size, print_details=details) as tt:
        for _ in range(size):
            build(verbose=False, ignore_names=True, checker='off')
        return tt.elapsed


def scale_spss(size_in_k=30, verbose=False):
    from examples.client_models.spss_campaign import RandomCampaignModel

    print("* scale spss model size = {0}k".format(size_in_k))
    campaign_mdl = RandomCampaignModel(size=size_in_k * 1000)
    spss_times = []
    with MyTimer(header="  variables", verbose=verbose) as spss_v:
        campaign_mdl.setup_variables()
    spss_times.append(spss_v.elapsed_msecs)

    with MyTimer(header="  constraints", verbose=verbose) as spss_c:
        campaign_mdl.setup_constraints()
    spss_times.append(spss_c.elapsed_msecs)

    with MyTimer(header="  objective", verbose=verbose) as spss_o:
        campaign_mdl.setup_objective()
    spss_times.append(spss_o.elapsed_msecs)

    campaign_mdl.end()
    del campaign_mdl

    return spss_times


def scale_pwl_ct2(size_in_k=1, detailed=False, **kwargs):
    # fixed nb of vars
    with Model(**kwargs) as mdl:
        nb_vars = 1000*size_in_k
        # create N x vars
        xs = mdl.continuous_var_list(nb_vars)
        ys = mdl.continuous_var_list(nb_vars)
        pwfs = [mdl.piecewise(0, [(i, 10), (2*i+1, 20)], 0) for i in range(nb_vars)]
        with MyTimer('adding {0}k cts of type y==pw(x)'.format(size_in_k), print_details=detailed) as tt:
            for p in range(nb_vars):
                mdl.add(ys[p] == pwfs[p](xs[p]))
        return tt.elapsed_msecs

def scale_pwl_ct1(size_in_k=1, detailed=False, **kwargs):
    # fixed nb of vars
    with Model(**kwargs) as mdl:
        nb_vars = 1000*size_in_k
        # create N x vars
        xs = mdl.continuous_var_list(nb_vars)
        ys = mdl.continuous_var_list(nb_vars)
        pwfs = [mdl.piecewise(0, [(i, 10), (2*i+1, 20)], 0) for i in range(nb_vars)]
        with MyTimer('adding {0}k cts of type Model.piecewise(y, f, x)'.format(size_in_k), print_details=detailed) as tt:
            for p in range(nb_vars):
                mdl.add_piecewise_constraint(ys[p], pwfs[p], xs[p])
        return tt.elapsed_msecs


def scale_assignment_model(n_targets, n_sources_in_k=1, detailed=False):
    # define a very simple assignment model from SRC to TGT
    # we have NS * NT binary variables with
    #  -- cardinality max  per target: forall t  sum(a[s,t]) <= NS/2 (not more than 50% of customers on each tgt)
    #  --- cardinality min per source: at least 5 tget for each s
    nt = n_targets

    with MyTimer("assignment_ns_{0}k_nt_{1}".format(n_sources_in_k, nt), print_details=detailed) as tt:
        make_assignment_model(nt, n_sources_in_k, multiplier=1000, cost_fn=None)


def scale_assignment_cpx_trans(n_targets, n_sources_in_k=1, detailed=False):
    # define a very simple assignment model from SRC to TGT
    # we have NS * NT binary variables with
    #  -- cardinality max  per target: forall t  sum(a[s,t]) <= NS/2 (not more than 50% of customers on each tgt)
    #  --- cardinality min per source: at least 5 tget for each s
    nt = n_targets

    with MyTimer("assignment_cpxt_ns_{0}k_nt_{1}".format(n_sources_in_k, nt), print_details=detailed) as tt:
        amm = make_sparse_assignment_matrix(n_sources_in_k, nt, multiplier=1000)
        lpt = CplexTransformer(sense="min", modeler="cplex")
        lpt.transform(amm, ubs=1, solve=False)

def make_full_numpy_matrix(nr, nc):
    import numpy as np
    # create a numpy matrix with
    def kij(i, j) :
        return (13*i + 23*j + 11 % 1103) / 4
    rows = [[ kij(r, c) for c in range(nc)] for r in range(nr)]
    return np.array(rows)



def scale_full_matrix_transformer(n_rows_in_k=1, ncols_in_k=1, detailed=False):
    # make a dumb nrxnc matrix with dumb rhs
    nm = make_full_numpy_matrix(n_rows_in_k * 1000, ncols_in_k * 1000+1)
    print("matrix shape : {0}".format(nm.shape))
    with MyTimer("full_matrix_cpxt_nr_{0}k_nc_{1}k".format(n_rows_in_k, ncols_in_k), print_details=detailed) as tt:
        lpt = CplexTransformer(sense="min", modeler="cplex")
        lpt.transform(nm, solve=False)

def scale_indicators(n_indicators_in_k=1, procedural=True, detailed=False):
    with Model(name='indicators_%dk' % n_indicators_in_k, checker='off') as m:
        # create vars
        size = n_indicators_in_k * 1000
        bs = m.binary_var_list(size)
        xs = m.continuous_var_list(size)
        CplexEngine.procedural = procedural
        p = 'procedural' if procedural else 'non-procedural'
        with MyTimer('create_%dk indicators: %s' %
                     (n_indicators_in_k, p), print_details=detailed) as tt:
            inds = m.add_indicators(bs, (xs[k] >= k for k in range(size)))
    return tt.elapsed

def time_model_read(path, verbose=False, details=True):
    from docplex.mp.model_reader import ModelReader

    msg = "-> reading file %s" % path
    with MyTimer(header=msg, verbose=verbose, print_details=details) as tt:
        mdl = ModelReader.read(path, verbose=verbose)
    print("<- read model: {0}, time={1} s.".format(path, tt.elapsed_secs))
    return mdl, tt.elapsed_secs


def print_as_dict(df, argname=None):
    if argname is not None:
        print("{0} = {{".format(argname))
    else:
        print("{")
    df_cols = df.columns
    nb_cols = len(df_cols)
    max_col_len = max(len(c) for c in df_cols)
    for c, topic in enumerate(df_cols):
        comma = "," if c < nb_cols - 1 else ""
        spaces = " " * (max_col_len - len(topic))
        # "topic"  : [1,2,3,4],  # comma if not last
        print("    \"{0:s}\"{3}: {1}{2}".format(topic, df[topic].tolist(), comma, spaces, w=max_col_len))
    print("}")


SIZES = [10, 30, 50, 100, 130, 150, 180]
SIZES135 = [1, 3, 5, 10]


def compute_wpi(df):
    """ Returns the weight performance index

    :param df: a pandas dataframe
    :return:
    """
    idx = df.index
    time_columns = [c for c in sorted_topics if not c.startswith("nb")]
    # do not count nb_exprs columns
    df_times = df[time_columns]
    cumtimes = df_times.sum(axis=1)  # sum on rows
    wsum = sum(n * cumtimes[n] for n in idx)
    sum_of_ns = float(sum(k for k in idx))
    return math.ceil(wsum / sum_of_ns)


# ---- current ref

# * wpi for sizes=[10, 30, 50, 100, 130, 150, 180] is=8305.0

refdata = {
    "vars/nobounds" : [66, 209, 310, 688, 903, 1077, 1301],
    "vars/bounds"   : [63, 238, 401, 885, 1135, 1279, 1524],
    "sum/vars/seq"  : [12, 37, 58, 122, 178, 184, 232],
    "sum/vars/comp" : [17, 55, 89, 184, 256, 279, 342],
    "sum/AX/comp"   : [39, 121, 200, 437, 554, 620, 769],
    "sum/AX/seq"    : [42, 147, 240, 571, 740, 818, 1076],
    "sum/AX/mnm"    : [16, 51, 84, 177, 231, 263, 330],
    "sum/exprs/comp": [306, 966, 1495, 3040, 4048, 4580, 5401],
    "scal_prod"     : [21, 66, 109, 227, 302, 349, 417],
    "nb_exprs"      : [10015, 30015, 50015, 100015, 130015, 150015, 180015],
    "total"         : [582, 1890, 2986, 6331, 8347, 9449, 11392]
}


df_scaleref = DataFrame(refdata, index=SIZES)
ref_wpi = compute_wpi(df_scaleref)

if __name__ == "__main__":
    from optparse import OptionParser
    import sys

    SIZE = 50


    parser = OptionParser()
    parser.add_option("-s", "--size", dest="size", action="store", type="int", default=SIZE, help="set size")
    options, args = parser.parse_args(sys.argv)
    nkv = 3000
    # run_scaling_dot(size_in_kvars=nkv)
    # run_scaling_dotf(size_in_kvars=nkv)
    #run_scaling_is_mip(size_in_kvars=3, detailed=False)
    #scale_indicators(n_indicators_in_k=300, procedural=True, detailed=False)
    #scale_indicators(n_indicators_in_k=100, procedural=False)

    # scale_constant_linear_exprs(size_in_k=1000)
    # scale_constant_exprs(size_in_k=1000, details=False)
    #scale_nurse_build(size=30, details=True)

    run_scaling_create_named_vars(size_in_k=300, details=False, checker='off')
    #scale_assignment_model(n_sources_in_k=100, n_targets=50, detailed=False)
    #scale_full_matrix_transformer(n_rows_in_k=20, ncols_in_k=10, detailed=False)

    #run_scaling_comp_exprs(size=150, details=False)
    # print("* reference wpi is: {}".format(ref_wpi))
    #rr = run_iterated_scaling(ns=[10, 30, 50, 100, 130, 150, 180], nb_repeat=4, run_verbose=True)
    # # computing wpi
    # relative_wpi_delta = 100 * (compute_wpi(rr) - ref_wpi) / float(ref_wpi)
    # print(">>> wpi delta is: {0:.2f}%".format(relative_wpi_delta))
    # print_as_dict(rr, argname="refdata")
    #run_scaling_comp_exprs(size=150, details=True)


# as of 27/04/2016
# * wpi for sizes=[10, 30, 50, 100, 130, 150, 180] is=8281.0
# >>> wpi delta is: -10.52%
# perfdata = {
#     "vars/nobounds" : [66, 213, 314, 696, 921, 1077, 1310],
#     "vars/bounds"   : [65, 240, 404, 889, 1114, 1287, 1519],
#     "sum/vars/seq"  : [13, 37, 60, 124, 158, 182, 231],
#     "sum/vars/comp" : [18, 54, 90, 189, 243, 281, 347],
#     "sum/AX/comp"   : [40, 121, 205, 424, 538, 611, 747],
#     "sum/AX/seq"    : [43, 137, 242, 561, 725, 817, 1070],
#     "sum/AX/mnm"    : [18, 52, 85, 184, 232, 263, 329],
#     "sum/exprs/comp": [307, 967, 1526, 3112, 3965, 4540, 5412],
#     "scal_prod"     : [21, 67, 113, 239, 298, 349, 423],
#     "nb_exprs"      : [10015, 30015, 50015, 100015, 130015, 150015, 180015],
#     "total"         : [591, 1888, 3039, 6418, 8194, 9407, 11388]
# }

# as of 24/03/2016
# wpi for sizes=[10, 30, 50, 100, 130, 150, 180] is=8592.0
# perfdata = {
#     "vars/nobounds" : [57, 187, 317, 697, 927, 1028, 1290],
#     "vars/bounds"   : [64, 244, 408, 894, 1139, 1245, 1555],
#     "sum/vars/seq"  : [12, 41, 62, 123, 158, 181, 227],
#     "sum/vars/comp" : [18, 62, 100, 195, 255, 294, 362],
#     "sum/AX/comp"   : [43, 136, 233, 433, 581, 660, 802],
#     "sum/AX/seq"    : [40, 160, 324, 565, 735, 876, 1085],
#     "sum/AX/mnm"    : [17, 60, 94, 188, 250, 278, 354],
#     "sum/exprs/comp": [310, 986, 1781, 3062, 4075, 4538, 5395],
#     "scal_prod"     : [27, 93, 119, 357, 478, 601, 728],
#     "nb_exprs"      : [10015, 30015, 50015, 100015, 130015, 150015, 180015],
#     "total"         : [588, 1969, 3438, 6514, 8598, 9701, 11798]
# }



# as of 230/03/2016
# * wpi for sizes=[10, 30, 50, 100, 130, 150, 180] is=8272.0
# >>> wpi delta is: -10.62%
# perfdata = {
#     "vars/nobounds" : [74, 220, 330, 674, 904, 1008, 1205],
#     "vars/bounds"   : [68, 250, 431, 883, 1129, 1280, 1519],
#     "sum/vars/seq"  : [13, 37, 62, 121, 173, 190, 226],
#     "sum/vars/comp" : [20, 57, 102, 190, 265, 279, 340],
#     "sum/AX/comp"   : [44, 127, 218, 436, 533, 612, 746],
#     "sum/AX/seq"    : [48, 149, 288, 594, 710, 831, 1054],
#     "sum/AX/mnm"    : [18, 53, 90, 181, 225, 261, 325],
#     "sum/exprs/comp": [330, 1118, 1708, 3120, 3991, 4516, 5452],
#     "scal_prod"     : [23, 67, 128, 232, 292, 336, 417],
#     "nb_exprs"      : [10015, 30015, 50015, 100015, 130015, 150015, 180015],
#     "total"         : [638, 2078, 3357, 6431, 8222, 9313, 11284]
# }
# older
# * wpi for sizes=[10, 30, 50, 100, 130, 150, 180] is=9255.0
# refdata = {
#     "vars/nobounds": [70, 264, 406, 864, 1226, 1526, 1702],
#     "vars/bounds": [87, 285, 514, 1032, 1408, 1629, 1933],
#     "sum/vars/seq": [12, 38, 63, 143, 172, 200, 255],
#     "sum/vars/comp": [20, 61, 98, 205, 266, 311, 406],
#     "sum/AX/comp": [43, 131, 217, 441, 571, 662, 844],
#     "sum/AX/seq": [43, 154, 255, 596, 657, 764, 1204],
#     "sum/AX/mnm": [18, 56, 93, 192, 251, 290, 358],
#     "sum/exprs/comp": [320, 917, 1621, 3022, 3948, 4569, 5539],
#     "scal_prod": [23, 129, 117, 331, 504, 677, 712],
#     "nb_exprs": [10015, 30015, 50015, 100015, 130015, 150015, 180015],
#     "total": [636, 2035, 3384, 6826, 9003, 10628, 12953]
# }
