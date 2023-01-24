# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2022
# --------------------------------------------------------------------------

import os
import time

# docplex
from docplex.mp.model import Model
from docplex.mp.utils import DOcplexException, MockIterable
from docplex.mp.environment import Environment

from docplex.mp.params.cplex_params import get_params_from_cplex_version
from docplex.mp.constants import ComparisonType
from docplex.mp.constr import LinearConstraint

from docplex.mp.cplex_adapter import CplexAdapter
from docplex.mp.cplex_engine import CplexEngine

from docplex.mp.quad import VarPair

import os
import xml.etree.ElementTree as ET
import tempfile



class ModelReaderError(DOcplexException):
    pass


class _CplexReaderFileContext(object):
    def __init__(self, filename, read_method=None):
        self._cplex = None
        self._filename = filename
        self._read_method = read_method or ["read"]

    def __enter__(self):
        self.cpx_adapter = CplexAdapter()
        cpx = self.cpx_adapter.cpx
        # no output from CPLEX
        cpx.set_results_stream(None)
        cpx.set_log_stream(None)
        cpx.set_warning_stream(None)
        cpx.set_error_stream(None)
        self_read_fn = cpx
        for m in self._read_method:
            self_read_fn = self_read_fn.__getattribute__(m)

        try:
            self_read_fn(self._filename)
            self._cplex = cpx
            return self.cpx_adapter

        except self.cpx_adapter.CplexError as cpx_e:  # pragma: no cover
            # delete cplex instance
            del cpx
            raise ModelReaderError("*CPLEX error {0!s} reading file {1} - exiting".format(cpx_e, self._filename))

    # noinspection PyUnusedLocal
    def __exit__(self, exc_type, exc_val, exc_tb):
        cpx = self._cplex
        if cpx is not None:
            del cpx
            self._cplex = None


# def compute_full_dotf(mdl, coefs, constant=0):
#     def coef_fn(dvx):
#         return coefs[dvx]
#
#     lfactory = mdl._lfactory
#     terms_dict = lfactory._new_term_dict()
#     for dv in mdl.iter_variables():
#         coef = coef_fn(dv.index)
#         if coef:
#             terms_dict[dv] = coef
#     linear_expr = lfactory.linear_expr(terms_dict, constant=constant, safe=True)
#     return linear_expr


def compute_full_dot(mdl, coefs, constant=0):
    # compute a scalar product, with possible zeros as coefs
    lfactory = mdl._lfactory
    terms_dict = lfactory._new_term_dict()
    for dv, k in zip(mdl.iter_variables(), coefs):
        if k:
            terms_dict[dv] = k
    linear_expr = lfactory.linear_expr(terms_dict, constant=constant, safe=True)
    return linear_expr


# noinspection PyArgumentList
class ModelReader(object):
    """ This class is used to read models from CPLEX files (e.g.  SAV, LP, MPS)
    This class requires CPLEX runtime, otherwise an exception is raised.

    Example:
        Use class method ``read`` to read a model file.

        >>> m = ModelReader.read('mymodel.lp', ignore_names=True)

        reads a model while ignoring all names.

        Global function ``read_model`` is a synonym for ``ModelReader.read``,
        and is the preferred way to read model files.

    """

    @staticmethod
    def _build_linear_expr_from_sparse_pair(lfactory, var_map, cpx_sparsepair):
        terms = {var_map[ix]: k for ix, k in zip(cpx_sparsepair.ind, cpx_sparsepair.val)}
        expr = lfactory.linear_expr(arg=terms, safe=True)
        return expr

    _sense2comp_dict = {'L': ComparisonType.LE, 'E': ComparisonType.EQ, 'G': ComparisonType.GE}

    # noinspection PyDefaultArgument
    @classmethod
    def parse_sense(cls, cpx_sense, sense_dict=_sense2comp_dict):
        return sense_dict.get(cpx_sense)

    @classmethod
    def read_ops_file(cls, filename, version=None):
        """ Reads an OPL .ops setting file and builds a CPLEX parameter group.

        Reads a .ops file and returns a DOcplex parameter group instance.
        All CPO and OPL settings are ignored. This parameter object can be used in a solve().
        .ops file have dedicated User Interface in OPL IDE and Cloud Pak for Data Experiments, which ease
        the selection of engine settings.

        Args:
            filename: a path string
            version: optional CPLEX version.

        Returns:
            A `RootParameterGroup object`, if the read operation succeeds, else None.
        """
        # We parse the XML file to get the list of cplex params names in OPL naming convention and their value.
        # We rely on the old 12.x CPLEX naming convention CPX_PARAM_ instead of standard CPXPARAM_ as in OPL runtime to rebuild the names.
        # We build a temporary .prm and import it in docplex with existing method.
        # We can't do a cleaner way as not all CPLEX C methods related to settings are available in python.
        # Adding them would only work in a next future version and not previous ones...

        tree = ET.parse(filename)
        root = tree.getroot()
        # Find the CPLEX section
        # < settings version = "2" >
        # < category name = "cplex" >
        # < setting  name = "advind"  value = "2" / >

        cplex_node = None
        if root.tag == "settings":
            for child in root:
                if child.tag == "category" and child.attrib.get("name", "") == "cplex":
                    cplex_node = child
                    break
        values = {}
        if cplex_node is not None:
            for child in cplex_node:
                if child.tag == "setting":
                    if "name" in child.attrib and "value" in child.attrib:
                        # Some settings have different names from OPL to CPLEX
                        def get_name(name):
                            if name == "rootalg":
                                name = "STARTALG"
                            elif name == "nodealg":
                                name = "SUBALG"
                            return "CPX_PARAM_" + name.upper()

                        values[get_name(child.attrib["name"].lower())] = child.attrib["value"]
                    else:
                        raise Exception("Bad XML .ops file: bad attributes {0}".format(child.attrib))
                else:
                    raise Exception("Bad XML .ops file: unknown tag {0}".format(child.tag))
        else:
            return None

        # clean up param input
        def boolean_to_int(b):
            if b == "true":
                return 1
            elif b == "false":
                return 0
            else:
                return b

        values = {k: boolean_to_int(v) for k, v in values.items()}

        with tempfile.NamedTemporaryFile(suffix=".prm", delete=False) as prm_file:
            if version is not None:
                prm_file.write("CPLEX Parameter File Version {}\n".format(version).encode('utf-8'))
            for k, v in values.items():
                prm_file.write("{0} {1}\n".format(k, v).encode('utf-8'))

        prms = ModelReader.read_prm(prm_file.name)
        os.remove(prm_file.name)
        return prms
    @classmethod
    def read_prm(cls, filename):
        """ Reads a CPLEX PRM file.

        Reads a CPLEX parameters file and returns a DOcplex parameter group
        instance. This parameter object can be used in a solve().

        Args:
            filename: a path string

        Returns:
            A `RootParameterGroup object`, if the read operation succeeds, else None.
        """
        with _CplexReaderFileContext(filename, read_method=["parameters", "read_file"]) as adapter:
            cpx = adapter.cpx
            if cpx:
                # raw parameters
                params = get_params_from_cplex_version(cpx.get_version())
                for param in params:
                    try:
                        cpx_value = cpx._env.parameters._get(param.cpx_id)
                        if cpx_value != param.default_value:
                            param.set(cpx_value)

                    except adapter.CplexError:  # pragma: no cover
                        pass
                return params
            else:  # pragma: no cover
                return None

    @staticmethod
    def _safe_call_get_names(cpx_adapter, get_names_fn, fallback_names=None):
        # cplex crashes when calling get_names on some files (e.g. SAV)
        # in this case filter out error 1219
        # and return a fallback list with None or ""
        try:
            names = get_names_fn()
            return names
        # except TypeError:
        #     print("** type error ignored in call to {0}".format(get_names_fn.__name__))
        #     return fallback_names or []

        except cpx_adapter.CplexSolverError as cpxse:  # pragma: no cover
            errcode = cpxse.args[2]
            # when all indicators have no names, cplex raises this error
            # CPLEX Error  1219: No names exist.
            if errcode == 1219:
                return fallback_names or []
            else:
                # this is something else
                raise

    @classmethod
    def _read_cplex(cls, filename, datacheck=None, silent=True):
        cpx_adapter = CplexAdapter()
        cpx = cpx_adapter.cpx
        # no warnings
        if silent:
            cpx.set_results_stream(None)
            cpx.set_log_stream(None)
            cpx.set_warning_stream(None)
            cpx.set_error_stream(None)  # remove messages about names
        try:
            if datacheck is not None:
                assert datacheck in {0,1,2}
                cpx.parameters.read.datacheck.set(datacheck)
            t0 = time.time()
            cpx.read(filename)
            t1 = time.time() - t0
            if Model.is_docplex_debug() and t1 >= 0.1:
                print(f"-- cplex read time={t1:.1f}s")
            return cpx_adapter
        except cpx_adapter.CplexError as cpx_e:
            raise ModelReaderError("*CPLEX error {0!s} reading file {1} - exiting".format(cpx_e, filename))

    @classmethod
    def _make_expr_from_varmap_coefs_dict(cls, lfactory, varmap, var_indices, coefs, offset=0):
        terms_dict = {varmap.get(dvx): k for dvx, k in zip(var_indices, coefs)}
        return lfactory.linear_expr(arg=terms_dict, constant=offset, safe=True)

    @classmethod
    def _make_expr_from_var_coef_seq_dict(cls, lfactory, varmap, var_coefs, constant=0):
        terms_dict = {varmap.get(dvx): k for dvx, k in var_coefs}
        return lfactory.linear_expr(arg=terms_dict, constant=constant, safe=True)

    @classmethod
    def _make_expr_from_varmap_coefs_nodict(cls, lfactory, varmap, var_indices, coefs, offset=0):
        terms_dict = lfactory._new_term_dict()
        for dvx, k in zip(var_indices, coefs):
            dv = varmap.get(dvx)
            if dv is not None:
                terms_dict[dv] = k
        return lfactory.linear_expr(arg=terms_dict, constant=offset, safe=True)

    @classmethod
    def _make_expr_from_var_coef_seq_nodict(cls, lfactory, varmap, var_coefs, constant=0):
        terms_dict = lfactory._new_term_dict()
        for dvx, k in var_coefs:
            dv = varmap.get(dvx)
            if dv is not None:
                terms_dict[dv] = k
        return lfactory.linear_expr(arg=terms_dict, constant=constant, safe=True)

    @classmethod
    def _make_expr_from_varmap_coefs(cls, lfactory, varmap, var_indices, coefs, offset=0):
        if Environment.env_is_python36:
            terms_dict = {varmap.get(dvx): k for dvx, k in zip(var_indices, coefs)}
        else:
            terms_dict = lfactory._new_term_dict()
            for dvx, k in zip(var_indices, coefs):
                dv = varmap.get(dvx)
                if dv is not None:
                    terms_dict[dv] = k
        return lfactory.linear_expr(arg=terms_dict, constant=offset, safe=True)

    @classmethod
    def read(cls, filename, model_name=None, verbose=False, model_class=None, **kwargs):
        """ Reads a model from a CPLEX export file.

        Accepts all formats exported by CPLEX: LP, SAV, MPS.

        If an error occurs while reading the file, the message of the exception
        is printed and the function returns None.

        Args:
            filename: The file to read.
            model_name: An optional name for the newly created model. If None,
                the model name will be the path basename.
            verbose: An optional flag to print informative messages, default is False.
            model_class: An optional class type; must be a subclass of Model.
                The returned model is built using this model_class and the keyword arguments kwargs, if any.
                By default, the model class is :class:`docplex.mp.model.Model`.
            kwargs: A dict of keyword-based arguments that are used when creating the modelpassed to the
                model constructor.

        Example:
            `m = read_model("c:/temp/foo.mps", model_name="docplex_foo", ignore_names=True)`

            reads a ,odel from file "c:/temp/foo.mps", sets its name to "docplex_foo", and discard all names.

        Returns:
            An instance of Model, or None if an exception is raised.

        See Also:
            :class:`docplex.mp.model.Model`

        """
        if not os.path.exists(filename):
            raise IOError("* file not found: {0}".format(filename))

        # extract basename
        if model_name:
            name_to_use = model_name
        else:
            basename = os.path.basename(filename)
            if '.' not in filename:  # pragma: no cover
                raise RuntimeError('ModelReader.read_model(): path has no extension: {}'.format(filename))
            dotpos = basename.find(".")
            if dotpos > 0:
                name_to_use = basename[:dotpos]
            else:  # pragma: no cover
                name_to_use = basename

        model_class = model_class or Model

        if 0 == os.stat(filename).st_size:
            print("* file is empty: {0} - exiting".format(filename))
            return model_class(name=name_to_use, **kwargs)

        data_check = kwargs.pop('datacheck', None)
        if verbose:
            print("-> CPLEX starts reading file: {0}".format(filename))
        cpx_adapter = cls._read_cplex(filename, datacheck=data_check)
        cpx = cpx_adapter.cpx
        if verbose:
            print("<- CPLEX finished reading file: {0}".format(filename))

        if not cpx:  # pragma: no cover
            return None

        final_output_level = kwargs.get("output_level", "info")
        debug_read = kwargs.get("debug", False)
        final_agent = kwargs.get('agent', 'cplex')
        use_new = kwargs.pop('use_new', True)


        try:
            # force no tck
            if 'checker' in kwargs:
                final_checker = kwargs['checker']
            else:
                final_checker = 'default'
            # build the model with no checker, then restore final_checker in the end.
            kwargs['checker'] = 'off'
            kwargs['agent'] = 'zero'

            ignore_names = kwargs.get('ignore_names', False)
            # -------------
            if Environment.env_is_python36:
                make_terms_fn = cls._make_expr_from_varmap_coefs_dict
                make_terms_seq_fn = cls._make_expr_from_var_coef_seq_dict
            else:
                make_terms_fn = cls._make_expr_from_varmap_coefs_nodict
                make_terms_seq_fn = cls._make_expr_from_var_coef_seq_nodict

            mdl = model_class(name=name_to_use, **kwargs)
            if data_check is not None:
                mdl.parameters.read.datacheck = data_check
            mdl._provenance = filename
            lfactory = mdl._lfactory
            qfactory = mdl._qfactory
            mdl.set_quiet()  # output level set to ERROR
            vartype_cont = mdl.continuous_vartype
            vartype_map = {'B': mdl.binary_vartype,
                           'I': mdl.integer_vartype,
                           'C': mdl.continuous_vartype,
                           'S': mdl.semicontinuous_vartype}

            def cpx_type_to_docplex_type(cpxt):
                return vartype_map.get(cpxt, vartype_cont)

            # 1 upload variables
            cpx_nb_vars = cpx.variables.get_num()

            if verbose:
                print("-- uploading {0} variables...".format(cpx_nb_vars))

            cpx_env_e = cpx._env._e
            cpx_lp = cpx._lp
            has_non_default_lb = cpx_adapter.has_non_default_lb
            has_non_default_ub = cpx_adapter.has_non_default_ub
            has_name = cpx_adapter.has_name

            cpx_var_names = []
            if not ignore_names:
                if not has_name or has_name(cpx_env_e, cpx_lp, 0, cpx_nb_vars - 1):
                    cpx_var_names = cls._safe_call_get_names(cpx_adapter, cpx.variables.get_names)

            # check whether all varianbles have same type, it's worth it.
            unique_type = None
            docplex_vartypes = None
            if cpx._is_MIP():
                cpx_types = cpx.variables.get_types()
                first_cpxtype = cpx_types[0]
                if all(cpxt == first_cpxtype for cpxt in cpx_types):
                    unique_type = cpx_type_to_docplex_type(first_cpxtype)
                else:
                    # docplex_vartypes = [cpx_type_to_docplex_type(cpxt) for cpxt in cpx_types]
                    def kth_vartype(k):
                        return cpx_type_to_docplex_type(cpx_types[k])
                    docplex_vartypes = MockIterable(size=cpx_nb_vars, fn_k=kth_vartype)
            else:
                unique_type = vartype_cont

            if use_new and has_non_default_lb and not has_non_default_lb(cpx_env_e, cpx_lp, 0, cpx_nb_vars - 1):
                cpx_var_lbs = []
            else:
                # either we have no fast predicate or we dont use it.
                cpx_var_lbs = cpx.variables.get_lower_bounds()
                # we have no predicate, so we might try to avoid building a large list??
                if not has_non_default_lb and not any(cpx_var_lbs):  # all equal 0
                    cpx_var_lbs = []

            if use_new and has_non_default_ub and not has_non_default_ub(cpx_env_e, cpx_lp, 0, cpx_nb_vars - 1):
                cpx_var_ubs = []
            else:
                cpx_var_ubs = cpx.variables.get_upper_bounds()

                if not has_non_default_ub and all(cpxu >= 1e+20 for cpxu in cpx_var_ubs):
                    cpx_var_ubs = []

            # if no names, None is fine.
            model_varnames = cpx_var_names or None

            model_lbs = cpx_var_lbs
            model_ubs = cpx_var_ubs

            # vars
            if unique_type:
                model_vars = lfactory.new_var_list(var_container=None,
                                                   key_seq=range(cpx_nb_vars),
                                                   vartype=unique_type,
                                                   lb=model_lbs,
                                                   ub=model_ubs,
                                                   name=model_varnames,
                                                   _safe_bounds=True,
                                                   _safe_names=True
                                                   )
            else:
                assert docplex_vartypes
                model_vars = lfactory.new_multitype_var_list(cpx_nb_vars,
                                                             docplex_vartypes,
                                                             model_lbs,
                                                             model_ubs,
                                                             model_varnames)

            # inverse map from indices to docplex vars
            cpx_var_index_to_docplex = {v: model_vars[v] for v in range(cpx_nb_vars)}

            # 2. upload linear constraints and ranges (mixed in cplex)
            cpx_linearcts = cpx.linear_constraints
            nb_linear_cts = cpx_linearcts.get_num()

            all_rhs = cpx_linearcts.get_rhs()
            all_senses = cpx_adapter.getsense(cpx_env_e, cpx_lp, 0, nb_linear_cts - 1)
            if 'R' in all_senses:
                all_range_values = cpx_linearcts.get_range_values()
            else:
                # do not query ranges if no R in senses...
                all_range_values = []
            cpx_ctnames = [] if ignore_names else cls._safe_call_get_names(cpx_adapter,
                                                                           cpx_linearcts.get_names)

            def make_constant_expr(k):
                return lfactory.constant_expr(k, safe_number=True)

            if verbose:
                print("-- uploading {0} linear constraints...".format(nb_linear_cts))

            def upload_rows(row_iter):
                cts_ = []
                for c, (indices, coefs) in enumerate(row_iter):
                    sense = all_senses[c]
                    rhs = all_rhs[c]

                    ctname = cpx_ctnames[c] if cpx_ctnames else None

                    expr = make_terms_fn(lfactory, cpx_var_index_to_docplex, indices, coefs)

                    if sense == 'R':
                        # no use of querying range vars if no R constraints
                        range_val = all_range_values[c]
                        # rangeval can be negative !!! issue 52
                        if range_val >= 0:
                            range_lb = rhs
                            range_ub = rhs + range_val
                        else:
                            range_ub = rhs
                            range_lb = rhs + range_val

                        rgct = lfactory.new_range_constraint(lb=range_lb, ub=range_ub, expr=expr, name=ctname,
                                                             check_feasible=False)
                        cts_.append(rgct)
                    else:
                        op = cls.parse_sense(sense)
                        rhs_expr = make_constant_expr(rhs)

                        ct = LinearConstraint(mdl, expr, op, rhs_expr, ctname)
                        cts_.append(ct)

                return cts_
            if nb_linear_cts:
                fast_get_rows_tuple = cpx_adapter.fast_get_row_tuples
                if fast_get_rows_tuple:
                    with fast_get_rows_tuple(cpx_env_e, cpx_lp) as rowgen:
                        deferred_cts = upload_rows(rowgen)
                else:
                    all_rows = cpx_adapter.fast_get_rows(cpx)
                    deferred_cts = upload_rows(all_rows)
                # add constraint as a block
                lfactory._post_constraint_block(posted_cts=deferred_cts)

            # 3. upload Quadratic constraints
            cpx_quadraticcts = cpx.quadratic_constraints
            nb_quadratic_cts = cpx_quadraticcts.get_num()
            if nb_quadratic_cts:
                all_rhs = cpx_quadraticcts.get_rhs()
                all_linear_nb_non_zeros = cpx_quadraticcts.get_linear_num_nonzeros()
                all_linear_components = cpx_quadraticcts.get_linear_components()
                all_quadratic_nb_non_zeros = cpx_quadraticcts.get_quad_num_nonzeros()
                all_quadratic_components = cpx_quadraticcts.get_quadratic_components()
                all_senses = cpx_quadraticcts.get_senses()
                cpx_ctnames = [] if ignore_names else cls._safe_call_get_names(cpx_adapter,
                                                                               cpx_quadraticcts.get_names)

                for c in range(nb_quadratic_cts):
                    rhs = all_rhs[c]
                    linear_nb_non_zeros = all_linear_nb_non_zeros[c]
                    linear_component = all_linear_components[c]
                    quadratic_nb_non_zeros = all_quadratic_nb_non_zeros[c]
                    quadratic_component = all_quadratic_components[c]
                    sense = all_senses[c]
                    ctname = cpx_ctnames[c] if cpx_ctnames else None

                    if linear_nb_non_zeros > 0:
                        indices, coefs = linear_component.unpack()
                        # linexpr = mdl._aggregator._scal_prod((cpx_var_index_to_docplex[idx] for idx in indices), coefs)
                        linexpr = make_terms_fn(lfactory, cpx_var_index_to_docplex, indices, coefs)
                    else:
                        linexpr = None

                    if quadratic_nb_non_zeros > 0:
                        qfactory = mdl._qfactory
                        ind1, ind2, coefs = quadratic_component.unpack()
                        quads = qfactory.term_dict_type()
                        for idx1, idx2, coef in zip(ind1, ind2, coefs):
                            quads[VarPair(cpx_var_index_to_docplex[idx1], cpx_var_index_to_docplex[idx2])] = coef

                    else:  # pragma: no cover
                        # should not happen, but who knows
                        quads = None

                    quad_expr = mdl._aggregator._quad_factory.new_quad(quads=quads, linexpr=linexpr, safe=True)
                    op = ComparisonType.cplex_ctsense_to_python_op(sense)
                    ct = op(quad_expr, rhs)
                    mdl.add_constraint(ct, ctname)

            # 4. upload indicators
            cpx_indicators = cpx.indicator_constraints
            nb_indicators = cpx_indicators.get_num()
            if nb_indicators:
                all_ind_names = [] if ignore_names else cls._safe_call_get_names(cpx_adapter,
                                                                                 cpx_indicators.get_names)

                all_ind_bvars = cpx_indicators.get_indicator_variables()
                all_ind_rhs = cpx_indicators.get_rhs()
                all_ind_linearcts = cpx_indicators.get_linear_components()
                all_ind_senses = cpx_indicators.get_senses()
                all_ind_complemented = cpx_indicators.get_complemented()
                all_ind_types = cpx_indicators.get_types()
                ind_equiv_type = 3

                for i in range(nb_indicators):
                    ind_bvar = all_ind_bvars[i]
                    ind_name = all_ind_names[i] if all_ind_names else None
                    ind_rhs = all_ind_rhs[i]
                    ind_linear = all_ind_linearcts[i]  # SparsePair(ind, val)
                    ind_sense = cls.parse_sense(all_ind_senses[i])
                    ind_complemented = all_ind_complemented[i]
                    ind_type = all_ind_types[i]
                    # 1 . check the bvar is ok
                    ind_bvar = cpx_var_index_to_docplex[ind_bvar]
                    # each var appears once
                    ind_linexpr = cls._build_linear_expr_from_sparse_pair(lfactory, cpx_var_index_to_docplex,
                                                                          ind_linear)
                    ind_lct = lfactory._new_binary_constraint(ind_linexpr, ind_sense, ind_rhs)
                    if ind_type == ind_equiv_type:
                        logct = lfactory.new_equivalence_constraint(
                            ind_bvar, ind_lct, true_value=1 - ind_complemented, name=ind_name)
                    else:
                        logct = lfactory.new_indicator_constraint(
                            ind_bvar, ind_lct, true_value=1 - ind_complemented, name=ind_name)
                    mdl.add(logct)

            # 5. upload Piecewise linear constraints
            try:
                cpx_pwl = cpx.pwl_constraints
                cpx_pwl_defs = cpx_pwl.get_definitions()
                pwl_fallback_names = [""] * cpx_pwl.get_num()
                cpx_pwl_names = pwl_fallback_names if ignore_names else cls._safe_call_get_names(cpx_adapter,
                                                                                                 cpx_pwl.get_names,
                                                                                                 pwl_fallback_names)
                for (vary_idx, varx_idx, preslope, postslope, breakx, breaky), pwl_name in zip(cpx_pwl_defs,
                                                                                                cpx_pwl_names):
                    varx = cpx_var_index_to_docplex.get(varx_idx, None)
                    vary = cpx_var_index_to_docplex.get(vary_idx, None)
                    breakxy = [(brkx, brky) for brkx, brky in zip(breakx, breaky)]
                    pwl_func = mdl.piecewise(preslope, breakxy, postslope, name=pwl_name)
                    pwl_expr = mdl._lfactory.new_pwl_expr(pwl_func, varx, 0, resolve=False)
                    pwl_expr._f_var = vary
                    pwl_expr._ensure_resolved()

            except AttributeError:  # pragma: no cover
                pass  # Do not check for PWLs if Cplex version does not support them

            # 6. upload objective

            # noinspection PyPep8
            try:
                cpx_multiobj = cpx.multiobj
            except AttributeError:  # pragma: no cover
                # pre-12.9 version
                cpx_multiobj = None

            if cpx_multiobj is None or cpx_multiobj.get_num() <= 1:
                cpx_obj = cpx.objective
                cpx_sense = cpx_obj.get_sense()
                fast_getobj = cpx_adapter.fast_getobj
                if fast_getobj:
                    with fast_getobj(cpx_env_e, cpx_lp, 0, cpx_nb_vars-1) as var_coefs_tuple:
                        obj_expr = make_terms_seq_fn(lfactory, cpx_var_index_to_docplex, var_coefs_tuple)
                else:
                    cpx_all_lin_obj_coeffs = cpx_obj.get_linear()
                    obj_expr = compute_full_dot(mdl, cpx_all_lin_obj_coeffs)

                if cpx_obj.get_num_quadratic_variables() > 0:
                    cpx_all_quad_cols_coeffs = cpx_obj.get_quadratic()
                    quads = qfactory.term_dict_type()
                    for v, col_coefs in zip(cpx_var_index_to_docplex, cpx_all_quad_cols_coeffs):
                        var1 = cpx_var_index_to_docplex[v]
                        indices, coefs = col_coefs.unpack()
                        for idx, coef in zip(indices, coefs):
                            vp = VarPair(var1, cpx_var_index_to_docplex[idx])
                            quads[vp] = quads.get(vp, 0) + coef / 2

                    obj_expr += qfactory.new_quad(quads=quads, linexpr=None)

                obj_expr += cpx.objective.get_offset()
                is_maximize = cpx_sense == cpx_adapter.cplex_module._internal._subinterfaces.ObjSense.maximize

                if is_maximize:
                    mdl.maximize(obj_expr)
                else:
                    mdl.minimize(obj_expr)
            else:
                # we have multiple objective

                nb_multiobjs = cpx_multiobj.get_num()
                exprs = [0] * nb_multiobjs
                priorities = [1] * nb_multiobjs
                weights = [1] * nb_multiobjs
                abstols = [0] * nb_multiobjs
                reltols = [0] * nb_multiobjs
                if ignore_names:
                    names = ["Goal_{0}".format(g) for g in range(1, nb_multiobjs + 1)]
                else:
                    names = cpx_multiobj.get_names()

                fast_multiobj = cpx_adapter.fast_multiobj_getobj is not None
                if fast_multiobj:
                    def get_kth_multiobj(k):
                        weight = cpx_adapter.fast_multiobj_getweight(cpx_env_e, cpx_lp, k)
                        offset = cpx_adapter.fast_multiobj_getoffset(cpx_env_e, cpx_lp, k)
                        priority = cpx_adapter.fast_multiobj_getprio(cpx_env_e, cpx_lp, k)
                        abstol = cpx_adapter.fast_multiobj_getabstol(cpx_env_e, cpx_lp, k)
                        reltol = cpx_adapter.fast_multiobj_getreltol(cpx_env_e, cpx_lp, k)
                        with cpx_adapter.fast_multiobj_getobj(cpx_env_e, cpx_lp, k, 0, cpx_nb_vars-1) as objk_tuples:
                            obj_expr = make_terms_seq_fn(lfactory, cpx_var_index_to_docplex, objk_tuples, constant=offset)
                        return obj_expr, weight, priority, abstol, reltol
                else:
                    def get_kth_multiobj(k):
                        (obj_coeffs, obj_offset, weight, prio, abstol, reltol) = cpx_multiobj.get_definition(k)
                        obj_expr = compute_full_dot(mdl, obj_coeffs, obj_offset)
                        return obj_expr, weight, prio, abstol, reltol

                for m in range(nb_multiobjs):
                    (obj_expr, weight, prio, abstol, reltol) = get_kth_multiobj(m)
                    exprs[m] = obj_expr
                    priorities[m] = prio
                    weights[m] = weight
                    abstols[m] = abstol
                    reltols[m] = reltol
                sense = cpx_multiobj.get_sense()
                mdl.set_multi_objective(sense, exprs, priorities, weights, abstols, reltols, names)

            # upload sos
            cpx_sos = cpx.SOS
            cpx_sos_num = cpx_sos.get_num()
            if cpx_sos_num > 0:
                cpx_sos_types = cpx_sos.get_types()
                cpx_sos_indices = cpx_sos.get_sets()
                cpx_sos_names = cpx_sos.get_names()
                if not cpx_sos_names:
                    cpx_sos_names = [None] * cpx_sos_num
                for sostype, sos_sparse, sos_name in zip(cpx_sos_types, cpx_sos_indices, cpx_sos_names):
                    sos_var_indices = sos_sparse.ind
                    sos_weights = sos_sparse.val
                    isostype = int(sostype)
                    sos_vars = [cpx_var_index_to_docplex[var_ix] for var_ix in sos_var_indices]
                    mdl.add_sos(dvars=sos_vars, sos_arg=isostype, name=sos_name, weights=sos_weights)

            # upload lazy constraints
            cpx_linear_advanced = cpx.linear_constraints.advanced
            cpx_lazyct_num = cpx_linear_advanced.get_num_lazy_constraints()
            if cpx_lazyct_num:
                print("WARNING: found {0} lazy constraints that cannot be uploaded to DOcplex".format(cpx_lazyct_num))

            mdl.output_level = final_output_level
            if final_checker:
                # need to restore checker
                mdl.set_checker(final_checker)

        except cpx_adapter.CplexError as cpx_e:  # pragma: no cover
            print("* CPLEX error: {0!s} reading file {1}".format(cpx_e, filename))
            mdl = None
            if debug_read:
                raise

        except ModelReaderError as mre:  # pragma: no cover
            print("! Model reader error: {0!s} while reading file {1}".format(mre, filename))
            mdl = None
            if debug_read:
                raise

        except DOcplexException as doe:  # pragma: no cover
            print("! Internal DOcplex error: {0!s} while reading file {1}".format(doe, filename))
            mdl = None
            if debug_read:
                raise

        # except Exception as any_e:  # pragma: no cover
        #     print("Internal exception raised: {0} msg={1!s} while reading file '{2}'".format(type(any_e), any_e, filename))
        #     mdl = None
        #     if debug_read:
        #         raise

        finally:
            # clean up CPLEX instance...
            # cpx.end()
            pass
        if mdl:
            # reset engine, if necessary
            if final_agent == 'zero':
                pass  # nothing to do
            if final_agent == 'cplex':
                mdl._set_engine(CplexEngine(mdl, _cplex=cpx_adapter))
                # mdl._check_scope_indices()
            else:
                # set a new non-cplex agent
                mdl._set_new_engine_from_agent(final_agent)
            mdl._set_solver_agent(final_agent)

        return mdl

    @classmethod
    def read_model(cls, filename, model_name=None, verbose=False, model_class=None, **kwargs):
        """ This method is a synonym of `read` for compatibility.

        """
        import warnings
        warnings.warn("ModelReader.read_model is deprecated, use class method ModelReader.read()", DeprecationWarning)
        return cls.read(filename, model_name, verbose, model_class, **kwargs)


def read_model(filename, model_name=None, verbose=False, **kwargs):
    """ Reads a model from a CPLEX export file.

    Accepts all formats exported by CPLEX: LP, SAV, SAV.gz, MPS.

    If an error occurs while reading the file, the message of the exception
    is printed and the function returns None.

    Args:
        filename: The model file to read.
        model_name: An optional name for the newly created model. If None,
            the model name will be the path basename.
        verbose: An optional flag to print informative messages, default is False.
        kwargs: A dict of keyword-based arguments that are passed to the model contructor.

    Note:
        This function requires CPLEX runtime, otherwise an exceotion is raised.

    Example:
        `m = read_model("c:/temp/foo.mps", model_name="docplex_foo", solver_agent="local", output_level=100)`

    Returns:
        An instance of Model, or None if an exception is raised.



    See Also:
        :class:`docplex.mp.model.Model`

    """
    return ModelReader.read(filename, model_name=model_name, verbose=verbose, **kwargs)
