# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015 - 2022
# --------------------------------------------------------------------------

"""
This module contains the Python descriptors of the different CPO types and operations.
"""

#-----------------------------------------------------------------------------
# Public classes
#-----------------------------------------------------------------------------

class CpoType(object):
    """ CPO type descriptor """
    __slots__ = ('name',               # Name of the type
                 'public_name',        # Public name of the type
                 'is_variable',        # Indicate a type corresponding to a variable
                 'is_constant',        # Indicates that type denotes a constant (possibly array)
                 'is_const_or_var',    # Indicates that type denotes a constant (possibly array) or a variable
                 'is_constant_atom',   # Indicates that type denotes an atomic constant
                 'is_array',           # Indicates that type describes an array
                 'is_array_of_expr',   # Indicates that type describes an array of expressions (not constants)
                 'is_toplevel',        # Indicates that type describes an expression that can be used as top level expression
                 'higher_types',       # List of higher types in the hierarchy
                 'element_type',       # Type of array element (for arrays)
                 'parent_array_type',  # Type corresponding to an array of this type
                 'base_type',          # Base type to be used for signature matching
                 'id',                 # Unique type id (index) used to fasten type links
                 'kind_of_types',      # Set of types that are kind of this one
                 'common_types'        # Dictionary of types that are common with this and others.yield
                                       # Key is type name, value is common type with this one.
                )

    def __init__(self, name, isvar=False, iscst=False, istop=False, isatm=False, htyps=(), eltyp=None, bastyp=None, public=None):
        """ **Constructor**

        Args:
            name:   Name of the type
            isvar:  Indicates whether this type denotes a variable
            iscst:  Indicate whether this type denotes a constant
            istop:  Indicate whether this type is a top-level expression
            isatm:  Indicate whether this type denotes an atomic constant
            htyps:  List of types higher in the hierarchy
            eltyp:  Array element type, None (default) if not array
            bastyp: Base type to be used for signature matching
            public: Public name of the type
        """
        super(CpoType, self).__init__()
        self.name              = name
        self.is_variable       = isvar
        self.is_constant       = iscst
        self.is_const_or_var   = isvar or iscst
        self.is_constant_atom  = isatm
        self.is_toplevel       = istop
        self.higher_types = (self,) + htyps
        self.element_type = eltyp
        self.public_name = public
        if bastyp is None:
            self.base_type = self
        else:
            self.base_type = bastyp
            self.is_variable = bastyp.is_variable
            self.is_constant = bastyp.is_constant
            self.is_constant_atom = bastyp.is_constant_atom
        # Process array case
        if eltyp is not None:
            eltyp.parent_array_type = self
            self.is_constant  = eltyp.is_constant
        self.is_array = eltyp is not None
        self.is_array_of_expr = self.is_array and not self.is_constant
        self.parent_array_type = None

    def get_name(self):
        """ Get the name of the type

        Returns:
            Name of the type
        """
        return self.name

    def get_public_name(self):
        """ Get the public name of the type

        Returns:
            Public name of the type
        """
        return self.name if self.public_name is None else self.public_name

    def is_kind_of(self, tp):
        """ Check if this type is a kind of another type, i.e. other type is in is hierarchy

        Args:
            tp: Other type to check
        Returns:
            True if this type is a kind of tp
        """
        return self.kind_of_types[tp.id]

    def get_common_type(self, tp):
        """ Get first common type between this and the parameter.

        Args:
            tp: Other type
        Returns:
            The first common type between this and the parameter, None if none
        """
        return self.common_types[tp.id]

    def _compute_common_type(self, tp):
        """ Compute the first common type between this and the parameter.

        Args:
            tp: Other type
        Returns:
            The first common type between this and the parameter, None if none
        """
        # Check if given type is derived
        if tp.base_type is not tp:
            tp = tp.base_type
        # Check if this type is derived
        if self.base_type is not self:
            return self.base_type._compute_common_type(tp)
        # Check direct comparison
        if (self is tp) or tp.is_kind_of(self):
            return self
        elif self.is_kind_of(tp):
            return tp
        # Search common types in ancestors
        for ct in self.higher_types:
            if ct in tp.higher_types:
                return ct
        return None

    def _force_common_type(self, tp, ct):
        """ Force a common type in an already list of common types.

        Args:
            tp: Other type
            ct: Common type
        """
        ctypes = list(self.common_types)
        ctypes[tp.id] = ct
        self.common_types = tuple(ctypes)

    def __str__(self):
        """ Convert this object into a string """
        return self.name

    def __eq__(self, other):
        """ Check equality of this object with another """
        return (self is other) or \
               (isinstance(other, self.__class__) and (self.name == other.name))

    def __ne__(self, other):
        """ Check inequality of this object with another """
        return not self.__eq__(other)

    def __hash__(self):
        """ Return object hash-code """
        return id(self)


class CpoParam(object):
    """ Descriptor of an operation parameter """
    __slots__ = ('type',          # Parameter CPO type
                 'default_value'  # Parameter default value, None if none
                )

    def __init__(self, ptyp, dval=None):
        """ **Constructor**

        Args:
            ptyp: Parameter type
            dval: Default value
        """
        super(CpoParam, self).__init__()
        self.type = ptyp
        self.default_value = dval

    def __str__(self):
        if self.default_value is None:
            return self.type.name
        else:
            return self.type.name + "=" + str(self.default_value)

    def __eq__(self, other):
        """ Check equality of this object with another """
        return (self is other) or \
               (isinstance(other, self.__class__) and (self.type == other.type) and (self.default_value == other.defval))

    def __ne__(self, other):
        """ Check inequality of this object with another """
        return not self.__eq__(other)


# Marker to signal any type and number of arguments
TYPE_ANY = CpoType("Any")
ANY_ARGUMENTS = (CpoParam(TYPE_ANY),)

class CpoSignature(object):
    """ Descriptor of a CPO operation signature """
    __slots__ = ('return_type',  # Return type
                 'parameters',   # List of parameter descriptors
                 'operation'     # Parent operation
                 )

    def __init__(self, rtyp, ptyps):
        """ **Constructor**

        Args:
            rtyp:  Returned type
            ptyps: Array of parameter types
        """
        super(CpoSignature, self).__init__()
        self.return_type = rtyp

        # Build list of parameters
        if ptyps is ANY_ARGUMENTS:
            self.parameters = ANY_ARGUMENTS
        else:
            lpt = []
            for pt in ptyps:
                if isinstance(pt, CpoParam):
                    lpt.append(pt)
                else:
                    lpt.append(CpoParam(pt))
            self.parameters = tuple(lpt)

    def __str__(self):
        return str(self.return_type) + "[" + ", ".join(map(str, self.parameters)) + "]"

    def __eq__(self, other):
        """ Check equality of this object with another """
        if self is other:
            return True
        return isinstance(other, self.__class__) and (self.return_type == other.rtype) \
               and (self.operation == other.operation) and (self.parameters == other.params)

    def __ne__(self, other):
        """ Check inequality of this object with another """
        return not self.__eq__(other)


class CpoOperation(object):
    """ CPO operation descriptor """
    __slots__ = ('cpo_name',     # Operation CPO name
                 'python_name',  # Operation python name
                 'keyword',      # Operation keyword (operation symbol)
                 'priority',     # Operator priority, -1 for function call
                 'signatures'    # List of possible operation signatures
                 )

    def __init__(self, cpname, pyname, kwrd, prio, signs):
        """ **Constructor**

        Args:
            cpname:  Operation CPO name
            pyname:  Operation python name
            kwrd:    Keyword, None for same as cpo name
            prio:    Priority
            signs:   Array of possible signatures
        """
        super(CpoOperation, self).__init__()

        # Store attributes
        self.cpo_name = cpname
        self.python_name = pyname
        self.priority = prio
        if kwrd:
            self.keyword = kwrd
        else:
            self.keyword = cpname
        self.signatures = signs

        # Set pointer back on operation on each signature
        for s in signs:
            s.operation = self

    def get_cpo_name(self):
        """ Get the CPO name of the operation

        Returns:
            CPO Name of the operation
        """
        return self.cpo_name

    def __str__(self):
        return str(self.cpo_name) + "(" + ", ".join(map(str, self.signatures)) + ")"

    def __eq__(self, other):
        """ Check equality of this object with another """
        return (self is other) or \
               (isinstance(other, self.__class__) and (self.cpo_name == other.cpo_name))

    def __ne__(self, other):
        """ Check inequality of this object with another """
        return not self.__eq__(other)


#-----------------------------------------------------------------------------
# Descriptors of CPO types
#-----------------------------------------------------------------------------

Type_FloatExpr             = CpoType("FloatExpr", public="float expression")
Type_FloatVar              = CpoType("FloatVar", public="float variable", isvar=True, htyps=(Type_FloatExpr,))
Type_Float                 = CpoType("Float", public="float constant", iscst=True, isatm=True, htyps=(Type_FloatExpr,))
Type_FloatExprArray        = CpoType("FloatExprArray", public="array of float expressions", eltyp=Type_FloatExpr)
Type_FloatArray            = CpoType("FloatArray", public="array of float constants", iscst=True, htyps=(Type_FloatExprArray,), eltyp=Type_Float)
Type_FloatVarArray         = CpoType("FloatVarArray", public="array of float variables", htyps=(Type_FloatExprArray,), eltyp=Type_FloatVar)

Type_IntExpr               = CpoType("IntExpr", public="integer expression", htyps=(Type_FloatExpr,))
Type_IntVar                = CpoType("IntVar", public="integer variable", isvar=True, htyps=(Type_IntExpr, Type_FloatExpr,))
Type_Int                   = CpoType("Int", public="integer constant", iscst=True, isatm=True, htyps=(Type_IntExpr, Type_Float, Type_FloatExpr,))
Type_TimeInt               = CpoType("TimeInt", public="integer representing a time", iscst=True, htyps=(Type_Int, Type_IntExpr, Type_Float, Type_FloatExpr,), bastyp=Type_Int)
Type_PositiveInt           = CpoType("PositiveInt", public="positive integer constant", iscst=True, htyps=(Type_Int, Type_IntExpr, Type_Float, Type_FloatExpr,), bastyp=Type_Int)
Type_IntExprArray          = CpoType("IntExprArray", public="array of integer expressions", htyps=(Type_FloatExprArray,), eltyp=Type_IntExpr)
Type_IntArray              = CpoType("IntArray", public="array of integer constants", iscst=True, htyps=(Type_IntExprArray, Type_FloatArray, Type_FloatExprArray,), eltyp=Type_Int)
Type_IntVarArray           = CpoType("IntVarArray", public="array of integer variables", htyps=(Type_IntExprArray, Type_FloatExprArray,), eltyp=Type_IntVar)

Type_Constraint            = CpoType("Constraint", public="constraint", istop=True)

Type_BoolExpr              = CpoType("BoolExpr", public="boolean expression", istop=True, htyps=(Type_IntExpr, Type_FloatExpr, Type_Constraint,))
Type_Bool                  = CpoType("Bool", public="boolean constant", istop=True, iscst=True, isatm=True, htyps=(Type_Int, Type_Float, Type_BoolExpr, Type_IntExpr, Type_FloatExpr, Type_Constraint,))
Type_BoolVar               = CpoType("BoolVar", public="boolean variable", isvar=True, htyps=(Type_IntVar, Type_Bool, Type_Int, Type_Float, Type_BoolExpr, Type_IntExpr, Type_FloatExpr, Type_Constraint,))
Type_BoolInt               = CpoType("BoolInt", public="boolean integer", iscst=True, htyps=(Type_Int, Type_IntExpr, Type_Float, Type_FloatExpr,), bastyp=Type_Int)
Type_BoolExprArray         = CpoType("BoolExprArray", public="array of boolean expressions", htyps=(Type_IntExprArray, Type_FloatExprArray,), eltyp=Type_BoolExpr)
Type_BoolArray             = CpoType("BoolArray", public="array of boolean constants", iscst=True, htyps=(Type_IntArray, Type_FloatArray, Type_BoolExprArray, Type_IntExprArray, Type_FloatExprArray,), eltyp=Type_Bool)
Type_BoolVarArray          = CpoType("BoolVarArray", public="array of boolean variables", htyps=(Type_BoolExprArray, Type_IntExprArray, Type_FloatExprArray,), eltyp=Type_BoolVar)

Type_IntervalVar           = CpoType("IntervalVar", public="interval variable", isvar=True)
Type_IntervalVarArray      = CpoType("IntervalVarArray", public="array of interval variables", eltyp=Type_IntervalVar)

Type_SequenceVar           = CpoType("SequenceVar", public="sequence variable", isvar=True)
Type_SequenceVarArray      = CpoType("SequenceVarArray", public="array of sequence variables", eltyp=Type_SequenceVar)

Type_CumulExpr             = CpoType("CumulExpr", public="cumul expression")
Type_CumulExprArray        = CpoType("CumulExprArray", public="array of cumul expressions", eltyp=Type_CumulExpr)
Type_CumulAtom             = CpoType("CumulAtom", public="cumul atom", htyps=(Type_CumulExpr,))
Type_CumulAtomArray        = CpoType("CumulAtomArray", public="array of cumul atoms", htyps=(Type_CumulExprArray,), eltyp=Type_CumulAtom)
Type_CumulFunction         = CpoType("CumulFunction", public="cumul function", htyps=(Type_CumulExpr,))

Type_StateFunction         = CpoType("StateFunction", public="state function", isvar=True)
Type_SegmentedFunction     = CpoType("SegmentedFunction", public="segmented function", iscst=True)
Type_StepFunction          = CpoType("StepFunction", public="step function", iscst=True)
Type_TransitionMatrix      = CpoType("TransitionMatrix", public="transition matrix", iscst=True)
Type_IntervalArray         = CpoType("IntervalArray", public="array of intervals", iscst=True)
Type_Objective             = CpoType("Objective", public="objective function", istop=True)
Type_TupleSet              = CpoType("TupleSet", public="tuple set", iscst=True)

Type_IntValueEval          = CpoType("IntValueEval", public="evaluator of integer value")
Type_IntValueChooser       = CpoType("IntValueChooser", public="chooser of integer value")
Type_IntValueSelector      = CpoType("IntValueSelector", public="selector of integer value", htyps=(Type_IntValueChooser,))
Type_IntValueSelectorArray = CpoType("IntValueSelectorArray", public="array of integer value selectors", htyps=(Type_IntValueChooser,), eltyp=Type_IntValueSelector)
Type_IntVarEval            = CpoType("IntVarEval", public="evaluator of integer variable")
Type_IntVarChooser         = CpoType("IntVarChooser", public="chooser of integer variable")
Type_IntVarSelector        = CpoType("IntVarSelector", public="selector of integer variable", htyps=(Type_IntVarChooser,))
Type_IntVarSelectorArray   = CpoType("IntVarSelectorArray", public="array of integer variable selectors", htyps=(Type_IntVarChooser,), eltyp=Type_IntVarSelector)
Type_SearchPhase           = CpoType("SearchPhase", public="search phase")
Type_Blackbox              = CpoType("Blackbox", public="blackbox function")

# Special types
Type_Any                   = TYPE_ANY             # Fake type
Type_Unknown               = CpoType("Unknown")   # Fake type
Type_Python                = CpoType("Python")
#Type_Identifier            = CpoType("Identifier", isvar=True)
Type_IntInterval           = CpoType("IntInterval", public="interval of integer constants", iscst=True, htyps=(Type_Int,))


#-----------------------------------------------------------------------------
# Descriptors of private CPO operations
#-----------------------------------------------------------------------------

Oper__added                      = CpoOperation("_added", "_added", None, -1, ( CpoSignature(Type_Constraint, (Type_IntExpr, )),
                                                                                CpoSignature(Type_Constraint, (Type_BoolExpr, )),) )
Oper_custom_constraint           = CpoOperation("customConstraint", "custom_constraint", None, -1, ( CpoSignature(Type_BoolExpr, ANY_ARGUMENTS),) )
Oper__alternative_expr           = CpoOperation("_alternativeExpr", "_alternative_expr", None, -1, ( CpoSignature(Type_IntExpr, (Type_IntervalVar, Type_IntervalVarArray, Type_IntArray)),
                                                                                                     CpoSignature(Type_FloatExpr, (Type_IntervalVar, Type_IntervalVarArray, Type_FloatArray))) )
Oper__asym_distance              = CpoOperation("_asymDistance", "_asym_distance", None, -1, ( CpoSignature(Type_IntExpr, (Type_IntExpr, Type_IntExpr)),) )
Oper__binary_s_p_equal           = CpoOperation("_binarySPEqual", "_binary_s_p_equal", None, -1, ( CpoSignature(Type_Constraint, (Type_IntArray, Type_IntExprArray, Type_Int)),) )
Oper__bound_pruning              = CpoOperation("_boundPruning", "_bound_pruning", None, -1, ( CpoSignature(Type_IntVarEval, ()),) )
Oper__ceil                       = CpoOperation("_ceil", "_ceil", None, -1, ( CpoSignature(Type_IntExpr, (Type_FloatExpr,)),) )
Oper__choice                     = CpoOperation("_choice", "_choice", None, -1, ( CpoSignature(Type_Constraint, (Type_IntervalVar, Type_IntExpr, Type_IntervalVarArray, CpoParam(Type_Int, dval=-1))),) )
Oper__conditional                = CpoOperation("_conditional", "_conditional", None, -1, ( CpoSignature(Type_IntExpr, (Type_BoolExpr, Type_IntExpr, Type_IntExpr)),
                                                                                            CpoSignature(Type_FloatExpr, (Type_BoolExpr, Type_FloatExpr, Type_FloatExpr))) )
Oper__cumul_atom_array           = CpoOperation("_cumulAtomArray", "_cumul_atom_array", None, -1, ( CpoSignature(Type_CumulAtomArray, ()),) )
Oper__cumul_function             = CpoOperation("_cumulFunction", "_cumul_function", None, -1, ( CpoSignature(Type_CumulFunction, (Type_CumulExprArray, Type_CumulExprArray)),) )
Oper__degree                     = CpoOperation("_degree", "_degree", None, -1, ( CpoSignature(Type_IntVarEval, ()),) )
Oper__dichotomy                  = CpoOperation("_dichotomy", "_dichotomy", None, -1, ( CpoSignature(Type_IntVarEval, ()),) )
Oper__dist_to_int                = CpoOperation("_distToInt", "_dist_to_int", None, -1, ( CpoSignature(Type_FloatExpr, (Type_FloatExpr,)),) )
Oper__distance                   = CpoOperation("_distance", "_distance", None, -1, ( CpoSignature(Type_IntExpr, (Type_IntExpr, Type_IntExpr)),) )
Oper__end_modulo                 = CpoOperation("_endModulo", "_end_modulo", None, -1, ( CpoSignature(Type_IntExpr, (Type_IntervalVar, Type_Int, CpoParam(Type_Int, dval=0))),) )
Oper__eq_asym_distance           = CpoOperation("_eqAsymDistance", "_eq_asym_distance", None, -1, ( CpoSignature(Type_Constraint, (Type_IntExpr, Type_IntExpr, Type_IntExpr)),) )
Oper__equivalence                = CpoOperation("_equivalence", "_equivalence", None, -1, ( CpoSignature(Type_Constraint, (Type_IntervalVar, Type_IntervalVar)),) )
Oper__explicit_var_eval          = CpoOperation("_explicitVarEval", "_explicit_var_eval", None, -1, ( CpoSignature(Type_IntVarEval, (Type_FloatArray, CpoParam(Type_Float, dval=0))),) )
Oper__float_to_int               = CpoOperation("_floatToInt", "_float_to_int", None, -1, ( CpoSignature(Type_IntExpr, (Type_FloatExpr,)),) )
Oper__floor                      = CpoOperation("_floor", "_floor", None, -1, ( CpoSignature(Type_IntExpr, (Type_FloatExpr,)),) )
Oper__forbid_end                 = CpoOperation("_forbidEnd", "_forbid_end", None, -1, ( CpoSignature(Type_Constraint, (Type_IntervalVar, Type_IntervalArray)),) )
Oper__forbid_extent              = CpoOperation("_forbidExtent", "_forbid_extent", None, -1, ( CpoSignature(Type_Constraint, (Type_IntervalVar, Type_IntervalArray)),) )
Oper__forbid_start               = CpoOperation("_forbidStart", "_forbid_start", None, -1, ( CpoSignature(Type_Constraint, (Type_IntervalVar, Type_IntervalArray)),) )
Oper__fract                      = CpoOperation("_fract", "_fract", None, -1, ( CpoSignature(Type_FloatExpr, (Type_FloatExpr,)),) )
Oper__implies                    = CpoOperation("_implies", "_implies", None, -1, ( CpoSignature(Type_Constraint, (Type_IntervalVar, Type_IntervalVar)),) )
Oper__implies_not                = CpoOperation("_impliesNot", "_implies_not", None, -1, ( CpoSignature(Type_Constraint, (Type_IntervalVar, Type_IntervalVar)),) )
Oper__length_modulo              = CpoOperation("_lengthModulo", "_length_modulo", None, -1, ( CpoSignature(Type_IntExpr, (Type_IntervalVar, Type_Int, CpoParam(Type_Int, dval=0))),) )
Oper__max                        = CpoOperation("_max", "_max", None, -1, ( CpoSignature(Type_IntExpr, (Type_CumulExpr, Type_TimeInt, Type_TimeInt)),) )
Oper__max_distance               = CpoOperation("_maxDistance", "_max_distance", None, -1, ( CpoSignature(Type_BoolExpr, (Type_IntExpr, Type_IntExpr, Type_Int)),) )
Oper__max_size_var_evaluator     = CpoOperation("_maxSizeVarEvaluator", "_max_size_var_evaluator", None, -1, ( CpoSignature(Type_IntVarEval, ()),) )
Oper__maximize                   = CpoOperation("_maximize", "_maximize", None, -1, ( CpoSignature(Type_Objective, (Type_FloatExprArray,)),) )
Oper__maximize_dynamic_lex       = CpoOperation("_maximizeDynamicLex", "_maximize_dynamic_lex", None, -1, ( CpoSignature(Type_Objective, (Type_FloatExprArray,)),) )
Oper__min                        = CpoOperation("_min", "_min", None, -1, ( CpoSignature(Type_IntExpr, (Type_CumulExpr, Type_TimeInt, Type_TimeInt)),) )
Oper__min_distance               = CpoOperation("_minDistance", "_min_distance", None, -1, ( CpoSignature(Type_BoolExpr, (Type_IntExpr, Type_IntExpr, Type_Int)),) )
Oper__minimize                   = CpoOperation("_minimize", "_minimize", None, -1, ( CpoSignature(Type_Objective, (Type_FloatExprArray,)),) )
Oper__minimize_dynamic_lex       = CpoOperation("_minimizeDynamicLex", "_minimize_dynamic_lex", None, -1, ( CpoSignature(Type_Objective, (Type_FloatExprArray,)),) )
Oper__multi_span                 = CpoOperation("_multiSpan", "_multi_span", None, -1, ( CpoSignature(Type_Constraint, (Type_IntervalVarArray, Type_IntervalVarArray, CpoParam(Type_IntExprArray, dval=0), CpoParam(Type_Int, dval=0))),) )
Oper__no_overlap                 = CpoOperation("_noOverlap", "_no_overlap", None, -1, ( CpoSignature(Type_Constraint, (Type_SequenceVar, Type_TransitionMatrix, Type_TransitionMatrix)),) )
Oper__not_range                  = CpoOperation("_notRange", "_not_range", None, -1, ( CpoSignature(Type_BoolExpr, (Type_IntExpr, Type_Float, Type_Float)),) )
Oper__opposite                   = CpoOperation("_opposite", "_opposite", None, -1, ( CpoSignature(Type_Constraint, (Type_IntervalVar, Type_IntervalVar)),) )
Oper__or                         = CpoOperation("_or", "_or", None, -1, ( CpoSignature(Type_Constraint, (Type_IntervalVar, Type_IntervalVar)),) )
Oper__piecewise_linear           = CpoOperation("_piecewiseLinear", "_piecewise_linear", None, -1, ( CpoSignature(Type_FloatExpr, (Type_FloatExpr, Type_SegmentedFunction)),) )
Oper__random_value               = CpoOperation("_randomValue", "_random_value", None, -1, ( CpoSignature(Type_IntValueEval, ()),) )
Oper__random_var                 = CpoOperation("_randomVar", "_random_var", None, -1, ( CpoSignature(Type_IntVarEval, ()),) )
Oper__regret_on_max              = CpoOperation("_regretOnMax", "_regret_on_max", None, -1, ( CpoSignature(Type_IntVarEval, ()),) )
Oper__regret_on_min              = CpoOperation("_regretOnMin", "_regret_on_min", None, -1, ( CpoSignature(Type_IntVarEval, ()),) )
Oper__round                      = CpoOperation("_round", "_round", None, -1, ( CpoSignature(Type_IntExpr, (Type_FloatExpr,)),) )
Oper__same_cumul_atom            = CpoOperation("_sameCumulAtom", "_same_cumul_atom", None, -1, ( CpoSignature(Type_Constraint, (Type_CumulAtom, Type_CumulAtom)),) )
Oper__same_interval_domain       = CpoOperation("_sameIntervalDomain", "_same_interval_domain", None, -1, ( CpoSignature(Type_Constraint, (Type_IntervalVar, Type_IntervalVar)),) )
Oper__sgn                        = CpoOperation("_sgn", "_sgn", None, -1, ( CpoSignature(Type_IntExpr, (Type_FloatExpr,)),) )
Oper__size_modulo                = CpoOperation("_sizeModulo", "_size_modulo", None, -1, ( CpoSignature(Type_IntExpr, (Type_IntervalVar, Type_Int, CpoParam(Type_Int, dval=0))),) )
Oper__size_over_degree           = CpoOperation("_sizeOverDegree", "_size_over_degree", None, -1, ( CpoSignature(Type_IntVarEval, ()),) )
Oper__start_modulo               = CpoOperation("_startModulo", "_start_modulo", None, -1, ( CpoSignature(Type_IntExpr, (Type_IntervalVar, Type_Int, CpoParam(Type_Int, dval=0))),) )
Oper__state_condition            = CpoOperation("_stateCondition", "_state_condition", None, -1, ( CpoSignature(Type_Constraint, (Type_StateFunction, Type_IntervalVar, Type_BoolInt, Type_BoolInt, Type_PositiveInt, Type_PositiveInt, CpoParam(Type_BoolInt, dval=0), CpoParam(Type_BoolInt, dval=0))),
                                                                                                   CpoSignature(Type_Constraint, (Type_StateFunction, Type_TimeInt, Type_TimeInt, Type_BoolInt, Type_BoolInt, Type_PositiveInt, Type_PositiveInt, CpoParam(Type_BoolInt, dval=0), CpoParam(Type_BoolInt, dval=0)))) )
Oper__table_element               = CpoOperation("_tableElement", "_table_element", None, -1, ( CpoSignature(Type_Constraint, (Type_IntExpr, Type_IntArray, Type_IntExpr)),) )
Oper__trunc                       = CpoOperation("_trunc", "_trunc", None, -1, ( CpoSignature(Type_IntExpr, (Type_FloatExpr,)),) )
Oper__value_local_impact          = CpoOperation("_valueLocalImpact", "_value_local_impact", None, -1, ( CpoSignature(Type_IntValueEval, ()),) )
Oper__value_lower_obj_variation   = CpoOperation("_valueLowerObjVariation", "_value_lower_obj_variation", None, -1, ( CpoSignature(Type_IntValueEval, ()),) )
Oper__value_pruning               = CpoOperation("_valuePruning", "_value_pruning", None, -1, ( CpoSignature(Type_IntVarEval, ()),) )
Oper__value_upper_obj_variation   = CpoOperation("_valueUpperObjVariation", "_value_upper_obj_variation", None, -1, ( CpoSignature(Type_IntValueEval, ()),) )
Oper__var_index                   = CpoOperation("_varIndex", "_var_index", None, -1, ( CpoSignature(Type_IntVarEval, (CpoParam(Type_Float, dval=-1),)),) )
Oper__var_lower_obj_variation     = CpoOperation("_varLowerObjVariation", "_var_lower_obj_variation", None, -1, ( CpoSignature(Type_IntVarEval, ()),) )
Oper__var_upper_obj_variation     = CpoOperation("_varUpperObjVariation", "_var_upper_obj_variation", None, -1, ( CpoSignature(Type_IntVarEval, ()),) )


#-----------------------------------------------------------------------------
# Descriptors of public CPO operations
#-----------------------------------------------------------------------------

Oper_abs                         = CpoOperation("abs", "abs", None, -1, ( CpoSignature(Type_IntExpr, (Type_IntExpr,)),
                                                                          CpoSignature(Type_FloatExpr, (Type_FloatExpr,))) )
Oper_abstraction                 = CpoOperation("abstraction", "abstraction", None, -1, ( CpoSignature(Type_Constraint, (Type_IntExprArray, Type_IntExprArray, Type_IntArray, Type_Int)),) )
Oper_all_diff                    = CpoOperation("alldiff", "all_diff", None, -1, ( CpoSignature(Type_Constraint, (Type_IntExprArray,)),) )
Oper_all_min_distance            = CpoOperation("allMinDistance", "all_min_distance", None, -1, ( CpoSignature(Type_BoolExpr, (Type_IntExprArray, Type_Int)),) )
# Oper_allowed_assignments         = CpoOperation("allowedAssignments", "allowed_assignments", None, -1, ( CpoSignature(Type_BoolExpr, (Type_IntExpr, Type_IntArray)),
#                                                                                                          CpoSignature(Type_BoolExpr, (Type_IntExprArray, Type_TupleSet))) )  # Signature added to accept arrays of couples
# Signature added to allow array of couples that may appear as integer array
Oper_allowed_assignments         = CpoOperation("allowedAssignments", "allowed_assignments", None, -1, ( CpoSignature(Type_BoolExpr, (Type_IntExpr, Type_IntArray)),
                                                                                                         CpoSignature(Type_BoolExpr, (Type_IntExprArray, Type_TupleSet)),
                                                                                                         CpoSignature(Type_BoolExpr, (Type_IntExprArray, Type_IntArray))) )  # Signature added to accept arrays of couples
Oper_alternative                 = CpoOperation("alternative", "alternative", None, -1, ( CpoSignature(Type_Constraint, (Type_IntervalVar, Type_IntervalVarArray, CpoParam(Type_IntExpr, dval=0))),) )
Oper_always_constant             = CpoOperation("alwaysConstant", "always_constant", None, -1, ( CpoSignature(Type_Constraint, (Type_StateFunction, Type_IntervalVar, CpoParam(Type_BoolInt, dval=0), CpoParam(Type_BoolInt, dval=0))),
                                                                                                 CpoSignature(Type_Constraint, (Type_StateFunction, Type_TimeInt, Type_TimeInt, CpoParam(Type_BoolInt, dval=0), CpoParam(Type_BoolInt, dval=0)))) )
Oper_always_equal                = CpoOperation("alwaysEqual", "always_equal", None, -1, ( CpoSignature(Type_Constraint, (Type_StateFunction, Type_IntervalVar, Type_PositiveInt, CpoParam(Type_BoolInt, dval=0), CpoParam(Type_BoolInt, dval=0))),
                                                                                           CpoSignature(Type_Constraint, (Type_StateFunction, Type_TimeInt, Type_TimeInt, Type_PositiveInt, CpoParam(Type_BoolInt, dval=0), CpoParam(Type_BoolInt, dval=0)))) )
Oper_always_in                   = CpoOperation("alwaysIn", "always_in", None, -1, ( CpoSignature(Type_Constraint, (Type_CumulExpr, Type_IntervalVar, Type_PositiveInt, Type_PositiveInt)),
                                                                                     CpoSignature(Type_Constraint, (Type_CumulExpr, Type_TimeInt, Type_TimeInt, Type_PositiveInt, Type_PositiveInt)),
                                                                                     CpoSignature(Type_Constraint, (Type_StateFunction, Type_IntervalVar, Type_PositiveInt, Type_PositiveInt)),
                                                                                     CpoSignature(Type_Constraint, (Type_StateFunction, Type_TimeInt, Type_TimeInt, Type_PositiveInt, Type_PositiveInt))) )
Oper_always_no_state             = CpoOperation("alwaysNoState", "always_no_state", None, -1, ( CpoSignature(Type_Constraint, (Type_StateFunction, Type_IntervalVar)),
                                                                                                CpoSignature(Type_Constraint, (Type_StateFunction, Type_TimeInt, Type_TimeInt))) )
Oper_before                      = CpoOperation("before", "before", None, -1, ( CpoSignature(Type_Constraint, (Type_SequenceVar, Type_IntervalVar, Type_IntervalVar)),) )
Oper_bool_abstraction            = CpoOperation("boolAbstraction", "bool_abstraction", None, -1, ( CpoSignature(Type_Constraint, (Type_IntExprArray, Type_IntExprArray, Type_IntArray)),) )
Oper_bool_expr_array             = CpoOperation("boolExprArray", "bool_expr_array", None, -1, ( CpoSignature(Type_BoolExprArray, ()),) )
Oper_ceil                        = CpoOperation("ceil", "ceil", None, -1, ( CpoSignature(Type_IntExpr, (Type_FloatExpr,)),) )
Oper_conditional                 = CpoOperation("conditional", "conditional", None, -1, ( CpoSignature(Type_IntExpr, (Type_BoolExpr, Type_IntExpr, Type_IntExpr)),
                                                                                          CpoSignature(Type_FloatExpr, (Type_BoolExpr, Type_FloatExpr, Type_FloatExpr))) )
Oper_constant                    = CpoOperation("constant", "constant", "", -1, ( CpoSignature(Type_Int, (Type_Int,)),
                                                                                  CpoSignature(Type_Float, (Type_Float,))) )
Oper_coordinate_piecewise_linear = CpoOperation("coordinatePiecewiseLinear", "coordinate_piecewise_linear", None, -1, ( CpoSignature(Type_FloatExpr, (Type_FloatExpr, Type_Float, Type_FloatArray, Type_FloatArray, Type_Float)),) )
Oper_count                       = CpoOperation("count", "count", None, -1, ( CpoSignature(Type_IntExpr, (Type_IntExprArray, Type_Int)),
                                                                              CpoSignature(Type_IntExpr, (Type_Int, Type_IntExprArray))) )
Oper_count_different             = CpoOperation("countDifferent", "count_different", None, -1, ( CpoSignature(Type_IntExpr, (Type_IntExprArray,)),) )
Oper_cumul_expr_array            = CpoOperation("cumulExprArray", "cumul_expr_array", None, -1, ( CpoSignature(Type_CumulExprArray, ()),) )
Oper_cumul_range                 = CpoOperation("cumulRange", "cumul_range", None, -1, ( CpoSignature(Type_Constraint, (Type_CumulExpr, Type_IntExpr, Type_IntExpr)),) )
Oper_custom_value_chooser        = CpoOperation("customValueChooser", "custom_value_chooser", None, -1, ( CpoSignature(Type_IntValueChooser, ()),) )
Oper_custom_value_eval           = CpoOperation("customValueEval", "custom_value_eval", None, -1, ( CpoSignature(Type_IntValueEval, ()),) )
Oper_custom_var_chooser          = CpoOperation("customVarChooser", "custom_var_chooser", None, -1, ( CpoSignature(Type_IntVarChooser, ()),) )
Oper_custom_var_eval             = CpoOperation("customVarEval", "custom_var_eval", None, -1, ( CpoSignature(Type_IntVarEval, ()),) )
Oper_diff                        = CpoOperation("diff", "diff", "!=", 6, ( CpoSignature(Type_BoolExpr, (Type_IntExpr, Type_IntExpr)),
                                                                           CpoSignature(Type_BoolExpr, (Type_FloatExpr, Type_FloatExpr))) )
Oper_dist_to_int                 = CpoOperation("distToInt", "dist_to_int", None, -1, ( CpoSignature(Type_FloatExpr, (Type_FloatExpr,)),) )
Oper_distribute                  = CpoOperation("distribute", "distribute", None, -1, ( CpoSignature(Type_Constraint, (Type_IntExprArray, Type_IntArray, Type_IntExprArray)),
                                                                                        CpoSignature(Type_Constraint, (Type_IntExprArray, Type_IntExprArray))) )
Oper_domain_max                  = CpoOperation("domainMax", "domain_max", None, -1, ( CpoSignature(Type_IntVarEval, ()),) )
Oper_domain_min                  = CpoOperation("domainMin", "domain_min", None, -1, ( CpoSignature(Type_IntVarEval, ()),) )
Oper_domain_size                 = CpoOperation("domainSize", "domain_size", None, -1, ( CpoSignature(Type_IntVarEval, ()),) )
Oper_element                     = CpoOperation("element", "element", None, -1, ( CpoSignature(Type_IntExpr, (Type_IntArray, Type_IntExpr)),
                                                                                  CpoSignature(Type_IntExpr, (Type_IntExprArray, Type_IntExpr)),
                                                                                  CpoSignature(Type_FloatExpr, (Type_FloatArray, Type_IntExpr)),
                                                                                  CpoSignature(Type_IntExpr, (Type_IntExpr, Type_IntArray)),
                                                                                  CpoSignature(Type_IntExpr, (Type_IntExpr, Type_IntExprArray)),
                                                                                  CpoSignature(Type_FloatExpr, (Type_IntExpr, Type_FloatArray))) )
Oper_end_at_end                  = CpoOperation("endAtEnd", "end_at_end", None, -1, ( CpoSignature(Type_Constraint, (Type_IntervalVar, Type_IntervalVar, CpoParam(Type_IntExpr, dval=0))),) )
Oper_end_at_start                = CpoOperation("endAtStart", "end_at_start", None, -1, ( CpoSignature(Type_Constraint, (Type_IntervalVar, Type_IntervalVar, CpoParam(Type_IntExpr, dval=0))),) )
Oper_end_before_end              = CpoOperation("endBeforeEnd", "end_before_end", None, -1, ( CpoSignature(Type_Constraint, (Type_IntervalVar, Type_IntervalVar, CpoParam(Type_IntExpr, dval=0))),) )
Oper_end_before_start            = CpoOperation("endBeforeStart", "end_before_start", None, -1, ( CpoSignature(Type_Constraint, (Type_IntervalVar, Type_IntervalVar, CpoParam(Type_IntExpr, dval=0))),) )
Oper_end_eval                    = CpoOperation("endEval", "end_eval", None, -1, ( CpoSignature(Type_FloatExpr, (Type_IntervalVar, Type_SegmentedFunction, CpoParam(Type_Float, dval=0))),) )
Oper_end_of                      = CpoOperation("endOf", "end_of", None, -1, ( CpoSignature(Type_IntExpr, (Type_IntervalVar, CpoParam(Type_Int, dval=0))),) )
Oper_end_of_next                 = CpoOperation("endOfNext", "end_of_next", None, -1, ( CpoSignature(Type_IntExpr, (Type_SequenceVar, Type_IntervalVar, CpoParam(Type_Int, dval=0), CpoParam(Type_Int, dval=0))),) )
Oper_end_of_prev                 = CpoOperation("endOfPrev", "end_of_prev", None, -1, ( CpoSignature(Type_IntExpr, (Type_SequenceVar, Type_IntervalVar, CpoParam(Type_Int, dval=0), CpoParam(Type_Int, dval=0))),) )
Oper_equal                       = CpoOperation("equal", "equal", "==", 6, ( CpoSignature(Type_BoolExpr, (Type_IntExpr, Type_IntExpr)),
                                                                             CpoSignature(Type_BoolExpr, (Type_FloatExpr, Type_FloatExpr))) )
Oper_inferred                    = CpoOperation("inferred", "inferred", None, -1, (CpoSignature(Type_Constraint, (Type_IntVar,)),
                                                                                   CpoSignature(Type_Constraint, (Type_IntVarArray,))))
Oper_equal_or_escape             = CpoOperation("equalOrEscape", "equal_or_escape", None, -1, ( CpoSignature(Type_Constraint, (Type_IntExpr, Type_IntExpr, Type_Int)),) )
Oper_equivalence                 = CpoOperation("equivalence", "equivalence", None, -1, ( CpoSignature(Type_Constraint, (Type_IntervalVar, Type_IntervalVar)),) )
Oper_exotic_object               = CpoOperation("exoticObject", "exotic_object", None, -1, ( CpoSignature(Type_Constraint, ()),) )
Oper_explicit_value_eval         = CpoOperation("explicitValueEval", "explicit_value_eval", None, -1, ( CpoSignature(Type_IntValueEval, (Type_IntArray, Type_FloatArray, CpoParam(Type_Float, dval=0))),) )
Oper_explicit_var_eval           = CpoOperation("explicitVarEval", "explicit_var_eval", None, -1, ( CpoSignature(Type_IntVarEval, (Type_IntExprArray, Type_FloatArray, CpoParam(Type_Float, dval=0))),) )
Oper_exponent                    = CpoOperation("exponent", "exponent", None, -1, ( CpoSignature(Type_FloatExpr, (Type_FloatExpr,)),) )
Oper_false                       = CpoOperation("false", "false", None, -1, ( CpoSignature(Type_BoolExpr, ()),) )
Oper_first                       = CpoOperation("first", "first", None, -1, ( CpoSignature(Type_Constraint, (Type_SequenceVar, Type_IntervalVar)),) )
Oper_float_array                 = CpoOperation("floatArray", "float_array", None, -1, ( CpoSignature(Type_FloatArray, ()),) )
Oper_float_div                   = CpoOperation("floatDiv", "float_div", "/", 3, ( CpoSignature(Type_FloatExpr, (Type_FloatExpr, Type_FloatExpr)),) )
Oper_float_expr_array            = CpoOperation("floatExprArray", "float_expr_array", None, -1, ( CpoSignature(Type_FloatExprArray, ()),) )
Oper_float_to_int                = CpoOperation("floatToInt", "float_to_int", None, -1, ( CpoSignature(Type_IntExpr, (Type_FloatExpr,)),) )
Oper_floor                       = CpoOperation("floor", "floor", None, -1, ( CpoSignature(Type_IntExpr, (Type_FloatExpr,)),) )
Oper_forbid_end                  = CpoOperation("forbidEnd", "forbid_end", None, -1, ( CpoSignature(Type_Constraint, (Type_IntervalVar, Type_StepFunction)),) )
Oper_forbid_extent               = CpoOperation("forbidExtent", "forbid_extent", None, -1, ( CpoSignature(Type_Constraint, (Type_IntervalVar, Type_StepFunction)),) )
Oper_forbid_start                = CpoOperation("forbidStart", "forbid_start", None, -1, ( CpoSignature(Type_Constraint, (Type_IntervalVar, Type_StepFunction)),) )
# Oper_forbidden_assignments       = CpoOperation("forbiddenAssignments", "forbidden_assignments", None, -1, ( CpoSignature(Type_BoolExpr, (Type_IntExpr, Type_IntArray)),
#                                                                                                              CpoSignature(Type_BoolExpr, (Type_IntExprArray, Type_TupleSet))) )  # Signature added to accept arrays of couples
# Signature added to allow array of couples that may appear as integer array
Oper_forbidden_assignments       = CpoOperation("forbiddenAssignments", "forbidden_assignments", None, -1, ( CpoSignature(Type_BoolExpr, (Type_IntExpr, Type_IntArray)),
                                                                                                             CpoSignature(Type_BoolExpr, (Type_IntExprArray, Type_TupleSet)),
                                                                                                             CpoSignature(Type_BoolExpr, (Type_IntExprArray, Type_IntArray))) )  # Signature added to accept arrays of couples
Oper_fract                       = CpoOperation("fract", "fract", None, -1, ( CpoSignature(Type_FloatExpr, (Type_FloatExpr,)),) )
Oper_greater                     = CpoOperation("greater", "greater", ">", 5, ( CpoSignature(Type_BoolExpr, (Type_IntExpr, Type_IntExpr)),
                                                                                CpoSignature(Type_BoolExpr, (Type_FloatExpr, Type_FloatExpr))) )
Oper_greater_or_equal            = CpoOperation("greaterOrEqual", "greater_or_equal", ">=", 5, ( CpoSignature(Type_BoolExpr, (Type_IntExpr, Type_IntExpr)),
                                                                                                 CpoSignature(Type_BoolExpr, (Type_FloatExpr, Type_FloatExpr)),
                                                                                                 CpoSignature(Type_Constraint, (Type_PositiveInt, Type_CumulExpr)),
                                                                                                 CpoSignature(Type_Constraint, (Type_CumulExpr, Type_PositiveInt)),
                                                                                                 CpoSignature(Type_Constraint, (Type_CumulExpr, Type_IntExpr)),
                                                                                                 CpoSignature(Type_Constraint, (Type_IntExpr, Type_CumulExpr))) )
Oper_height_at_end               = CpoOperation("heightAtEnd", "height_at_end", None, -1, ( CpoSignature(Type_IntExpr, (Type_IntervalVar, Type_CumulExpr, CpoParam(Type_Int, dval=0))),) )
Oper_height_at_start             = CpoOperation("heightAtStart", "height_at_start", None, -1, ( CpoSignature(Type_IntExpr, (Type_IntervalVar, Type_CumulExpr, CpoParam(Type_Int, dval=0))),) )
Oper_if_then                     = CpoOperation("ifThen", "if_then", "=>", 6, ( CpoSignature(Type_BoolExpr, (Type_BoolExpr, Type_BoolExpr)),) )
Oper_impact_of_last_branch       = CpoOperation("impactOfLastBranch", "impact_of_last_branch", None, -1, ( CpoSignature(Type_IntVarEval, ()),) )
Oper_implies                     = CpoOperation("implies", "implies", None, -1, ( CpoSignature(Type_Constraint, (Type_IntervalVar, Type_IntervalVar)),) )
Oper_implies_not                 = CpoOperation("impliesNot", "implies_not", None, -1, ( CpoSignature(Type_Constraint, (Type_IntervalVar, Type_IntervalVar)),) )
Oper_int_array                   = CpoOperation("intArray", "int_array", None, -1, ( CpoSignature(Type_IntArray, ()),) )
Oper_int_div                     = CpoOperation("intDiv", "int_div", "div", 3, ( CpoSignature(Type_IntExpr, (Type_IntExpr, Type_IntExpr)),) )
Oper_int_expr_array              = CpoOperation("intExprArray", "int_expr_array", None, -1, ( CpoSignature(Type_IntExprArray, ()),) )
Oper_int_value_selector_array    = CpoOperation("intValueSelectorArray", "int_value_selector_array", None, -1, ( CpoSignature(Type_IntValueSelectorArray, ()),) )
Oper_int_var_selector_array      = CpoOperation("intVarSelectorArray", "int_var_selector_array", None, -1, ( CpoSignature(Type_IntVarSelectorArray, ()),) )
Oper_interval_var_array          = CpoOperation("intervalVarArray", "interval_var_array", None, -1, ( CpoSignature(Type_IntervalVarArray, ()),) )
Oper_inverse                     = CpoOperation("inverse", "inverse", None, -1, ( CpoSignature(Type_Constraint, (Type_IntExprArray, Type_IntExprArray)),) )
Oper_isomorphism                 = CpoOperation("isomorphism", "isomorphism", None, -1, ( CpoSignature(Type_Constraint, (Type_IntervalVarArray, Type_IntervalVarArray, CpoParam(Type_IntExprArray, dval=0), CpoParam(Type_Int, dval=0))),
                                                                                          CpoSignature(Type_Constraint, (Type_IntervalVarArray, Type_IntervalVarArray, CpoParam(Type_Int, dval=0), CpoParam(Type_IntExprArray, dval=0)))) )
Oper_last                        = CpoOperation("last", "last", None, -1, ( CpoSignature(Type_Constraint, (Type_SequenceVar, Type_IntervalVar)),) )
Oper_length_eval                 = CpoOperation("lengthEval", "length_eval", None, -1, ( CpoSignature(Type_FloatExpr, (Type_IntervalVar, Type_SegmentedFunction, CpoParam(Type_Float, dval=0))),) )
Oper_length_of                   = CpoOperation("lengthOf", "length_of", None, -1, ( CpoSignature(Type_IntExpr, (Type_IntervalVar, CpoParam(Type_Int, dval=0))),) )
Oper_length_of_next              = CpoOperation("lengthOfNext", "length_of_next", None, -1, ( CpoSignature(Type_IntExpr, (Type_SequenceVar, Type_IntervalVar, CpoParam(Type_Int, dval=0), CpoParam(Type_Int, dval=0))),) )
Oper_length_of_prev              = CpoOperation("lengthOfPrev", "length_of_prev", None, -1, ( CpoSignature(Type_IntExpr, (Type_SequenceVar, Type_IntervalVar, CpoParam(Type_Int, dval=0), CpoParam(Type_Int, dval=0))),) )
Oper_less                        = CpoOperation("less", "less", "<", 5, ( CpoSignature(Type_BoolExpr, (Type_IntExpr, Type_IntExpr)),
                                                                          CpoSignature(Type_BoolExpr, (Type_FloatExpr, Type_FloatExpr))) )
Oper_less_or_equal               = CpoOperation("lessOrEqual", "less_or_equal", "<=", 5, ( CpoSignature(Type_BoolExpr, (Type_IntExpr, Type_IntExpr)),
                                                                                           CpoSignature(Type_BoolExpr, (Type_FloatExpr, Type_FloatExpr)),
                                                                                           CpoSignature(Type_Constraint, (Type_CumulExpr, Type_PositiveInt)),
                                                                                           CpoSignature(Type_Constraint, (Type_PositiveInt, Type_CumulExpr)),
                                                                                           CpoSignature(Type_Constraint, (Type_CumulExpr, Type_IntExpr)),
                                                                                           CpoSignature(Type_Constraint, (Type_IntExpr, Type_CumulExpr))) )
Oper_lexicographic               = CpoOperation("lexicographic", "lexicographic", None, -1, ( CpoSignature(Type_Constraint, (Type_IntExprArray, Type_IntExprArray)),) )
Oper_strict_lexicographic        = CpoOperation("strictLexicographic", "strict_lexicographic", None, -1, ( CpoSignature(Type_Constraint, (Type_IntExprArray, Type_IntExprArray)),) )
Oper_log                         = CpoOperation("log", "log", None, -1, ( CpoSignature(Type_FloatExpr, (Type_FloatExpr,)),) )
Oper_logical_and                 = CpoOperation("and", "logical_and", "&&", 7, ( CpoSignature(Type_BoolExpr, (Type_BoolExpr, Type_BoolExpr)),
                                                                                 CpoSignature(Type_BoolExpr, (Type_BoolExprArray,)),) )
Oper_logical_not                 = CpoOperation("not", "logical_not", "!", 1, ( CpoSignature(Type_BoolExpr, (Type_BoolExpr,)),) )
Oper_logical_or                  = CpoOperation("or", "logical_or", "||", 8, ( CpoSignature(Type_BoolExpr, (Type_BoolExpr, Type_BoolExpr)),
                                                                               CpoSignature(Type_BoolExpr, (Type_BoolExprArray,)),))
Oper_max                         = CpoOperation("max", "max", None, -1, ( CpoSignature(Type_IntExpr, (Type_IntExpr, Type_IntExpr)),
                                                                          CpoSignature(Type_IntExpr, (Type_IntExprArray,)),
                                                                          CpoSignature(Type_FloatExpr, (Type_FloatExpr, Type_FloatExpr)),
                                                                          CpoSignature(Type_FloatExpr, (Type_FloatExprArray,))) )
Oper_maximize                    = CpoOperation("maximize", "maximize", None, -1, ( CpoSignature(Type_Objective, (Type_FloatExpr,)),
                                                                                    CpoSignature(Type_Objective, (Type_FloatExprArray,))) )
Oper_maximize_static_lex         = CpoOperation("maximizeStaticLex", "maximize_static_lex", None, -1, ( CpoSignature(Type_Objective, (Type_FloatExprArray,)),) )
Oper_member                      = CpoOperation("member", "member", None, -1, ( CpoSignature(Type_BoolExpr, (Type_IntExpr, Type_IntArray)),) )
Oper_min                         = CpoOperation("min", "min", None, -1, ( CpoSignature(Type_IntExpr, (Type_IntExpr, Type_IntExpr)),
                                                                          CpoSignature(Type_IntExpr, (Type_IntExprArray,)),
                                                                          CpoSignature(Type_FloatExpr, (Type_FloatExpr, Type_FloatExpr)),
                                                                          CpoSignature(Type_FloatExpr, (Type_FloatExprArray,))) )
Oper_minimize                    = CpoOperation("minimize", "minimize", None, -1, ( CpoSignature(Type_Objective, (Type_FloatExpr,)),
                                                                                    CpoSignature(Type_Objective, (Type_FloatExprArray,))) )
Oper_minimize_static_lex         = CpoOperation("minimizeStaticLex", "minimize_static_lex", None, -1, ( CpoSignature(Type_Objective, (Type_FloatExprArray,)),) )
Oper_minus                       = CpoOperation("minus", "minus", "-", 4, ( CpoSignature(Type_IntExpr, (Type_IntExpr, Type_IntExpr)),
                                                                            CpoSignature(Type_IntExpr, (Type_IntExpr,)),
                                                                            CpoSignature(Type_FloatExpr, (Type_FloatExpr, Type_FloatExpr)),
                                                                            CpoSignature(Type_FloatExpr, (Type_FloatExpr,)),
                                                                            CpoSignature(Type_CumulExpr, (Type_CumulExpr, Type_CumulExpr)),
                                                                            CpoSignature(Type_CumulExpr, (Type_CumulExpr,))) )
Oper_mod                         = CpoOperation("mod", "mod", "%", 3, ( CpoSignature(Type_IntExpr, (Type_IntExpr, Type_IntExpr)),) )
Oper_multi_span                  = CpoOperation("multiSpan", "multi_span", None, -1, ( CpoSignature(Type_Constraint, (Type_IntervalVarArray, Type_IntervalVarArray, CpoParam(Type_Int, dval=0), CpoParam(Type_IntExprArray, dval=0))),) )
Oper_next                        = CpoOperation("next", "next", None, -1, ( CpoSignature(Type_Constraint, (Type_SequenceVar, Type_IntervalVar, Type_IntervalVar)),) )
Oper_no_overlap                  = CpoOperation("noOverlap", "no_overlap", None, -1, ( CpoSignature(Type_Constraint, (Type_SequenceVar, CpoParam(Type_TransitionMatrix, dval=0), CpoParam(Type_BoolInt, dval=0))),
                                                                                       CpoSignature(Type_Constraint, (Type_IntervalVarArray,))) )
Oper_not_member                  = CpoOperation("notMember", "not_member", None, -1, ( CpoSignature(Type_BoolExpr, (Type_IntExpr, Type_IntArray)),) )
Oper_opposite                    = CpoOperation("opposite", "opposite", None, -1, ( CpoSignature(Type_Constraint, (Type_IntervalVar, Type_IntervalVar)),) )
Oper_overlap_length              = CpoOperation("overlapLength", "overlap_length", None, -1, ( CpoSignature(Type_IntExpr, (Type_IntervalVar, Type_IntervalVar, CpoParam(Type_Int, dval=0))),
                                                                                               CpoSignature(Type_IntExpr, (Type_IntervalVar, Type_TimeInt, Type_TimeInt, CpoParam(Type_Int, dval=0)))) )
Oper_pack                        = CpoOperation("pack", "pack", None, -1, ( CpoSignature(Type_Constraint, (Type_IntExprArray, Type_IntExprArray, Type_IntArray, CpoParam(Type_IntExpr, dval=0))),) )
Oper_phase                       = CpoOperation("phase", "phase", None, -1, ( CpoSignature(Type_SearchPhase, (Type_IntExprArray,)),
                                                                              CpoSignature(Type_SearchPhase, (Type_IntervalVarArray,))) )
Oper_piecewise_linear            = CpoOperation("piecewiseLinear", "piecewise_linear", None, -1, ( CpoSignature(Type_FloatExpr, (Type_FloatExpr, Type_SegmentedFunction)),) )
Oper_plus                        = CpoOperation("plus", "plus", "+", 4, ( CpoSignature(Type_IntExpr, (Type_IntExpr, Type_IntExpr)),
                                                                          CpoSignature(Type_FloatExpr, (Type_FloatExpr, Type_FloatExpr)),
                                                                          CpoSignature(Type_CumulExpr, (Type_CumulExpr, Type_CumulExpr))) )
Oper_power                       = CpoOperation("power", "power", "^", 2, ( CpoSignature(Type_FloatExpr, (Type_FloatExpr, Type_FloatExpr)),) )
Oper_presence_of                 = CpoOperation("presenceOf", "presence_of", None, -1, ( CpoSignature(Type_BoolExpr, (Type_IntervalVar,)),) )
Oper_previous                    = CpoOperation("previous", "previous", None, -1, ( CpoSignature(Type_Constraint, (Type_SequenceVar, Type_IntervalVar, Type_IntervalVar)),) )
Oper_pulse                       = CpoOperation("pulse", "pulse", None, -1, ( CpoSignature(Type_CumulAtom, (Type_TimeInt, Type_TimeInt, Type_PositiveInt)),
                                                                              CpoSignature(Type_CumulAtom, (Type_IntervalVar, Type_PositiveInt)),
                                                                              CpoSignature(Type_CumulAtom, (Type_IntervalVar, Type_PositiveInt, Type_PositiveInt))) )
Oper_random_value                = CpoOperation("RandomValue", "random_value", None, -1, ( CpoSignature(Type_IntValueEval, ()),) )
Oper_range                       = CpoOperation("range", "range", None, -1, ( CpoSignature(Type_BoolExpr, (Type_IntExpr, Type_Float, Type_Float)),
                                                                              CpoSignature(Type_BoolExpr, (Type_FloatExpr, Type_Float, Type_Float))) )
Oper_round                       = CpoOperation("round", "round", None, -1, ( CpoSignature(Type_IntExpr, (Type_FloatExpr,)),) )
Oper_same_common_sub_sequence    = CpoOperation("sameCommonSubSequence", "same_common_sub_sequence", None, -1, ( CpoSignature(Type_Constraint, (Type_SequenceVar, Type_SequenceVar, Type_IntervalVarArray, Type_IntervalVarArray, Type_BoolInt)),) )
Oper_same_common_subsequence     = CpoOperation("sameCommonSubsequence", "same_common_subsequence", None, -1, ( CpoSignature(Type_Constraint, (Type_SequenceVar, Type_SequenceVar)),
                                                                                                                CpoSignature(Type_Constraint, (Type_SequenceVar, Type_SequenceVar, Type_IntervalVarArray, Type_IntervalVarArray))) )
Oper_same_sequence               = CpoOperation("sameSequence", "same_sequence", None, -1, ( CpoSignature(Type_Constraint, (Type_SequenceVar, Type_SequenceVar)),
                                                                                             CpoSignature(Type_Constraint, (Type_SequenceVar, Type_SequenceVar, Type_IntervalVarArray, Type_IntervalVarArray))) )
Oper_scal_prod                   = CpoOperation("scalProd", "scal_prod", None, -1, ( CpoSignature(Type_IntExpr, (Type_IntArray, Type_IntExprArray)),
                                                                                     CpoSignature(Type_IntExpr, (Type_IntExprArray, Type_IntArray)),
                                                                                     CpoSignature(Type_IntExpr, (Type_IntExprArray, Type_IntExprArray)),
                                                                                     CpoSignature(Type_FloatExpr, (Type_FloatArray, Type_FloatExprArray)),
                                                                                     CpoSignature(Type_FloatExpr, (Type_FloatExprArray, Type_FloatArray)),
                                                                                     CpoSignature(Type_FloatExpr, (Type_FloatExprArray, Type_FloatExprArray))) )
Oper_search_phase                = CpoOperation("searchPhase", "search_phase", None, -1, ( CpoSignature(Type_SearchPhase, (Type_IntExprArray,)),
                                                                                           CpoSignature(Type_SearchPhase, (Type_IntVarChooser, Type_IntValueChooser)),
                                                                                           CpoSignature(Type_SearchPhase, (Type_IntExprArray, Type_IntVarChooser, Type_IntValueChooser)),
                                                                                           CpoSignature(Type_SearchPhase, (Type_IntervalVarArray,)),
                                                                                           CpoSignature(Type_SearchPhase, (Type_SequenceVarArray,))) )
Oper_search_phase_int            = CpoOperation("searchPhaseInt", "search_phase_int", None, -1, ( CpoSignature(Type_SearchPhase, (Type_IntExprArray,)),
                                                                                                  CpoSignature(Type_SearchPhase, (Type_IntVarChooser, Type_IntValueChooser)),
                                                                                                  CpoSignature(Type_SearchPhase, (Type_IntExprArray, Type_IntVarChooser, Type_IntValueChooser))) )
Oper_search_phase_rank           = CpoOperation("searchPhaseRank", "search_phase_rank", None, -1, ( CpoSignature(Type_SearchPhase, (CpoParam(Type_SequenceVarArray, dval=0),)),) )
Oper_search_phase_set_times      = CpoOperation("searchPhaseSetTimes", "search_phase_set_times", None, -1, ( CpoSignature(Type_SearchPhase, (CpoParam(Type_IntervalVarArray, dval=0),)),) )
Oper_segmented_function          = CpoOperation("segmentedFunction", "segmented_function", None, -1, ( CpoSignature(Type_SegmentedFunction, ()),) )
Oper_select_largest              = CpoOperation("selectLargest", "select_largest", None, -1, ( CpoSignature(Type_IntVarSelector, (Type_Float, Type_IntVarEval)),
                                                                                               CpoSignature(Type_IntVarSelector, (Type_IntVarEval, CpoParam(Type_Float, dval=0))),
                                                                                               CpoSignature(Type_IntValueSelector, (Type_Float, Type_IntValueEval)),
                                                                                               CpoSignature(Type_IntValueSelector, (Type_IntValueEval, CpoParam(Type_Float, dval=0))),
                                                                                               CpoSignature(Type_IntVarSelector, (Type_IntVarEval, Type_Int, Type_Float)),
                                                                                               CpoSignature(Type_IntValueSelector, (Type_IntValueEval, Type_Int, Type_Float))) )
Oper_select_random_value         = CpoOperation("selectRandomValue", "select_random_value", None, -1, ( CpoSignature(Type_IntValueSelector, ()),) )
Oper_select_random_var           = CpoOperation("selectRandomVar", "select_random_var", None, -1, ( CpoSignature(Type_IntVarSelector, ()),) )
Oper_select_smallest             = CpoOperation("selectSmallest", "select_smallest", None, -1, ( CpoSignature(Type_IntVarSelector, (Type_Float, Type_IntVarEval)),
                                                                                                 CpoSignature(Type_IntVarSelector, (Type_IntVarEval, CpoParam(Type_Float, dval=0))),
                                                                                                 CpoSignature(Type_IntValueSelector, (Type_Float, Type_IntValueEval)),
                                                                                                 CpoSignature(Type_IntValueSelector, (Type_IntValueEval, CpoParam(Type_Float, dval=0))),
                                                                                                 CpoSignature(Type_IntVarSelector, (Type_IntVarEval, Type_Int, Type_Float)),
                                                                                                 CpoSignature(Type_IntValueSelector, (Type_IntValueEval, Type_Int, Type_Float))) )
Oper_sequence                    = CpoOperation("sequence", "sequence", None, -1, ( CpoSignature(Type_Constraint, (Type_Int, Type_Int, Type_Int, Type_IntExprArray, Type_IntArray, Type_IntExprArray)),) )
Oper_sequence_var_array          = CpoOperation("sequenceVarArray", "sequence_var_array", None, -1, ( CpoSignature(Type_SequenceVarArray, ()),) )
Oper_sgn                         = CpoOperation("sgn", "sgn", None, -1, ( CpoSignature(Type_IntExpr, (Type_FloatExpr,)),) )
Oper_size_eval                   = CpoOperation("sizeEval", "size_eval", None, -1, ( CpoSignature(Type_FloatExpr, (Type_IntervalVar, Type_SegmentedFunction, CpoParam(Type_Float, dval=0))),) )
Oper_size_of                     = CpoOperation("sizeOf", "size_of", None, -1, ( CpoSignature(Type_IntExpr, (Type_IntervalVar, CpoParam(Type_Int, dval=0))),) )
Oper_size_of_next                = CpoOperation("sizeOfNext", "size_of_next", None, -1, ( CpoSignature(Type_IntExpr, (Type_SequenceVar, Type_IntervalVar, CpoParam(Type_Int, dval=0), CpoParam(Type_Int, dval=0))),) )
Oper_size_of_prev                = CpoOperation("sizeOfPrev", "size_of_prev", None, -1, ( CpoSignature(Type_IntExpr, (Type_SequenceVar, Type_IntervalVar, CpoParam(Type_Int, dval=0), CpoParam(Type_Int, dval=0))),) )
Oper_slope_piecewise_linear      = CpoOperation("slopePiecewiseLinear", "slope_piecewise_linear", None, -1, ( CpoSignature(Type_FloatExpr, (Type_FloatExpr, Type_FloatArray, Type_FloatArray, Type_Float, Type_Float)),) )
Oper_span                        = CpoOperation("span", "span", None, -1, ( CpoSignature(Type_Constraint, (Type_IntervalVar, Type_IntervalVarArray)),) )
Oper_spread                      = CpoOperation("spread", "spread", None, -1, ( CpoSignature(Type_Constraint, (Type_IntExprArray, Type_FloatExpr, Type_FloatExpr)),) )
Oper_square                      = CpoOperation("square", "square", None, -1, ( CpoSignature(Type_IntExpr, (Type_IntExpr,)),
                                                                                CpoSignature(Type_FloatExpr, (Type_FloatExpr,))) )
Oper_standard_deviation          = CpoOperation("standardDeviation", "standard_deviation", None, -1, ( CpoSignature(Type_FloatExpr, (Type_IntExprArray, Type_Float, Type_Float)),
                                                                                                       CpoSignature(Type_FloatExpr, (Type_IntExprArray,)),) )

Oper_start_at_end                = CpoOperation("startAtEnd", "start_at_end", None, -1, ( CpoSignature(Type_Constraint, (Type_IntervalVar, Type_IntervalVar, CpoParam(Type_IntExpr, dval=0))),) )
Oper_start_at_start              = CpoOperation("startAtStart", "start_at_start", None, -1, ( CpoSignature(Type_Constraint, (Type_IntervalVar, Type_IntervalVar, CpoParam(Type_IntExpr, dval=0))),) )
Oper_start_before_end            = CpoOperation("startBeforeEnd", "start_before_end", None, -1, ( CpoSignature(Type_Constraint, (Type_IntervalVar, Type_IntervalVar, CpoParam(Type_IntExpr, dval=0))),) )
Oper_start_before_start          = CpoOperation("startBeforeStart", "start_before_start", None, -1, ( CpoSignature(Type_Constraint, (Type_IntervalVar, Type_IntervalVar, CpoParam(Type_IntExpr, dval=0))),) )
Oper_start_eval                  = CpoOperation("startEval", "start_eval", None, -1, ( CpoSignature(Type_FloatExpr, (Type_IntervalVar, Type_SegmentedFunction, CpoParam(Type_Float, dval=0))),) )
Oper_start_of                    = CpoOperation("startOf", "start_of", None, -1, ( CpoSignature(Type_IntExpr, (Type_IntervalVar, CpoParam(Type_Int, dval=0))),) )
Oper_start_of_next               = CpoOperation("startOfNext", "start_of_next", None, -1, ( CpoSignature(Type_IntExpr, (Type_SequenceVar, Type_IntervalVar, CpoParam(Type_Int, dval=0), CpoParam(Type_Int, dval=0))),) )
Oper_start_of_prev               = CpoOperation("startOfPrev", "start_of_prev", None, -1, ( CpoSignature(Type_IntExpr, (Type_SequenceVar, Type_IntervalVar, CpoParam(Type_Int, dval=0), CpoParam(Type_Int, dval=0))),) )
Oper_step_at                     = CpoOperation("stepAt", "step_at", None, -1, ( CpoSignature(Type_CumulAtom, (Type_TimeInt, Type_PositiveInt)),) )
Oper_step_at_end                 = CpoOperation("stepAtEnd", "step_at_end", None, -1, ( CpoSignature(Type_CumulAtom, (Type_IntervalVar, Type_PositiveInt)),
                                                                                        CpoSignature(Type_CumulAtom, (Type_IntervalVar, Type_PositiveInt, Type_PositiveInt))) )
Oper_step_at_start               = CpoOperation("stepAtStart", "step_at_start", None, -1, ( CpoSignature(Type_CumulAtom, (Type_IntervalVar, Type_PositiveInt)),
                                                                                            CpoSignature(Type_CumulAtom, (Type_IntervalVar, Type_PositiveInt, Type_PositiveInt))) )
Oper_step_function               = CpoOperation("stepFunction", "step_function", None, -1, ( CpoSignature(Type_StepFunction, ()),) )
Oper_strong                      = CpoOperation("strong", "strong", None, -1, ( CpoSignature(Type_Constraint, (Type_IntExprArray,)),) )
Oper_strong_constraint           = CpoOperation("StrongConstraint", "strong_constraint", None, -1, ( CpoSignature(Type_Constraint, (Type_IntExprArray,)),) )
Oper_sub_circuit                 = CpoOperation("subCircuit", "sub_circuit", None, -1, ( CpoSignature(Type_Constraint, (Type_IntExprArray,)),) )
Oper_sum                         = CpoOperation("sum", "sum", None, -1, ( CpoSignature(Type_IntExpr, (Type_IntExprArray,)),
                                                                          CpoSignature(Type_FloatExpr, (Type_FloatExprArray,)),
                                                                          CpoSignature(Type_CumulExpr, (Type_CumulExprArray,))) )
Oper_synchronize                 = CpoOperation("synchronize", "synchronize", None, -1, ( CpoSignature(Type_Constraint, (Type_IntervalVar, Type_IntervalVarArray)),) )
Oper_table_element               = CpoOperation("tableElement", "table_element", None, -1, ( CpoSignature(Type_Constraint, (Type_IntExpr, Type_IntArray, Type_IntExpr)),) )
Oper_times                       = CpoOperation("times", "times", "*", 3, ( CpoSignature(Type_IntExpr, (Type_IntExpr, Type_IntExpr)),
                                                                            CpoSignature(Type_FloatExpr, (Type_FloatExpr, Type_FloatExpr))) )
Oper_transition_matrix           = CpoOperation("transitionMatrix", "transition_matrix", None, -1, ( CpoSignature(Type_TransitionMatrix, ()),) )
Oper_true                        = CpoOperation("true", "true", None, -1, ( CpoSignature(Type_BoolExpr, ()),) )
Oper_trunc                       = CpoOperation("trunc", "trunc", None, -1, ( CpoSignature(Type_IntExpr, (Type_FloatExpr,)),) )
Oper_tuple_set                   = CpoOperation("tupleSet", "tuple_set", None, -1, ( CpoSignature(Type_TupleSet, ()),) )
Oper_type_of_next                = CpoOperation("typeOfNext", "type_of_next", None, -1, ( CpoSignature(Type_IntExpr, (Type_SequenceVar, Type_IntervalVar, CpoParam(Type_Int, dval=0), CpoParam(Type_Int, dval=0))),) )
Oper_type_of_prev                = CpoOperation("typeOfPrev", "type_of_prev", None, -1, ( CpoSignature(Type_IntExpr, (Type_SequenceVar, Type_IntervalVar, CpoParam(Type_Int, dval=0), CpoParam(Type_Int, dval=0))),) )
Oper_value                       = CpoOperation("value", "value", None, -1, ( CpoSignature(Type_IntValueEval, ()),) )
Oper_value_impact                = CpoOperation("valueImpact", "value_impact", None, -1, ( CpoSignature(Type_IntValueEval, ()),) )
Oper_value_index                 = CpoOperation("valueIndex", "value_index", None, -1, ( CpoSignature(Type_IntValueEval, (Type_IntArray, CpoParam(Type_Float, dval=-1))),) )
Oper_value_index_eval            = CpoOperation("valueIndexEval", "value_index_eval", None, -1, ( CpoSignature(Type_IntValueEval, (Type_IntArray, CpoParam(Type_Float, dval=-1))),) )
Oper_value_local_impact          = CpoOperation("ValueLocalImpact", "value_local_impact", None, -1, ( CpoSignature(Type_IntValueEval, ()),) )
Oper_value_lower_obj_variation   = CpoOperation("ValueLowerObjVariation", "value_lower_obj_variation", None, -1, ( CpoSignature(Type_IntValueEval, ()),) )
Oper_value_success_rate          = CpoOperation("ValueSuccessRate", "value_success_rate", None, -1, ( CpoSignature(Type_IntValueEval, ()),) )
Oper_value_upper_obj_variation   = CpoOperation("ValueUpperObjVariation", "value_upper_obj_variation", None, -1, ( CpoSignature(Type_IntValueEval, ()),) )
Oper_var_impact                  = CpoOperation("varImpact", "var_impact", None, -1, ( CpoSignature(Type_IntVarEval, ()),) )
Oper_var_index                   = CpoOperation("varIndex", "var_index", None, -1, ( CpoSignature(Type_IntVarEval, (Type_IntExprArray, CpoParam(Type_Float, dval=-1))),) )
Oper_var_index_eval              = CpoOperation("varIndexEval", "var_index_eval", None, -1, ( CpoSignature(Type_IntVarEval, (Type_IntExprArray, CpoParam(Type_Float, dval=-1))),) )
Oper_var_local_impact            = CpoOperation("varLocalImpact", "var_local_impact", None, -1, ( CpoSignature(Type_IntVarEval, (CpoParam(Type_Int, dval=-1),)),) )
Oper_var_success_rate            = CpoOperation("varSuccessRate", "var_success_rate", None, -1, ( CpoSignature(Type_IntVarEval, ()),) )

Oper_eval                        = CpoOperation("eval", "eval", None, -1, ( CpoSignature(Type_FloatExpr, (Type_Blackbox, Type_Int)),) )


# Mutable expressions
Oper_mutable_bool_expr           = CpoOperation("mutableBoolExpr", "mutable_bool_expr ", None, -1, ( CpoSignature(Type_BoolExpr, ()),
                                                                                                     CpoSignature(Type_BoolExpr, (Type_BoolExpr,))) )
Oper_mutable_int_expr            = CpoOperation("mutableIntExpr", "mutable_int_expr ", None, -1, ( CpoSignature(Type_IntExpr, ()),
                                                                                                   CpoSignature(Type_IntExpr, (Type_IntExpr,))) )
Oper_mutable_float_expr          = CpoOperation("mutableFloatExpr", "mutable_float_expr ", None, -1, ( CpoSignature(Type_FloatExpr, ()),
                                                                                                       CpoSignature(Type_FloatExpr, (Type_FloatExpr,))) )
Oper_mutable_cumul_expr          = CpoOperation("mutableCumulExpr", "mutable_cumul_expr ", None, -1, ( CpoSignature(Type_CumulExpr, ()),
                                                                                                       CpoSignature(Type_CumulExpr, (Type_CumulExpr,))) )


# Duplicatas because of typos, not same char case
Oper_all_diff_2                  = CpoOperation("allDiff", "all_diff", None, -1, ( CpoSignature(Type_Constraint, (Type_IntExprArray,)),) )
Oper_value_2                     = CpoOperation("Value", "value", None, -1, ( CpoSignature(Type_IntValueEval, ()),) )
Oper_value_impact_2              = CpoOperation("ValueImpact", "value_impact", None, -1, ( CpoSignature(Type_IntValueEval, ()),) )
Oper_value_success_rate_2        = CpoOperation("valueSuccessRate", "value_success_rate", None, -1, ( CpoSignature(Type_IntValueEval, ()),) )
Oper_var_index_eval_2            = CpoOperation("VarIndexEval", "var_index_eval", None, -1, ( CpoSignature(Type_IntVarEval, (Type_IntExprArray, CpoParam(Type_Float, dval=-1))),) )


# Variable creators (hardcoded in the parser)
Oper_int_var                     = CpoOperation("intVar", "int_var", None, -1, ( ) )
Oper__int_var                    = CpoOperation("_intVar", "_int_var", None, -1, ( ) )
Oper_bool_var                    = CpoOperation("boolVar", "bool_var", None, -1, ( ) )
Oper__bool_var                   = CpoOperation("_boolVar", "_bool_var", None, -1, ( ) )
Oper_float_var                   = CpoOperation("floatVar", "float_var", None, -1, ( CpoSignature(Type_FloatVar, (Type_Float, Type_Float,)),) )
Oper__float_var                  = CpoOperation("_floatVar", "_float_var", None, -1, ( CpoSignature(Type_FloatVar, (Type_Float, Type_Float)),) )
Oper_interval_var                = CpoOperation("intervalVar", "interval_var", None, -1, ( ) )
Oper_sequence_var                = CpoOperation("sequenceVar", "sequence_var", None, -1, ( CpoSignature(Type_SequenceVar, (Type_IntervalVarArray, CpoParam(Type_IntArray, dval=0))),) )
Oper_state_function              = CpoOperation("stateFunction", "state_function", None, -1, ( CpoSignature(Type_StateFunction, (CpoParam(Type_TransitionMatrix, dval=0),)),) )
Oper__interval_array             = CpoOperation("_intervalArray", "_interval_array", None, -1, ( CpoSignature(Type_IntervalArray, ()),) )


#-----------------------------------------------------------------------------
# Private methods
#-----------------------------------------------------------------------------

def _compute_all_type_links(ltypes):
    """ Compute all links between the different data types.

    Args:
        ltypes: List of all types
    """
    # Allocate id to each each type
    nbtypes = len(ltypes)
    for i, tp in enumerate(ltypes):
        tp.id = i

    # Compute kind of for each type
    for tp1 in ltypes:
        tp1.kind_of_types = tuple(map(lambda tp2: (tp2.base_type in tp1.higher_types), ltypes))

    # Compute common type
    for tp1 in ltypes:
        ctypes = [None] * nbtypes
        for tp2 in ltypes:
            ct = tp1._compute_common_type(tp2)
            if ct is not None:
                ctypes[tp2.id] = ct
        tp1.common_types = tuple(ctypes)

    # Special case for intervals and arrays
    Type_IntInterval._force_common_type(Type_IntArray, Type_IntArray)
    Type_IntArray._force_common_type(Type_IntInterval, Type_IntArray)


#-----------------------------------------------------------------------------
# Build working structures
#-----------------------------------------------------------------------------

import sys
_module = sys.modules[__name__]
_attrs = dir(_module)

# Build sorted list of all types, starting by those used in variables, in declaration order
_VAR_TYPES_IN_ORDER = (Type_Bool, Type_Int, Type_Float, Type_IntArray, Type_FloatArray, Type_TupleSet,
                       Type_SegmentedFunction, Type_StepFunction, Type_TransitionMatrix,
                       Type_IntVar, Type_FloatVar, Type_StateFunction, Type_IntervalVar, Type_SequenceVar)
ALL_TYPES = list(_VAR_TYPES_IN_ORDER)
_tset = set(ALL_TYPES)
for t in [getattr(_module, x) for x in _attrs if x.startswith("Type_")]:
    if t not in _tset:
        ALL_TYPES.append(t)
        _tset.add(t)
ALL_TYPES = tuple(ALL_TYPES)

# Compute all dependency links between types
_compute_all_type_links(ALL_TYPES)

# Build a dictionary of types per name
ALL_TYPES_PER_NAME = {t.get_name(): t for t in ALL_TYPES}

# Build list of all operations
ALL_OPERATIONS = tuple(getattr(_module, x) for x in _attrs if x.startswith("Oper_"))

# Build a dictionary of operations per CPO name
ALL_OPERATIONS_PER_NAME = {o.get_cpo_name(): o for o in ALL_OPERATIONS}
