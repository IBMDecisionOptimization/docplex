# ------------------------------------------------------------------------
# File: model_info.py
# ------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2008, 2026. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# ------------------------------------------------------------------------
"""Modeling information IDs returned by the Callable Library.

This module defines symbolic names for the integer modeling information
IDs returned by the Callable Library. The names to which the modeling
information IDs are assigned are the same names used in the Callable
Library, all of which begin with CPXMI. The modeling information IDs are
accessible through the modeling assistance callback. These symbolic names
can be used to test if a particular modeling issue has been detected.

See `Cplex.set_modeling_assistance_callback`.
"""


CPXMI_BIGM_COEF = 1040
"""See :macros:`CPXMI_BIGM_COEF` in the C API."""

CPXMI_BIGM_TO_IND = 1041
"""See :macros:`CPXMI_BIGM_TO_IND` in the C API."""

CPXMI_BIGM_VARBOUND = 1042
"""See :macros:`CPXMI_BIGM_VARBOUND` in the C API."""

CPXMI_CANCEL_TOL = 1045
"""See :macros:`CPXMI_CANCEL_TOL` in the C API."""

CPXMI_EPGAP_LARGE = 1038
"""See :macros:`CPXMI_EPGAP_LARGE` in the C API."""

CPXMI_EPGAP_OBJOFFSET = 1037
"""See :macros:`CPXMI_EPGAP_OBJOFFSET` in the C API."""

CPXMI_FEAS_TOL = 1043
"""See :macros:`CPXMI_FEAS_TOL` in the C API."""

CPXMI_FRACTION_SCALING = 1047
"""See :macros:`CPXMI_FRACTION_SCALING` in the C API."""

CPXMI_IND_NZ_LARGE_NUM = 1019
"""See :macros:`CPXMI_IND_NZ_LARGE_NUM` in the C API."""

CPXMI_IND_NZ_SMALL_NUM = 1020
"""See :macros:`CPXMI_IND_NZ_SMALL_NUM` in the C API."""

CPXMI_IND_RHS_LARGE_NUM = 1021
"""See :macros:`CPXMI_IND_RHS_LARGE_NUM` in the C API."""

CPXMI_IND_RHS_SMALL_NUM = 1022
"""See :macros:`CPXMI_IND_RHS_SMALL_NUM` in the C API."""

CPXMI_KAPPA_ILLPOSED = 1035
"""See :macros:`CPXMI_KAPPA_ILLPOSED` in the C API."""

CPXMI_KAPPA_SUSPICIOUS = 1033
"""See :macros:`CPXMI_KAPPA_SUSPICIOUS` in the C API."""

CPXMI_KAPPA_UNSTABLE = 1034
"""See :macros:`CPXMI_KAPPA_UNSTABLE` in the C API."""

CPXMI_LB_LARGE_NUM = 1003
"""See :macros:`CPXMI_LB_LARGE_NUM` in the C API."""

CPXMI_LB_SMALL_NUM = 1004
"""See :macros:`CPXMI_LB_SMALL_NUM` in the C API."""

CPXMI_LC_NZ_LARGE_NUM = 1023
"""See :macros:`CPXMI_LC_NZ_LARGE_NUM` in the C API."""

CPXMI_LC_NZ_SMALL_NUM = 1024
"""See :macros:`CPXMI_LC_NZ_SMALL_NUM` in the C API."""

CPXMI_LC_RHS_LARGE_NUM = 1025
"""See :macros:`CPXMI_LC_RHS_LARGE_NUM` in the C API."""

CPXMI_LC_RHS_SMALL_NUM = 1026
"""See :macros:`CPXMI_LC_RHS_SMALL_NUM` in the C API."""

CPXMI_MULTIOBJ_COEFFS = 1062
"""See :macros:`CPXMI_MULTIOBJ_COEFFS` in the C API."""

CPXMI_MULTIOBJ_LARGE_NUM = 1058
"""See :macros:`CPXMI_MULTIOBJ_LARGE_NUM` in the C API."""

CPXMI_MULTIOBJ_MIX = 1063
"""See :macros:`CPXMI_MULTIOBJ_MIX` in the C API."""

CPXMI_MULTIOBJ_OPT_TOL = 1060
"""See :macros:`CPXMI_MULTIOBJ_OPT_TOL` in the C API."""

CPXMI_MULTIOBJ_SMALL_NUM = 1059
"""See :macros:`CPXMI_MULTIOBJ_SMALL_NUM` in the C API."""

CPXMI_NZ_LARGE_NUM = 1009
"""See :macros:`CPXMI_NZ_LARGE_NUM` in the C API."""

CPXMI_NZ_SMALL_NUM = 1010
"""See :macros:`CPXMI_NZ_SMALL_NUM` in the C API."""

CPXMI_OBJ_LARGE_NUM = 1001
"""See :macros:`CPXMI_OBJ_LARGE_NUM` in the C API."""

CPXMI_OBJ_SMALL_NUM = 1002
"""See :macros:`CPXMI_OBJ_SMALL_NUM` in the C API."""

CPXMI_OPT_TOL = 1044
"""See :macros:`CPXMI_OPT_TOL` in the C API."""

CPXMI_PWL_SLOPE_LARGE_NUM = 1064
"""See :macros:`CPXMI_PWL_SLOPE_LARGE_NUM` in the C API."""

CPXMI_PWL_SLOPE_SMALL_NUM = 1065
"""See :macros:`CPXMI_PWL_SLOPE_SMALL_NUM` in the C API."""

CPXMI_QC_LINNZ_LARGE_NUM = 1015
"""See :macros:`CPXMI_QC_LINNZ_LARGE_NUM` in the C API."""

CPXMI_QC_LINNZ_SMALL_NUM = 1016
"""See :macros:`CPXMI_QC_LINNZ_SMALL_NUM` in the C API."""

CPXMI_QC_QNZ_LARGE_NUM = 1017
"""See :macros:`CPXMI_QC_QNZ_LARGE_NUM` in the C API."""

CPXMI_QC_QNZ_SMALL_NUM = 1018
"""See :macros:`CPXMI_QC_QNZ_SMALL_NUM` in the C API."""

CPXMI_QC_RHS_LARGE_NUM = 1013
"""See :macros:`CPXMI_QC_RHS_LARGE_NUM` in the C API."""

CPXMI_QC_RHS_SMALL_NUM = 1014
"""See :macros:`CPXMI_QC_RHS_SMALL_NUM` in the C API."""

CPXMI_QOBJ_LARGE_NUM = 1011
"""See :macros:`CPXMI_QOBJ_LARGE_NUM` in the C API."""

CPXMI_QOBJ_SMALL_NUM = 1012
"""See :macros:`CPXMI_QOBJ_SMALL_NUM` in the C API."""

CPXMI_QOPT_TOL = 1046
"""See :macros:`CPXMI_QOPT_TOL` in the C API."""

CPXMI_RHS_LARGE_NUM = 1007
"""See :macros:`CPXMI_RHS_LARGE_NUM` in the C API."""

CPXMI_RHS_SMALL_NUM = 1008
"""See :macros:`CPXMI_RHS_SMALL_NUM` in the C API."""

CPXMI_SAMECOEFF_COL = 1050
"""See :macros:`CPXMI_SAMECOEFF_COL` in the C API."""

CPXMI_SAMECOEFF_IND = 1051
"""See :macros:`CPXMI_SAMECOEFF_IND` in the C API."""

CPXMI_SAMECOEFF_LAZY = 1054
"""See :macros:`CPXMI_SAMECOEFF_LAZY` in the C API."""

CPXMI_SAMECOEFF_MULTIOBJ = 1061
"""See :macros:`CPXMI_SAMECOEFF_MULTIOBJ` in the C API."""

CPXMI_SAMECOEFF_OBJ = 1057
"""See :macros:`CPXMI_SAMECOEFF_OBJ` in the C API."""

CPXMI_SAMECOEFF_QLIN = 1052
"""See :macros:`CPXMI_SAMECOEFF_QLIN` in the C API."""

CPXMI_SAMECOEFF_QUAD = 1053
"""See :macros:`CPXMI_SAMECOEFF_QUAD` in the C API."""

CPXMI_SAMECOEFF_RHS = 1056
"""See :macros:`CPXMI_SAMECOEFF_RHS` in the C API."""

CPXMI_SAMECOEFF_ROW = 1049
"""See :macros:`CPXMI_SAMECOEFF_ROW` in the C API."""

CPXMI_SAMECOEFF_UCUT = 1055
"""See :macros:`CPXMI_SAMECOEFF_UCUT` in the C API."""

CPXMI_SINGLE_PRECISION = 1036
"""See :macros:`CPXMI_SINGLE_PRECISION` in the C API."""

CPXMI_SYMMETRY_BREAKING_INEQ = 1039
"""See :macros:`CPXMI_SYMMETRY_BREAKING_INEQ` in the C API."""

CPXMI_UB_LARGE_NUM = 1005
"""See :macros:`CPXMI_UB_LARGE_NUM` in the C API."""

CPXMI_UB_SMALL_NUM = 1006
"""See :macros:`CPXMI_UB_SMALL_NUM` in the C API."""

CPXMI_UC_NZ_LARGE_NUM = 1027
"""See :macros:`CPXMI_UC_NZ_LARGE_NUM` in the C API."""

CPXMI_UC_NZ_SMALL_NUM = 1028
"""See :macros:`CPXMI_UC_NZ_SMALL_NUM` in the C API."""

CPXMI_UC_RHS_LARGE_NUM = 1029
"""See :macros:`CPXMI_UC_RHS_LARGE_NUM` in the C API."""

CPXMI_UC_RHS_SMALL_NUM = 1030
"""See :macros:`CPXMI_UC_RHS_SMALL_NUM` in the C API."""

CPXMI_WIDE_COEFF_RANGE = 1048
"""See :macros:`CPXMI_WIDE_COEFF_RANGE` in the C API."""
