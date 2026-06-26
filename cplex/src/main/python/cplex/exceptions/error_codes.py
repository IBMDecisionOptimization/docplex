# ------------------------------------------------------------------------
# File: exceptions/error_codes.py
# ------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2008, 2026. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# ------------------------------------------------------------------------
"""Error codes returned by the Callable Library.

This module defines symbolic names for the integer error codes returned by
the Callable Library. The names to which the error codes are assigned are
the same names used in the Callable Library, all of which begin with
CPXERR. The error codes are accessible as the third element of the args
attribute of the exception that is raised. These symbolic names should be
used to test if a particular error has occurred.

Example usage:

>>> import cplex
>>> from cplex.exceptions import CplexSolverError
>>> try:
...    c = cplex.Cplex()
... except CplexSolverError as exc:
...    if exc.args[2] == cplex.exceptions.error_codes.CPXERR_NO_MEMORY:
...        pass  # handle a specific exception
...    raise
"""


CPXERR_ABORT_STRONGBRANCH = 1263
"""See :macros:`CPXERR_ABORT_STRONGBRANCH` in the C API."""

CPXERR_ADJ_SIGN_QUAD = 1606
"""See :macros:`CPXERR_ADJ_SIGN_QUAD` in the C API."""

CPXERR_ADJ_SIGN_SENSE = 1604
"""See :macros:`CPXERR_ADJ_SIGN_SENSE` in the C API."""

CPXERR_ADJ_SIGNS = 1602
"""See :macros:`CPXERR_ADJ_SIGNS` in the C API."""

CPXERR_ARC_INDEX_RANGE = 1231
"""See :macros:`CPXERR_ARC_INDEX_RANGE` in the C API."""

CPXERR_ARRAY_BAD_SOS_TYPE = 3009
"""See :macros:`CPXERR_ARRAY_BAD_SOS_TYPE` in the C API."""

CPXERR_ARRAY_NOT_ASCENDING = 1226
"""See :macros:`CPXERR_ARRAY_NOT_ASCENDING` in the C API."""

CPXERR_ARRAY_TOO_LONG = 1208
"""See :macros:`CPXERR_ARRAY_TOO_LONG` in the C API."""

CPXERR_BAD_ARGUMENT = 1003
"""See :macros:`CPXERR_BAD_ARGUMENT` in the C API."""

CPXERR_BAD_BOUND_SENSE = 1622
"""See :macros:`CPXERR_BAD_BOUND_SENSE` in the C API."""

CPXERR_BAD_BOUND_TYPE = 1457
"""See :macros:`CPXERR_BAD_BOUND_TYPE` in the C API."""

CPXERR_BAD_CHAR = 1537
"""See :macros:`CPXERR_BAD_CHAR` in the C API."""

CPXERR_BAD_CTYPE = 3021
"""See :macros:`CPXERR_BAD_CTYPE` in the C API."""

CPXERR_BAD_DECOMPOSITION = 2002
"""See :macros:`CPXERR_BAD_DECOMPOSITION` in the C API."""

CPXERR_BAD_DIRECTION = 3012
"""See :macros:`CPXERR_BAD_DIRECTION` in the C API."""

CPXERR_BAD_EXPO_RANGE = 1435
"""See :macros:`CPXERR_BAD_EXPO_RANGE` in the C API."""

CPXERR_BAD_EXPONENT = 1618
"""See :macros:`CPXERR_BAD_EXPONENT` in the C API."""

CPXERR_BAD_FILETYPE = 1424
"""See :macros:`CPXERR_BAD_FILETYPE` in the C API."""

CPXERR_BAD_ID = 1617
"""See :macros:`CPXERR_BAD_ID` in the C API."""

CPXERR_BAD_INDCONSTR = 1439
"""See :macros:`CPXERR_BAD_INDCONSTR` in the C API."""

CPXERR_BAD_INDICATOR = 1551
"""See :macros:`CPXERR_BAD_INDICATOR` in the C API."""

CPXERR_BAD_INDTYPE = 1216
"""See :macros:`CPXERR_BAD_INDTYPE` in the C API."""

CPXERR_BAD_LAZY_UCUT = 1438
"""See :macros:`CPXERR_BAD_LAZY_UCUT` in the C API."""

CPXERR_BAD_LUB = 1229
"""See :macros:`CPXERR_BAD_LUB` in the C API."""

CPXERR_BAD_METHOD = 1292
"""See :macros:`CPXERR_BAD_METHOD` in the C API."""

CPXERR_BAD_MULTIOBJ_ATTR = 1488
"""See :macros:`CPXERR_BAD_MULTIOBJ_ATTR` in the C API."""

CPXERR_BAD_NAME = 1220
"""See :macros:`CPXERR_BAD_NAME` in the C API."""

CPXERR_BAD_NUMBER = 1434
"""See :macros:`CPXERR_BAD_NUMBER` in the C API."""

CPXERR_BAD_OBJ_SENSE = 1487
"""See :macros:`CPXERR_BAD_OBJ_SENSE` in the C API."""

CPXERR_BAD_PARAM_NAME = 1028
"""See :macros:`CPXERR_BAD_PARAM_NAME` in the C API."""

CPXERR_BAD_PARAM_NUM = 1013
"""See :macros:`CPXERR_BAD_PARAM_NUM` in the C API."""

CPXERR_BAD_PIVOT = 1267
"""See :macros:`CPXERR_BAD_PIVOT` in the C API."""

CPXERR_BAD_PRIORITY = 3006
"""See :macros:`CPXERR_BAD_PRIORITY` in the C API."""

CPXERR_BAD_PROB_TYPE = 1022
"""See :macros:`CPXERR_BAD_PROB_TYPE` in the C API."""

CPXERR_BAD_ROW_ID = 1532
"""See :macros:`CPXERR_BAD_ROW_ID` in the C API."""

CPXERR_BAD_SECTION_BOUNDS = 1473
"""See :macros:`CPXERR_BAD_SECTION_BOUNDS` in the C API."""

CPXERR_BAD_SECTION_ENDATA = 1462
"""See :macros:`CPXERR_BAD_SECTION_ENDATA` in the C API."""

CPXERR_BAD_SECTION_QMATRIX = 1475
"""See :macros:`CPXERR_BAD_SECTION_QMATRIX` in the C API."""

CPXERR_BAD_SENSE = 1215
"""See :macros:`CPXERR_BAD_SENSE` in the C API."""

CPXERR_BAD_SOS_TYPE = 1442
"""See :macros:`CPXERR_BAD_SOS_TYPE` in the C API."""

CPXERR_BAD_STATUS = 1253
"""See :macros:`CPXERR_BAD_STATUS` in the C API."""

CPXERR_BAS_FILE_SHORT = 1550
"""See :macros:`CPXERR_BAS_FILE_SHORT` in the C API."""

CPXERR_BAS_FILE_SIZE = 1555
"""See :macros:`CPXERR_BAS_FILE_SIZE` in the C API."""

CPXERR_BENDERS_MASTER_SOLVE = 2001
"""See :macros:`CPXERR_BENDERS_MASTER_SOLVE` in the C API."""

CPXERR_CALLBACK = 1006
"""See :macros:`CPXERR_CALLBACK` in the C API."""

CPXERR_CALLBACK_INCONSISTENT = 1060
"""See :macros:`CPXERR_CALLBACK_INCONSISTENT` in the C API."""

CPXERR_CAND_NOT_POINT = 3025
"""See :macros:`CPXERR_CAND_NOT_POINT` in the C API."""

CPXERR_CAND_NOT_RAY = 3026
"""See :macros:`CPXERR_CAND_NOT_RAY` in the C API."""

CPXERR_CNTRL_IN_NAME = 1236
"""See :macros:`CPXERR_CNTRL_IN_NAME` in the C API."""

CPXERR_COL_INDEX_RANGE = 1201
"""See :macros:`CPXERR_COL_INDEX_RANGE` in the C API."""

CPXERR_COL_REPEAT_PRINT = 1478
"""See :macros:`CPXERR_COL_REPEAT_PRINT` in the C API."""

CPXERR_COL_REPEATS = 1446
"""See :macros:`CPXERR_COL_REPEATS` in the C API."""

CPXERR_COL_ROW_REPEATS = 1443
"""See :macros:`CPXERR_COL_ROW_REPEATS` in the C API."""

CPXERR_COL_UNKNOWN = 1449
"""See :macros:`CPXERR_COL_UNKNOWN` in the C API."""

CPXERR_CONFLICT_UNSTABLE = 1720
"""See :macros:`CPXERR_CONFLICT_UNSTABLE` in the C API."""

CPXERR_COUNT_OVERLAP = 1228
"""See :macros:`CPXERR_COUNT_OVERLAP` in the C API."""

CPXERR_COUNT_RANGE = 1227
"""See :macros:`CPXERR_COUNT_RANGE` in the C API."""

CPXERR_CPUBINDING_FAILURE = 3700
"""See :macros:`CPXERR_CPUBINDING_FAILURE` in the C API."""

CPXERR_DBL_MAX = 1233
"""See :macros:`CPXERR_DBL_MAX` in the C API."""

CPXERR_DECOMPRESSION = 1027
"""See :macros:`CPXERR_DECOMPRESSION` in the C API."""

CPXERR_DETTILIM_STRONGBRANCH = 1270
"""See :macros:`CPXERR_DETTILIM_STRONGBRANCH` in the C API."""

CPXERR_DUP_ENTRY = 1222
"""See :macros:`CPXERR_DUP_ENTRY` in the C API."""

CPXERR_DYNFUNC = 1815
"""See :macros:`CPXERR_DYNFUNC` in the C API."""

CPXERR_DYNLOAD = 1814
"""See :macros:`CPXERR_DYNLOAD` in the C API."""

CPXERR_ENCODING_CONVERSION = 1235
"""See :macros:`CPXERR_ENCODING_CONVERSION` in the C API."""

CPXERR_EXTRA_BV_BOUND = 1456
"""See :macros:`CPXERR_EXTRA_BV_BOUND` in the C API."""

CPXERR_EXTRA_FR_BOUND = 1455
"""See :macros:`CPXERR_EXTRA_FR_BOUND` in the C API."""

CPXERR_EXTRA_FX_BOUND = 1454
"""See :macros:`CPXERR_EXTRA_FX_BOUND` in the C API."""

CPXERR_EXTRA_INTEND = 1481
"""See :macros:`CPXERR_EXTRA_INTEND` in the C API."""

CPXERR_EXTRA_INTORG = 1480
"""See :macros:`CPXERR_EXTRA_INTORG` in the C API."""

CPXERR_EXTRA_SOSEND = 1483
"""See :macros:`CPXERR_EXTRA_SOSEND` in the C API."""

CPXERR_EXTRA_SOSORG = 1482
"""See :macros:`CPXERR_EXTRA_SOSORG` in the C API."""

CPXERR_FAIL_OPEN_READ = 1423
"""See :macros:`CPXERR_FAIL_OPEN_READ` in the C API."""

CPXERR_FAIL_OPEN_WRITE = 1422
"""See :macros:`CPXERR_FAIL_OPEN_WRITE` in the C API."""

CPXERR_FILE_ENTRIES = 1553
"""See :macros:`CPXERR_FILE_ENTRIES` in the C API."""

CPXERR_FILE_FORMAT = 1563
"""See :macros:`CPXERR_FILE_FORMAT` in the C API."""

CPXERR_FILE_IO = 1426
"""See :macros:`CPXERR_FILE_IO` in the C API."""

CPXERR_FILTER_VARIABLE_TYPE = 3414
"""See :macros:`CPXERR_FILTER_VARIABLE_TYPE` in the C API."""

CPXERR_ILL_DEFINED_PWL = 1213
"""See :macros:`CPXERR_ILL_DEFINED_PWL` in the C API."""

CPXERR_IN_INFOCALLBACK = 1804
"""See :macros:`CPXERR_IN_INFOCALLBACK` in the C API."""

CPXERR_INDEX_NOT_BASIC = 1251
"""See :macros:`CPXERR_INDEX_NOT_BASIC` in the C API."""

CPXERR_INDEX_RANGE = 1200
"""See :macros:`CPXERR_INDEX_RANGE` in the C API."""

CPXERR_INDEX_RANGE_HIGH = 1206
"""See :macros:`CPXERR_INDEX_RANGE_HIGH` in the C API."""

CPXERR_INDEX_RANGE_LOW = 1205
"""See :macros:`CPXERR_INDEX_RANGE_LOW` in the C API."""

CPXERR_INT_TOO_BIG = 3018
"""See :macros:`CPXERR_INT_TOO_BIG` in the C API."""

CPXERR_INT_TOO_BIG_INPUT = 1463
"""See :macros:`CPXERR_INT_TOO_BIG_INPUT` in the C API."""

CPXERR_INVALID_NUMBER = 1650
"""See :macros:`CPXERR_INVALID_NUMBER` in the C API."""

CPXERR_LIMITS_TOO_BIG = 1012
"""See :macros:`CPXERR_LIMITS_TOO_BIG` in the C API."""

CPXERR_LINE_TOO_LONG = 1465
"""See :macros:`CPXERR_LINE_TOO_LONG` in the C API."""

CPXERR_LO_BOUND_REPEATS = 1459
"""See :macros:`CPXERR_LO_BOUND_REPEATS` in the C API."""

CPXERR_LOCK_CREATE = 1808
"""See :macros:`CPXERR_LOCK_CREATE` in the C API."""

CPXERR_LP_NOT_IN_ENVIRONMENT = 1806
"""See :macros:`CPXERR_LP_NOT_IN_ENVIRONMENT` in the C API."""

CPXERR_LP_PARSE = 1427
"""See :macros:`CPXERR_LP_PARSE` in the C API."""

CPXERR_MASTER_SOLVE = 2005
"""See :macros:`CPXERR_MASTER_SOLVE` in the C API."""

CPXERR_MIPSEARCH_WITH_CALLBACKS = 1805
"""See :macros:`CPXERR_MIPSEARCH_WITH_CALLBACKS` in the C API."""

CPXERR_MISS_SOS_TYPE = 3301
"""See :macros:`CPXERR_MISS_SOS_TYPE` in the C API."""

CPXERR_MSG_NO_CHANNEL = 1051
"""See :macros:`CPXERR_MSG_NO_CHANNEL` in the C API."""

CPXERR_MSG_NO_FILEPTR = 1052
"""See :macros:`CPXERR_MSG_NO_FILEPTR` in the C API."""

CPXERR_MSG_NO_FUNCTION = 1053
"""See :macros:`CPXERR_MSG_NO_FUNCTION` in the C API."""

CPXERR_MULTIOBJ_SUBPROB_SOLVE = 1300
"""See :macros:`CPXERR_MULTIOBJ_SUBPROB_SOLVE` in the C API."""

CPXERR_NAME_CREATION = 1209
"""See :macros:`CPXERR_NAME_CREATION` in the C API."""

CPXERR_NAME_NOT_FOUND = 1210
"""See :macros:`CPXERR_NAME_NOT_FOUND` in the C API."""

CPXERR_NAME_TOO_LONG = 1464
"""See :macros:`CPXERR_NAME_TOO_LONG` in the C API."""

CPXERR_NAN = 1225
"""See :macros:`CPXERR_NAN` in the C API."""

CPXERR_NEED_OPT_SOLN = 1252
"""See :macros:`CPXERR_NEED_OPT_SOLN` in the C API."""

CPXERR_NEGATIVE_SURPLUS = 1207
"""See :macros:`CPXERR_NEGATIVE_SURPLUS` in the C API."""

CPXERR_NET_DATA = 1530
"""See :macros:`CPXERR_NET_DATA` in the C API."""

CPXERR_NET_FILE_SHORT = 1538
"""See :macros:`CPXERR_NET_FILE_SHORT` in the C API."""

CPXERR_NO_BARRIER_SOLN = 1223
"""See :macros:`CPXERR_NO_BARRIER_SOLN` in the C API."""

CPXERR_NO_BASIC_SOLN = 1261
"""See :macros:`CPXERR_NO_BASIC_SOLN` in the C API."""

CPXERR_NO_BASIS = 1262
"""See :macros:`CPXERR_NO_BASIS` in the C API."""

CPXERR_NO_BOUND_SENSE = 1621
"""See :macros:`CPXERR_NO_BOUND_SENSE` in the C API."""

CPXERR_NO_BOUND_TYPE = 1460
"""See :macros:`CPXERR_NO_BOUND_TYPE` in the C API."""

CPXERR_NO_COLUMNS_SECTION = 1472
"""See :macros:`CPXERR_NO_COLUMNS_SECTION` in the C API."""

CPXERR_NO_CONFLICT = 1719
"""See :macros:`CPXERR_NO_CONFLICT` in the C API."""

CPXERR_NO_DECOMPOSITION = 2000
"""See :macros:`CPXERR_NO_DECOMPOSITION` in the C API."""

CPXERR_NO_DUAL_SOLN = 1232
"""See :macros:`CPXERR_NO_DUAL_SOLN` in the C API."""

CPXERR_NO_ENDATA = 1552
"""See :macros:`CPXERR_NO_ENDATA` in the C API."""

CPXERR_NO_ENVIRONMENT = 1002
"""See :macros:`CPXERR_NO_ENVIRONMENT` in the C API."""

CPXERR_NO_FILENAME = 1421
"""See :macros:`CPXERR_NO_FILENAME` in the C API."""

CPXERR_NO_ID = 1616
"""See :macros:`CPXERR_NO_ID` in the C API."""

CPXERR_NO_ID_FIRST = 1609
"""See :macros:`CPXERR_NO_ID_FIRST` in the C API."""

CPXERR_NO_INT_X = 3023
"""See :macros:`CPXERR_NO_INT_X` in the C API."""

CPXERR_NO_KAPPASTATS = 1269
"""See :macros:`CPXERR_NO_KAPPASTATS` in the C API."""

CPXERR_NO_LU_FACTOR = 1258
"""See :macros:`CPXERR_NO_LU_FACTOR` in the C API."""

CPXERR_NO_MEMORY = 1001
"""See :macros:`CPXERR_NO_MEMORY` in the C API."""

CPXERR_NO_MIPSTART = 3020
"""See :macros:`CPXERR_NO_MIPSTART` in the C API."""

CPXERR_NO_NAME_SECTION = 1441
"""See :macros:`CPXERR_NO_NAME_SECTION` in the C API."""

CPXERR_NO_NAMES = 1219
"""See :macros:`CPXERR_NO_NAMES` in the C API."""

CPXERR_NO_NORMS = 1264
"""See :macros:`CPXERR_NO_NORMS` in the C API."""

CPXERR_NO_NUMBER = 1615
"""See :macros:`CPXERR_NO_NUMBER` in the C API."""

CPXERR_NO_NUMBER_BOUND = 1623
"""See :macros:`CPXERR_NO_NUMBER_BOUND` in the C API."""

CPXERR_NO_NUMBER_FIRST = 1611
"""See :macros:`CPXERR_NO_NUMBER_FIRST` in the C API."""

CPXERR_NO_OBJ_NAME = 1489
"""See :macros:`CPXERR_NO_OBJ_NAME` in the C API."""

CPXERR_NO_OBJ_SENSE = 1436
"""See :macros:`CPXERR_NO_OBJ_SENSE` in the C API."""

CPXERR_NO_OBJECTIVE = 1476
"""See :macros:`CPXERR_NO_OBJECTIVE` in the C API."""

CPXERR_NO_OP_OR_SENSE = 1608
"""See :macros:`CPXERR_NO_OP_OR_SENSE` in the C API."""

CPXERR_NO_OPERATOR = 1607
"""See :macros:`CPXERR_NO_OPERATOR` in the C API."""

CPXERR_NO_ORDER = 3016
"""See :macros:`CPXERR_NO_ORDER` in the C API."""

CPXERR_NO_PROBLEM = 1009
"""See :macros:`CPXERR_NO_PROBLEM` in the C API."""

CPXERR_NO_QP_OPERATOR = 1614
"""See :macros:`CPXERR_NO_QP_OPERATOR` in the C API."""

CPXERR_NO_QUAD_EXP = 1612
"""See :macros:`CPXERR_NO_QUAD_EXP` in the C API."""

CPXERR_NO_RHS_COEFF = 1610
"""See :macros:`CPXERR_NO_RHS_COEFF` in the C API."""

CPXERR_NO_RHS_IN_OBJ = 1211
"""See :macros:`CPXERR_NO_RHS_IN_OBJ` in the C API."""

CPXERR_NO_ROW_NAME = 1486
"""See :macros:`CPXERR_NO_ROW_NAME` in the C API."""

CPXERR_NO_ROW_SENSE = 1453
"""See :macros:`CPXERR_NO_ROW_SENSE` in the C API."""

CPXERR_NO_ROWS_SECTION = 1471
"""See :macros:`CPXERR_NO_ROWS_SECTION` in the C API."""

CPXERR_NO_SENSIT = 1260
"""See :macros:`CPXERR_NO_SENSIT` in the C API."""

CPXERR_NO_SOLN = 1217
"""See :macros:`CPXERR_NO_SOLN` in the C API."""

CPXERR_NO_SOLNPOOL = 3024
"""See :macros:`CPXERR_NO_SOLNPOOL` in the C API."""

CPXERR_NO_SOS = 3015
"""See :macros:`CPXERR_NO_SOS` in the C API."""

CPXERR_NO_TREE = 3412
"""See :macros:`CPXERR_NO_TREE` in the C API."""

CPXERR_NO_VECTOR_SOLN = 1556
"""See :macros:`CPXERR_NO_VECTOR_SOLN` in the C API."""

CPXERR_NODE_INDEX_RANGE = 1230
"""See :macros:`CPXERR_NODE_INDEX_RANGE` in the C API."""

CPXERR_NODE_ON_DISK = 3504
"""See :macros:`CPXERR_NODE_ON_DISK` in the C API."""

CPXERR_NOT_DUAL_UNBOUNDED = 1265
"""See :macros:`CPXERR_NOT_DUAL_UNBOUNDED` in the C API."""

CPXERR_NOT_FIXED = 1221
"""See :macros:`CPXERR_NOT_FIXED` in the C API."""

CPXERR_NOT_FOR_BENDERS = 2004
"""See :macros:`CPXERR_NOT_FOR_BENDERS` in the C API."""

CPXERR_NOT_FOR_DISTMIP = 1071
"""See :macros:`CPXERR_NOT_FOR_DISTMIP` in the C API."""

CPXERR_NOT_FOR_MIP = 1017
"""See :macros:`CPXERR_NOT_FOR_MIP` in the C API."""

CPXERR_NOT_FOR_MULTIOBJ = 1070
"""See :macros:`CPXERR_NOT_FOR_MULTIOBJ` in the C API."""

CPXERR_NOT_FOR_QCP = 1031
"""See :macros:`CPXERR_NOT_FOR_QCP` in the C API."""

CPXERR_NOT_FOR_QP = 1018
"""See :macros:`CPXERR_NOT_FOR_QP` in the C API."""

CPXERR_NOT_MILPCLASS = 1024
"""See :macros:`CPXERR_NOT_MILPCLASS` in the C API."""

CPXERR_NOT_MIN_COST_FLOW = 1531
"""See :macros:`CPXERR_NOT_MIN_COST_FLOW` in the C API."""

CPXERR_NOT_MIP = 3003
"""See :macros:`CPXERR_NOT_MIP` in the C API."""

CPXERR_NOT_MIQPCLASS = 1029
"""See :macros:`CPXERR_NOT_MIQPCLASS` in the C API."""

CPXERR_NOT_ONE_PROBLEM = 1023
"""See :macros:`CPXERR_NOT_ONE_PROBLEM` in the C API."""

CPXERR_NOT_QP = 5004
"""See :macros:`CPXERR_NOT_QP` in the C API."""

CPXERR_NOT_SAV_FILE = 1560
"""See :macros:`CPXERR_NOT_SAV_FILE` in the C API."""

CPXERR_NOT_UNBOUNDED = 1254
"""See :macros:`CPXERR_NOT_UNBOUNDED` in the C API."""

CPXERR_NULL_POINTER = 1004
"""See :macros:`CPXERR_NULL_POINTER` in the C API."""

CPXERR_ORDER_BAD_DIRECTION = 3007
"""See :macros:`CPXERR_ORDER_BAD_DIRECTION` in the C API."""

CPXERR_OVERFLOW = 1810
"""See :macros:`CPXERR_OVERFLOW` in the C API."""

CPXERR_PARAM_INCOMPATIBLE = 1807
"""See :macros:`CPXERR_PARAM_INCOMPATIBLE` in the C API."""

CPXERR_PARAM_TOO_BIG = 1015
"""See :macros:`CPXERR_PARAM_TOO_BIG` in the C API."""

CPXERR_PARAM_TOO_SMALL = 1014
"""See :macros:`CPXERR_PARAM_TOO_SMALL` in the C API."""

CPXERR_PRESLV_ABORT = 1106
"""See :macros:`CPXERR_PRESLV_ABORT` in the C API."""

CPXERR_PRESLV_BASIS_MEM = 1107
"""See :macros:`CPXERR_PRESLV_BASIS_MEM` in the C API."""

CPXERR_PRESLV_COPYORDER = 1109
"""See :macros:`CPXERR_PRESLV_COPYORDER` in the C API."""

CPXERR_PRESLV_COPYSOS = 1108
"""See :macros:`CPXERR_PRESLV_COPYSOS` in the C API."""

CPXERR_PRESLV_CRUSHFORM = 1121
"""See :macros:`CPXERR_PRESLV_CRUSHFORM` in the C API."""

CPXERR_PRESLV_DETTIME_LIM = 1124
"""See :macros:`CPXERR_PRESLV_DETTIME_LIM` in the C API."""

CPXERR_PRESLV_DUAL = 1119
"""See :macros:`CPXERR_PRESLV_DUAL` in the C API."""

CPXERR_PRESLV_FAIL_BASIS = 1114
"""See :macros:`CPXERR_PRESLV_FAIL_BASIS` in the C API."""

CPXERR_PRESLV_INF = 1117
"""See :macros:`CPXERR_PRESLV_INF` in the C API."""

CPXERR_PRESLV_INForUNBD = 1101
"""See :macros:`CPXERR_PRESLV_INForUNBD` in the C API."""

CPXERR_PRESLV_NO_BASIS = 1115
"""See :macros:`CPXERR_PRESLV_NO_BASIS` in the C API."""

CPXERR_PRESLV_NO_PROB = 1103
"""See :macros:`CPXERR_PRESLV_NO_PROB` in the C API."""

CPXERR_PRESLV_SOLN_MIP = 1110
"""See :macros:`CPXERR_PRESLV_SOLN_MIP` in the C API."""

CPXERR_PRESLV_SOLN_QP = 1111
"""See :macros:`CPXERR_PRESLV_SOLN_QP` in the C API."""

CPXERR_PRESLV_START_LP = 1112
"""See :macros:`CPXERR_PRESLV_START_LP` in the C API."""

CPXERR_PRESLV_TIME_LIM = 1123
"""See :macros:`CPXERR_PRESLV_TIME_LIM` in the C API."""

CPXERR_PRESLV_UNBD = 1118
"""See :macros:`CPXERR_PRESLV_UNBD` in the C API."""

CPXERR_PRESLV_UNCRUSHFORM = 1120
"""See :macros:`CPXERR_PRESLV_UNCRUSHFORM` in the C API."""

CPXERR_PRIIND = 1257
"""See :macros:`CPXERR_PRIIND` in the C API."""

CPXERR_PRM_DATA = 1660
"""See :macros:`CPXERR_PRM_DATA` in the C API."""

CPXERR_Q_DIVISOR = 1619
"""See :macros:`CPXERR_Q_DIVISOR` in the C API."""

CPXERR_Q_DUP_ENTRY = 5011
"""See :macros:`CPXERR_Q_DUP_ENTRY` in the C API."""

CPXERR_Q_NOT_INDEF = 5014
"""See :macros:`CPXERR_Q_NOT_INDEF` in the C API."""

CPXERR_Q_NOT_POS_DEF = 5002
"""See :macros:`CPXERR_Q_NOT_POS_DEF` in the C API."""

CPXERR_Q_NOT_SYMMETRIC = 5012
"""See :macros:`CPXERR_Q_NOT_SYMMETRIC` in the C API."""

CPXERR_QCP_SENSE = 6002
"""See :macros:`CPXERR_QCP_SENSE` in the C API."""

CPXERR_QCP_SENSE_FILE = 1437
"""See :macros:`CPXERR_QCP_SENSE_FILE` in the C API."""

CPXERR_QUAD_EXP_NOT_2 = 1613
"""See :macros:`CPXERR_QUAD_EXP_NOT_2` in the C API."""

CPXERR_QUAD_IN_ROW = 1605
"""See :macros:`CPXERR_QUAD_IN_ROW` in the C API."""

CPXERR_RANGE_SECTION_ORDER = 1474
"""See :macros:`CPXERR_RANGE_SECTION_ORDER` in the C API."""

CPXERR_RESTRICTED_VERSION = 1016
"""See :macros:`CPXERR_RESTRICTED_VERSION` in the C API."""

CPXERR_RHS_IN_OBJ = 1603
"""See :macros:`CPXERR_RHS_IN_OBJ` in the C API."""

CPXERR_RIM_REPEATS = 1447
"""See :macros:`CPXERR_RIM_REPEATS` in the C API."""

CPXERR_RIM_ROW_REPEATS = 1444
"""See :macros:`CPXERR_RIM_ROW_REPEATS` in the C API."""

CPXERR_RIMNZ_REPEATS = 1479
"""See :macros:`CPXERR_RIMNZ_REPEATS` in the C API."""

CPXERR_ROW_INDEX_RANGE = 1203
"""See :macros:`CPXERR_ROW_INDEX_RANGE` in the C API."""

CPXERR_ROW_REPEAT_PRINT = 1477
"""See :macros:`CPXERR_ROW_REPEAT_PRINT` in the C API."""

CPXERR_ROW_REPEATS = 1445
"""See :macros:`CPXERR_ROW_REPEATS` in the C API."""

CPXERR_ROW_UNKNOWN = 1448
"""See :macros:`CPXERR_ROW_UNKNOWN` in the C API."""

CPXERR_SAV_FILE_DATA = 1561
"""See :macros:`CPXERR_SAV_FILE_DATA` in the C API."""

CPXERR_SAV_FILE_VALUE = 1564
"""See :macros:`CPXERR_SAV_FILE_VALUE` in the C API."""

CPXERR_SAV_FILE_WRITE = 1562
"""See :macros:`CPXERR_SAV_FILE_WRITE` in the C API."""

CPXERR_SBASE_ILLEGAL = 1554
"""See :macros:`CPXERR_SBASE_ILLEGAL` in the C API."""

CPXERR_SBASE_INCOMPAT = 1255
"""See :macros:`CPXERR_SBASE_INCOMPAT` in the C API."""

CPXERR_SINGULAR = 1256
"""See :macros:`CPXERR_SINGULAR` in the C API."""

CPXERR_STR_PARAM_TOO_LONG = 1026
"""See :macros:`CPXERR_STR_PARAM_TOO_LONG` in the C API."""

CPXERR_SUBPROB_SOLVE = 3019
"""See :macros:`CPXERR_SUBPROB_SOLVE` in the C API."""

CPXERR_SYNCPRIM_CREATE = 1809
"""See :macros:`CPXERR_SYNCPRIM_CREATE` in the C API."""

CPXERR_SYSCALL = 1813
"""See :macros:`CPXERR_SYSCALL` in the C API."""

CPXERR_THREAD_FAILED = 1234
"""See :macros:`CPXERR_THREAD_FAILED` in the C API."""

CPXERR_TILIM_CONDITION_NO = 1268
"""See :macros:`CPXERR_TILIM_CONDITION_NO` in the C API."""

CPXERR_TILIM_STRONGBRANCH = 1266
"""See :macros:`CPXERR_TILIM_STRONGBRANCH` in the C API."""

CPXERR_TOO_MANY_COEFFS = 1433
"""See :macros:`CPXERR_TOO_MANY_COEFFS` in the C API."""

CPXERR_TOO_MANY_COLS = 1432
"""See :macros:`CPXERR_TOO_MANY_COLS` in the C API."""

CPXERR_TOO_MANY_RIMNZ = 1485
"""See :macros:`CPXERR_TOO_MANY_RIMNZ` in the C API."""

CPXERR_TOO_MANY_RIMS = 1484
"""See :macros:`CPXERR_TOO_MANY_RIMS` in the C API."""

CPXERR_TOO_MANY_ROWS = 1431
"""See :macros:`CPXERR_TOO_MANY_ROWS` in the C API."""

CPXERR_TOO_MANY_THREADS = 1020
"""See :macros:`CPXERR_TOO_MANY_THREADS` in the C API."""

CPXERR_TREE_MEMORY_LIMIT = 3413
"""See :macros:`CPXERR_TREE_MEMORY_LIMIT` in the C API."""

CPXERR_TUNE_MIXED = 1730
"""See :macros:`CPXERR_TUNE_MIXED` in the C API."""

CPXERR_UNIQUE_WEIGHTS = 3010
"""See :macros:`CPXERR_UNIQUE_WEIGHTS` in the C API."""

CPXERR_UNSUPPORTED_CONSTRAINT_TYPE = 1212
"""See :macros:`CPXERR_UNSUPPORTED_CONSTRAINT_TYPE` in the C API."""

CPXERR_UNSUPPORTED_OPERATION = 1811
"""See :macros:`CPXERR_UNSUPPORTED_OPERATION` in the C API."""

CPXERR_UP_BOUND_REPEATS = 1458
"""See :macros:`CPXERR_UP_BOUND_REPEATS` in the C API."""

CPXERR_WORK_FILE_OPEN = 1801
"""See :macros:`CPXERR_WORK_FILE_OPEN` in the C API."""

CPXERR_WORK_FILE_READ = 1802
"""See :macros:`CPXERR_WORK_FILE_READ` in the C API."""

CPXERR_WORK_FILE_WRITE = 1803
"""See :macros:`CPXERR_WORK_FILE_WRITE` in the C API."""

CPXERR_XMLPARSE = 1425
"""See :macros:`CPXERR_XMLPARSE` in the C API."""
