# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2018
# --------------------------------------------------------------------------

# gendoc: ignore

# 'findspark' must be executed if running in Windows environment, before importing Spark
try:
    import findspark  # @UnresolvedImport
    findspark.init()
except (ImportError, IndexError, ValueError):
    pass


def make_solution(col_values, col_names, keep_zeros):
    # Return values as-is if not None, converting all values to float
    if col_values:
        return list(map(float, col_values))
    else:
        return []


__spark_dataframe_type = 0

def is_spark_dataframe(s):
    global __spark_dataframe_type
    if __spark_dataframe_type == 0:
        try:
            import pyspark

            __spark_dataframe_type = pyspark.sql.dataframe.DataFrame
        except ImportError:
            __spark_dataframe_type = None
    ok = __spark_dataframe_type and isinstance(s, __spark_dataframe_type)
    return bool(ok)
