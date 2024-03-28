# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2019
# --------------------------------------------------------------------------

from docplex.util.environment import get_environment

try:
    import pandas as pd
except ImportError:
    pd = None

from docplex.util.csv_utils import write_csv, write_table_as_csv


def get_auto_publish_names(context, prop_name, default_name):
    # comparing auto_publish to boolean values because it can be a non-boolean
    autopubs = context.solver.auto_publish
    if autopubs == None:
        return None
    if autopubs is True:
        return [default_name]
    elif autopubs is False:
        return None
    elif prop_name in autopubs:
        name = autopubs[prop_name]
    else:
        name = None

    if isinstance(name, str):
        # only one string value: make this the name of the table
        # in a list with one object
        name = [name]
    elif name is True:
        # if true, then use default name:
        name = [default_name]
    elif name is False:
        # Need to compare explicitely to False
        name = None
    else:
        # otherwise the kpi_table_name can be a collection-like of names,
        # just return it
        pass
    return name


def auto_publishing_result_output_names(context):
    # Return the list of result output names for saving
    return get_auto_publish_names(context, 'result_output', 'solution.json')


def auto_publishing_kpis_table_names(context):
    # Return the list of kpi table names for saving
    return get_auto_publish_names(context, 'kpis_output', 'kpis.csv')


def get_kpis_name_field(context):
    autopubs = context.solver.auto_publish
    if autopubs is True:
        field = 'Name'
    elif autopubs is False:
        field = None
    else:
        field = context.solver.auto_publish.kpis_output_field_name
    return field


def get_kpis_value_field(context):
    autopubs = context.solver.auto_publish
    if autopubs is True:
        field = 'Value'
    elif autopubs is False:
        field = None
    else:
        field = context.solver.auto_publish.kpis_output_field_value
    return field


class PublishResultAsDf(object):
    '''Mixin for classes publishing a result as data frame
    '''

    @staticmethod
    def value_if_defined(obj, attr_name, default=None):
        value = getattr(obj, attr_name) if hasattr(obj, attr_name) else None
        return value if value is not None else default

    def write_output_table(self, df, context,
                           output_property_name=None,
                           output_name=None):
        '''Publishes the output `df`.

        The `context` is used to control the output name:

            - If context.solver.auto_publish is true, the `df` is written using
              output_name.
            - If context.solver.auto_publish is false, This method does nothing.
            - If context.solver.auto_publish.output_property_name is true,
              then `df` is written using output_name.
            - If context.solver.auto_publish.output_propert_name is None or
              False, this method does nothing.
            - If context.solver.auto_publish.output_propert_name is a string,
              it is used as a name to publish the df

        Example:

            A solver can be defined as publishing a result as data frame::

                class SomeSolver(PublishResultAsDf)
                   def __init__(self, output_customizer):
                      # output something if context.solver.autopublish.somesolver_output is set
                      self.output_table_property_name = 'somesolver_output'
                      # output filename unless specified by somesolver_output:
                      self.default_output_table_name = 'somesolver.csv'
                      # customizer if users wants one
                      self.output_table_customizer = output_customizer
                      # uses pandas.DataFrame if possible, otherwise will use namedtuples
                      self.output_table_using_df = True

                    def solve(self):
                        # do something here and return a result as a df
                        result = pandas.DataFrame(columns=['A','B','C'])
                        return result

            Example usage::

               solver = SomeSolver()
               results = solver.solve()
               solver.write_output_table(results)

        '''

        prop = self.value_if_defined(self, 'output_table_property_name')
        prop = output_property_name if output_property_name else prop
        default_name = self.value_if_defined(self, 'default_output_table_name')
        default_name = output_name if output_name else default_name
        names = get_auto_publish_names(context, prop, default_name)
        use_df = self.value_if_defined(self, 'output_table_using_df', True)
        if names:
            env = get_environment()
            customizer = self.value_if_defined(self, 'output_table_customizer', lambda x: x)
            for name in names:
                r = customizer(df)
                if pd and use_df:
                    env.write_df(r, name)
                else:
                    # assume r is a namedtuple
                    write_csv(env, r, r[0]._fields, name)

    def is_publishing_output_table(self, context):
        prop = self.value_if_defined(self, 'output_table_property_name')
        default_name = self.value_if_defined(self, 'default_output_table_name')
        names = get_auto_publish_names(context, prop, default_name)
        return names


def write_kpis_table(env, context, model, solution, transaction=None):
    names = auto_publishing_kpis_table_names(context)
    kpis_table = []
    for k in model.iter_kpis():
        kpis_table.append([k.name, k.compute(solution)])
    if kpis_table:
        # do not create the kpi tables if there are no kpis to be written
        field_names = [get_kpis_name_field(context),
                       get_kpis_value_field(context)]
        for name in names:
            write_table_as_csv(env, kpis_table, name, field_names, transaction=transaction)


def write_solution(env, solution, name, transaction=None):
    with env.get_output_stream(name, transaction=transaction) as output:
        output.write(solution.export_as_json_string().encode('utf-8'))


def write_result_output(env, context, model, solution, transaction=None):
    names = auto_publishing_result_output_names(context)
    for name in names:
        write_solution(env, solution, name, transaction=transaction)
