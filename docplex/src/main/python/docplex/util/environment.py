# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2022
# --------------------------------------------------------------------------

'''
Representation of the DOcplex solving environment.

This module handles the various elements that allow an
optimization program to run independently from the solving environment.
This environment may be:

 * on premise, using a local version of CPLEX Optimization Studio to solve MP problems, or
 * on Watson Machine Learning, with the Python program running inside the Python Worker.
 * on Decision Optimization in Watson Machine Learning.

As much as possible, the adaptation to the solving environment is
automatic. The functions that are presented here are useful for handling
very specific use cases.

The following code is a program that sums its input (``sum.py``)::

    import json
    import docplex.util.environment as environment

    sum = 0
    # open program input named "data.txt" and sum the contents
    with environment.get_input_stream("data.txt") as input:
        for i in input.read().split():
            sum += int(i)
    # write the result as a simple json in program output "solution.json"
    with environment.get_output_stream("solution.json") as output:
        output.write(json.dumps({'result': sum}))

Let's put some data in a ``data.txt`` file::

    4 7 8
    19

When you run ``sum.py`` with a Python interpreter, it opens the ``data.txt`` file and sums all of the integers
in that file. The result is saved as a JSON fragment in file ``solution.json``::

    $ python sum.py
    $ more solution.json
    {"result": 38}

Environment representation can be accessed with different ways:

    * direct object method calls, after retrieving an instance using
      :meth:`docplex.util.environment.get_environment` and using methods of
      :class:`docplex.util.environment.Environment`.
    * using the function in package `docplex.util.environment`. They will call
       the corresponding methods of Environment in the platform
       `default_environment`:

           * :meth:`docplex.util.environment.get_input_stream`
           * :meth:`docplex.util.environment.get_output_stream`
           * :meth:`docplex.util.environment.read_df`
           * :meth:`docplex.util.environment.write_df`
           * :meth:`docplex.util.environment.get_available_core_count`
           * :meth:`docplex.util.environment.get_parameter`
           * :meth:`docplex.util.environment.update_solve_details`
           * :meth:`docplex.util.environment.add_abort_callback`
           * :meth:`docplex.util.environment.remove_abort_callback`

'''
from collections import deque
import json
import logging
import os
import shutil
import sys
import threading
import time
from typing import Dict
import warnings



try:
    from string import maketrans, translate
except ImportError:
    maketrans = str.maketrans
    translate = str.translate

try:
    import pandas
except ImportError:
    pandas = None

from six import iteritems

from docplex.util import lazy



def make_new_kpis_dict(allkpis=None, sense = None, model_type = None, int_vars=None, continuous_vars=None,
                       linear_constraints=None, bin_vars=None,
                       semicontinuous_vars = None, semiinteger_vars = None,
                       quadratic_constraints=None, total_constraints=None,
                       total_variables=None):
    # This is normally called once at the beginning of a solve
    # those are the details required for docplexcloud and DODS legacy
    kpis_name= [ kpi.name for kpi in allkpis ]
    kpis = {'MODEL_DETAIL_OBJECTIVE_SENSE' : sense,
            'MODEL_DETAIL_INTEGER_VARS': int_vars,
            'MODEL_DETAIL_CONTINUOUS_VARS': continuous_vars,
            'MODEL_DETAIL_CONSTRAINTS': linear_constraints,
            'MODEL_DETAIL_BOOLEAN_VARS': bin_vars,
            'MODEL_DETAIL_KPIS': json.dumps(kpis_name)}
    # those are the ones required per https://github.ibm.com/IBMDecisionOptimization/dd-planning/issues/2491
    new_details = {'STAT.cplex.modelType' : model_type,
                   'STAT.cplex.size.integerVariables': int_vars,
                   'STAT.cplex.size.semicontinuousVariables': semicontinuous_vars,
                   'STAT.cplex.size.semiintegerVariables': semiinteger_vars,
                   'STAT.cplex.size.continousVariables': continuous_vars,
                   'STAT.cplex.size.linearConstraints': linear_constraints,
                   'STAT.cplex.size.booleanVariables': bin_vars,
                   'STAT.cplex.size.constraints': total_constraints,
                   'STAT.cplex.size.quadraticConstraints': quadratic_constraints,
                   'STAT.cplex.size.variables': total_variables,
                   }

    kpis.update(new_details)
    return kpis


log_level_mapping = {'OFF': None,
                     'SEVERE': logging.ERROR,
                     'WARNING': logging.WARNING,
                     'INFO': logging.INFO,
                     'CONFIG': logging.INFO,
                     'FINE': logging.DEBUG,
                     'FINER': logging.DEBUG,
                     'FINEST': logging.DEBUG,
                     'ALL': logging.DEBUG}


class NotAvailableError(Exception):
    ''' The exception raised when a feature is not available
    '''
    pass


def default_solution_storage_handler(env, solution):
    ''' The default solution storage handler.

    The storage handler is a function which first argument is the
    :class:`~Environment` on which a solution should be saved. The `solution`
    is a dict containing all the data for an optimization solution.

    The storage handler is responsible for storing the solution in the
    environment.

    For each (key, value) pairs of the solution, the default solution storage
    handler does the following depending of the type of `value`, in the
    following order:

        * If `value` is a `pandas.DataFrame`, then the data frame is saved
          as an output with the specified `name`. Note that `name` must include
          an extension file for the serialization. See
          :meth:`Environment.write_df` for supported formats.
        * If `value` is a `bytes`, it is saved as binary data with `name`.
        * The `value` is saved as an output with the `name`, after it has been
          converted to JSON.

    Args:
        env: The :class:`~Environment`
        solution: a dict containing the solution.
    '''
    for (name, value) in iteritems(solution):
        if pandas and isinstance(value, pandas.DataFrame):
            _, ext = os.path.splitext(name)
            if ext.lower() == '':
                name = '%s.csv' % name  # defaults to csv if no format specified
            env.write_df(value, name)
        elif isinstance(value, bytes):
            with env.get_output_stream(name) as fp:
                fp.write(value)
        else:
            # try jsonify
            with env.get_output_stream(name) as fp:
                json.dump(value, fp)


# The global output lock
global_output_lock = threading.Lock()


class SolveDetailsFilter(object):
    '''Default solve detail filter class.

    This default class filters details so that there are no more than 1 solve
    details per second.
    '''
    def __init__(self, interval=1):
        self.last_accept_time = 0
        self.interval = interval

    def filter(self, details):
        '''Filters the details.

        Returns:
            True if the details are to be published.
        '''
        ret_val = None
        now = time.time()
        if (now - self.last_accept_time > self.interval):
            ret_val = details
            self.last_accept_time = now
        return ret_val


class OutputAttachmentTransaction(object):
    """Handles output attachment transactions

    *New in version 2.26*
    """
    def __init__(self, env):
        self.env = env
        self.attachments = dict()

    def set_output_attachments(self, attachments: Dict[str, str]):
        """
        Adds new attachments to this transaction
        """
        self.attachments.update(attachments)

    def commit(self):
        """
        Commit the attachments waiting to be publushed.
        :return:
        """
        if self.attachments:
            self.env.set_output_attachments(self.attachments)
            self.attachments.clear()

    def close(self):
        """
        Closes the transaction, actually setting all the attachments at once.
        """
        self.commit()

class Environment(object):
    ''' Methods for interacting with the execution environment.

    Internally, the ``docplex`` package provides the appropriate implementation
    according to the actual execution environment.
    The correct instance of this class is returned by the method
    :meth:`docplex.util.environment.get_environment` that is provided in this
    module.

    Attributes:
        abort_callbacks: A list of callbacks that are called when the script is
            run on Watson Machine Learning and a job abort operation is requested. You
            add your own callback using::

                env.abort_callbacks += [your_cb]

            or::

                env.abort_callbacks.append(your_cb)

            You remove a callback using::

                env.abort_callbacks.remove(your_cb)

        solution_storage_handler: A function called when a solution is to be
            stored. The storage handler is a function which first argument is
            the :class:`~Environment` on which a solution should be saved. The
            `solution` is a dict containing all the data for an optimization
            solution. The default is :meth:`~default_solution_storage_handler`.
        record_history_fields: Fields which history is to be kept
        record_history_size: maximum number of records in history
        record_interval: min time between to history records
    '''
    def __init__(self):
        self.output_lock = global_output_lock
        self.solution_storage_handler = default_solution_storage_handler
        self.abort_callbacks = []
        self.interpret_df_type = True
        self.update_solve_details_dict = True
        self.last_solve_details = {}  # stores the latest published details
        # private behaviour for now: allows to filter details
        # the SolveDetailsFilter.filter() method returns true if the details
        # are to be kept
        self.details_filter = None
        self.unpublished_details = None
        self._record_history_fields = None
        # self.record_history_fields = ['PROGRESS_CURRENT_OBJECTIVE']
        self.record_history = {}  # maps name -> deque
        self.last_history_record = {}  # we keep the last here so that we can publish at end of solve
        self.record_history_time_decimals = 2  # number of decimals for time
        self.record_history_size = 100
        self.record_min_time = 1
        self.recorded_solve_details_count = 0  # number of solve details that have been sent to recording
        self.autoreset = True
        self.logger = logging.getLogger("docplex.util.environment.logger")

    def create_transaction(self):
        """
        Creates a new output attachment transaction
        :return: an OutputAttachmentTransaction

        *New in version 2.26*
        """
        return OutputAttachmentTransaction(self)


    def _reset_record_history(self, force=False):
        if self.autoreset or force:
            self.record_history = {}
            self.unpublished_details = None
            self.last_history_record = {}
            self.recorded_solve_details_count = 0

    def get_record_history_fields(self):
        if self._record_history_fields is None:
            if self.is_wmlworker:
                self._record_history_fields = ['PROGRESS_BEST_OBJECTIVE',
                                               'PROGRESS_CURRENT_OBJECTIVE',
                                               'PROGRESS_GAP']
            else:
                # the default out of dods is to not record any history
                self._record_history_fields = []
        return self._record_history_fields

    def set_record_history_fields(self, value):
        self._record_history_fields = value

    # let record_history_fields be a property that is lazy initialized
    # this gives the opportunity to set is_wmlworker before record history fields are needed
    record_history_fields = property(get_record_history_fields, set_record_history_fields)

    def store_solution(self, solution):
        '''Stores the specified solution.

        This method guarantees that the solution is fully saved if the model
        is running on Watson Machine Learning python worker and an abort of the job is
        triggered.

        For each (key, value) pairs of the solution, the default solution
        storage handler does the following depending of the type of `value`, in
        the following order:

        * If `value` is a `pandas.DataFrame`, then the data frame is saved
          as an output with the specified `name`. Note that `name` must include
          an extension file for the serialization. See
          :meth:`Environment.write_df` for supported formats.
        * If `value` is a `bytes`, it is saved as binary data with `name`.
        * The `value` is saved as an output with the `name`, after it has been
          converted to JSON.

        Args:
            solution: a dict containing the solution.
        '''
        with self.output_lock:
            self.solution_storage_handler(self, solution)

    def get_input_stream(self, name):
        ''' Get an input of the program as a stream (file-like object).

        An input of the program is a file that is available in the working directory.

        When run on Watson Machine Learning, all input attachments are copied to the working directory before
        the program is run. ``get_input_stream`` lets you open the input attachments of the job.

        Args:
            name: Name of the input object.
        Returns:
            A file object to read the input from.
        '''
        self.logger.debug(lazy(lambda: f"set input stream: name={name}"))
        return None

    def read_df(self, name, reader=None, **kwargs):
        ''' Reads an input of the program as a ``pandas.DataFrame``.

        ``pandas`` must be installed.

        ``name`` is the name of the input object, as a filename. If a reader
        is not user provided, the reader used depends on the filename extension.

        The default reader used depending on extension are:

            * ``.csv``: ``pandas.read_csv()``
            * ``.msg``: ``pandas.read_msgpack()``

        Args:
            name: The name of the input object
            reader: an optional reader function
            **kwargs: additional parameters passed to the reader
        Raises:
            NotAvailableError: raises this error when ``pandas`` is not
                available.
        '''
        if pandas is None:
            raise NotAvailableError('read_df() is only available if pandas is installed')
        _, ext = os.path.splitext(name)
        default_kwargs = None
        if reader is None:
            default_readers = {'.csv': (pandas.read_csv, {'index_col': 0}),
                               '.msg': (pandas.read_msgpack, None)}
            reader, default_kwargs = default_readers.get(ext.lower(), None)
        if reader is None:
            raise ValueError('no default reader defined for files with extension: \'%s\'' % ext)
        with self.get_input_stream(name) as ost:
            # allow
            params = {}
            if default_kwargs:
                params.update(default_kwargs)
            if kwargs:
                params.update(kwargs)
            return reader(ost, **params)

    def write_df(self, df, name, writer=None, **kwargs):
        ''' Write a ``pandas.DataFrame`` as an output of the program.

        ``pandas`` must be installed.

        ``name`` is the name of the input object, as a filename. If a writer
        is not user provided, the writer used depends on the filename extension.

        This currently only supports csv output.

        Args:
            name: The name of the input object
            writer: an optional writer function
            **kwargs: additional parameters passed to the writer
        Raises:
            NotAvailableError: raises this error when ``pandas`` is not
                available.
        '''
        if pandas is None:
            raise NotAvailableError('write_df() is only available if pandas is installed')
        _, ext = os.path.splitext(name)
        if writer is None:
            try:
                default_writers = {'.csv': df.to_csv}
                writer = default_writers.get(ext.lower(), None)
            except AttributeError:
                raise NotAvailableError('Could not write writer function for extension: %s' % ext)
        if writer is None:
            raise ValueError('no default writer defined for files with extension: \'%s\'' % ext)
        with self.get_output_stream(name) as ost:
            if sys.version_info[0] < 3:
                ost.write(writer(index=False, encoding='utf8'))
            else:
                ost.write(writer(index=False).encode(encoding='utf8'))

    def set_output_attachment(self, name, filename):
        '''Attach the file which filename is specified as an output of the
        program.

        The file is recorded as being part of the program output.
        This method can be called multiple times if the program contains
        multiple output objects.

        When run on premise, ``filename`` is copied to the the working
        directory (if not already there) under the name ``name``.

        When run on Watson Machine Learning, the file is attached as output attachment.

        Args:
            name: Name of the output object.
            filename: The name of the file to attach.
        '''
        self.logger.debug(lazy(lambda: f"set output attachment: name={name}, filename={filename}"))
        self.set_output_attachments({name: filename})

    def set_output_attachments(self, attachments):
        """Sets the output attachments.

        Attachments are recorded as being part of the program output.

        Each attachment is a pair of `name`,`filename` entries.

        When run on premise, ``filename`` is copied to the the working
        directory (if not already there) under the name ``name``.

        When run on Watson Machine Learning, attachments are set as output.

        Args:
            attachments: a dict with pairs of `name`,`filename` association.

        *New in version 2.26*
        """
        self.logger.debug(lazy(lambda: f"set output attachment: attachments={attachments}"))

    def get_output_stream(self, name, transaction = None):
        ''' Get a file-like object to write the output of the program.

        The file is recorded as being part of the program output.
        This method can be called multiple times if the program contains
        multiple output objects.

        When run on premise, the output of the program is written as files in
        the working directory. When run on Watson Machine Learning, the files are attached
        as output attachments.

        The stream is opened in binary mode, and will accept 8 bits data.

        Args:
            name: Name of the output object.
            transaction (optional): The transaction handler. Can be ignored if the environment does not support
                transactions.
        Returns:
            A file object to write the output to.
        '''
        self.logger.debug(lazy(lambda: f"set output stream: name={name}"))
        return open(os.devnull, "w+b")

    def get_available_core_count(self):
        ''' Returns the number of cores available for processing if the environment
        sets a limit.

        This number is used in the solving engine as the number of threads.

        Returns:
            The available number of cores or ``None`` if the environment does not
            limit the number of cores.
        '''
        return None

    def get_parameters(self):
        ''' Returns a dict containing all parameters of the program.

        on Watson Machine Learning, this method returns the job parameters.
        On local solver, this method returns ``os.environ``.

        Returns:
            The job parameters
        '''
        return None

    def get_parameter(self, name):
        ''' Returns a parameter of the program.

        on Watson Machine Learning, this method returns the job parameter whose name is specified.
        On local solver, this method returns the environment variable whose name is specified.

        Args:
            name: The name of the parameter.
        Returns:
            The parameter whose name is specified or None if the parameter does
            not exists.
        '''
        return None

    def notify_start_solve(self, solve_details, engine_type=None):
        # ===============================================================================
        #         '''Notify the solving environment that a solve is starting.
        #
        #         If ``context.solver.auto_publish.solve_details`` is set, the underlying solver will automatically
        #         send details. If you want to craft and send your own solve details, you can use the following
        #         keys (non exhaustive list):
        #
        #             - MODEL_DETAIL_TYPE : Model type
        #             - MODEL_DETAIL_CONTINUOUS_VARS : Number of continuous variables
        #             - MODEL_DETAIL_INTEGER_VARS : Number of integer variables
        #             - MODEL_DETAIL_BOOLEAN_VARS : Number of boolean variables
        #             - MODEL_DETAIL_INTERVAL_VARS : Number of interval variables
        #             - MODEL_DETAIL_SEQUENCE_VARS : Number of sequence variables
        #             - MODEL_DETAIL_NON_ZEROS : Number of non zero variables
        #             - MODEL_DETAIL_CONSTRAINTS : Number of constraints
        #             - MODEL_DETAIL_LINEAR_CONSTRAINTS : Number of linear constraints
        #             - MODEL_DETAIL_QUADRATIC_CONSTRAINTS : Number of quadratic constraints
        #
        #         Args:
        #             solve_details: A ``dict`` with solve details as key/value pairs
        #         See:
        #             :attr:`.Context.solver.auto_publish.solve_details`
        #         '''
        # ===============================================================================
        self.logger.debug(lazy(lambda: f"Notify start solve: engine_type={engine_type}, solve_details={json.dumps(solve_details, indent=3)}"))
        self._reset_record_history()

    def update_solve_details(self, details, transaction=None):
        """Update the solve details.

        You use this method to send solve details to the solve service.
        If ``context.solver.auto_publish`` is set, the underlying
        solver will automatically update solve details once the solve has
        finished.

        This method might filter details and publish them with rate limitations.
        It actually publish de details by calling `publish_solve_details`.

        Args:
            details: A ``dict`` with solve details as key/value pairs.
            transaction: The transaction to publish output attachments
        """
        self.logger.debug(lazy(lambda: f"Update solve details: {json.dumps(details, indent=3)}"))
        # publish details
        to_publish = None
        if self.update_solve_details_dict:
            previous = self.last_solve_details
            to_publish = {}
            if details:
                to_publish.update(previous)
                to_publish.update(details)
            self.last_solve_details = to_publish
        else:
            to_publish = details
        # process history
        to_publish = self.record_in_history(to_publish)

        if self.details_filter:
            if self.details_filter.filter(details):
                self.logger.debug("Published as filtered details")
                self.publish_solve_details(to_publish, transaction=transaction)
            else:
                # just store the details for later use
                self.logger.debug("Publish filter refused details, stored as unpublished")
                self.unpublished_details = to_publish
        else:
            self.logger.debug("Published as unfiltered details")
            self.publish_solve_details(to_publish, transaction=transaction)

    def record_in_history(self, details):
        self.recorded_solve_details_count += 1
        self.logger.debug(lazy(lambda: f"record in history: {json.dumps(details)}"))
        current_ts = round(time.time(), self.record_history_time_decimals)
        for f in self.record_history_fields:
            if f in details:
                current_history_element = [current_ts, details[f]]
                l = self.record_history.get(f, deque([], self.record_history_size))
                self.record_history[f] = l
                last_ts = l[-1][0] if len(l) >= 1 else -9999
                if (current_ts - last_ts) >= self.record_min_time:
                    self.logger.debug(lazy(lambda: f"record added in history for field {f}"))
                    l.append(current_history_element)
                    details['%s.history' % f] = json.dumps(list(l))  # make new copy
                    # make current also last history record
                    self.last_history_record[f] = current_history_element
                else:
                    self.logger.debug(lazy(lambda: f"record stored as current for field {f}"))
                    self.last_history_record[f] = current_history_element
        return details

    def prepare_last_history(self):
        details = {}
        details.update(self.last_solve_details)
        any_added = False
        for k, v in self.last_history_record.items():
            the_list = self.record_history[k]
            do_append = True
            if len(the_list) >= 1:
                last_date_history = the_list[-1][0]
                last_date = v[0]
                do_append = (abs(last_date - last_date_history) >= 0.01)
            if do_append:
                the_list.append(v)
                any_added = True
            details['%s.history' % k] = json.dumps(list(the_list))
        return details if any_added else False

    def publish_solve_details(self, details, transaction=None):
        """Actually publish the solve specified details.

        Returns:
            The published details
        """
        self.logger.debug(lazy(lambda: f"Publish solve details: {json.dumps(details, indent=3)}"))

    def notify_end_solve(self, status, solve_time=None, transaction=None):
        # ===============================================================================
        #         '''Notify the solving environment that the solve as ended.
        #
        #         The ``status`` can be a JobSolveStatus enum or an integer.
        #
        #         When ``status`` is an integer, it is converted with the following conversion table:
        #
        #             0 - UNKNOWN: The algorithm has no information about the solution.
        #             1 - FEASIBLE_SOLUTION: The algorithm found a feasible solution.
        #             2 - OPTIMAL_SOLUTION: The algorithm found an optimal solution.
        #             3 - INFEASIBLE_SOLUTION: The algorithm proved that the model is infeasible.
        #             4 - UNBOUNDED_SOLUTION: The algorithm proved the model unbounded.
        #             5 - INFEASIBLE_OR_UNBOUNDED_SOLUTION: The model is infeasible or unbounded.
        #
        #         Args:
        #             status: The solve status
        #             solve_time: The solve time
        #         '''
        # ===============================================================================
        self.logger.debug(f"Notify end solve, status={status}, solve_time={solve_time}")
        if self.unpublished_details:
            self.logger.debug("Notify end solve: has unpublished details, so publish them")
            self.publish_solve_details(self.unpublished_details, transaction=transaction)
        if self.recorded_solve_details_count >= 1 and self.last_history_record:
            self.logger.debug("Notify end solve: has more than 1 solve details, prepare and publish history")
            last_details = self.prepare_last_history()
            if last_details:
                self.publish_solve_details(last_details, transaction=transaction)

    def set_stop_callback(self, cb):
        '''Sets a callback that is called when the script is run on
        DOcplexcloud and a job abort operation is requested.

        You can also use the ``stop_callback`` property to set the callback.

        Deprecated since 2.4 - Use self.abort_callbacks += [cb] instead'

        Args:
            cb: The callback function
        '''
        warnings.warn('set_stop_callback() is deprecated since 2.4 - Use Environment.abort_callbacks.append(cb) instead')

    def get_stop_callback(self):
        '''Returns the stop callback that is called when the script is run on
        DOcplexcloud and a job abort operation is requested.

        You can also use the ``stop_callback`` property to get the callback.

        Deprecated since 2.4 - Use the abort_callbacks property instead')

        '''
        warnings.warn('get_stop_callback() is deprecated since 2.4 - Use the abort_callbacks property instead')
        return None

    stop_callback = property(get_stop_callback, set_stop_callback)

    def get_engine_log_level(self):
        '''Returns the engine log level as set by job parameter oaas.engineLogLevel.

        oaas.engineLogLevel values are: OFF, SEVERE, WARNING, INFO, CONFIG, FINE, FINER, FINEST, ALL

        The mapping to logging levels in python are:

           * OFF: None
           * SEVERE: logging.ERROR
           * WARNING: logging.WARNING
           * INFO, CONFIG: logging.INFO
           * FINE, FINER, FINEST: logging.DEBUG
           * ALL: logging.DEBUG


        All other values are considered invalid values and will return None.

        Returns:
            The logging level or None if not set (off)
        '''
        oaas_level = self.get_parameter('oaas.engineLogLevel')
        log_level = log_level_mapping.get(oaas_level.upper(), None) if oaas_level else None
        return log_level

    def is_debug_mode(self):
        '''Returns true if the engine should run in debug mode.

        This is equivalent to ``env.get_engine_log_level() <= logging.DEBUG``
        '''
        lvl = self.get_engine_log_level()
        # logging.NOTSET is zero so will return false
        return (self.get_engine_log_level() <= logging.DEBUG) if lvl is not None else False



    @property
    def is_local(self):
        return False
    @property
    def is_wmlworker(self):
        return False
    @property
    def is_wsnotebook(self):
        return False


class AbstractLocalEnvironment(Environment):
    # The environment solving environment using all local input and outputs.
    def __init__(self):
        super(AbstractLocalEnvironment, self).__init__()
        self.logger = logging.getLogger('docplex.util.environment.logger')

        # init number of cores. Default is no limits (engines will use
        # number of cores reported by system).
        # On Watson studio runtimes, the system reports the total number
        # of physical cores but not the number of cores available to the
        # runtime. The number of cores available to the runtime are
        # specified in an environment variable instead.
        self._available_cores = None
        #TODO VB: Is it useful? We have a priori call to self.solve_hook.get_available_core_count() in WorkerEnv?
        RUNTIME_HARDWARE_SPEC = os.environ.get('RUNTIME_HARDWARE_SPEC', None)
        if RUNTIME_HARDWARE_SPEC:
            try:
                spec = json.loads(RUNTIME_HARDWARE_SPEC)
                num = int(spec.get('num_cpu')) if ('num_cpu' in spec) else None
                self._available_cores = num
            except:
                pass

    def get_input_stream(self, name):
        return open(name, "rb")

    def get_output_stream(self, name, transaction = None):
        return open(name, "wb")

    def get_parameter(self, name):
        return os.environ.get(name, None)

    def get_parameters(self):
        return os.environ

    def set_output_attachment(self, name, filename):
        self.set_output_attachments({name: filename})

    def set_output_attachments(self, attachments):
        for name, filename in attachments.items():
            # check that name leads to a file in cwd
            attachment_abs_path = os.path.dirname(os.path.abspath(name))
            if attachment_abs_path != os.getcwd():
                raise ValueError(f'Illegal attachment name: {name}')

            if os.path.dirname(os.path.abspath(filename)) != os.getcwd():
                shutil.copyfile(filename, name)  # copy to current

    def get_available_core_count(self):
        return self._available_cores


class LocalEnvironment(AbstractLocalEnvironment):
    def __init__(self):
        super(LocalEnvironment, self).__init__()
    @property
    def is_local(self):
        return True

class OverrideEnvironment(object):
    '''Allows to temporarily replace the default environment.

    If the override environment is None, nothing happens and the default
    environment is not replaced
    '''
    def __init__(self, new_env=None):
        self.set_env = new_env
        self.saved_env = None

    def __enter__(self):
        if self.set_env:
            global default_environment
            self.saved_env = default_environment
            default_environment = self.set_env
        else:
            self.saved_env = None

    def __exit__(self, type, value, traceback):
        if self.saved_env:
            global default_environment
            default_environment = self.saved_env


def _get_default_environment():
    try:
        try:
            from docplex_wml.util import _get_wml_default_environment
            return _get_wml_default_environment()
        except ImportError:
            pass
        return LocalEnvironment()
    except Exception as e:
        print("We should never get here")
        print(e)

default_environment = _get_default_environment()


def _get_cplex_edition():
    import docplex.mp.model
    import docplex.mp.environment
    version = docplex.mp.environment.Environment().cplex_version
    if default_environment.is_wmlworker:
        return "%s%s" % (version, "")
    with OverrideEnvironment(Environment()):
        edition = " ce" if docplex.mp.model.Model.is_cplex_ce() else ""
        return "%s%s" % (version, edition)


def get_environment():
    ''' Returns the Environment object that represents the actual execution
    environment.

    Note: the default environment is the value of the
    ``docplex.util.environment.default_environment`` property.

    Returns:
        An instance of the :class:`.Environment` class that implements methods
        corresponding to actual execution environment.
    '''
    return default_environment


def get_input_stream(name):
    ''' Get an input of the program as a stream (file-like object),
    with the default environment.

    An input of the program is a file that is available in the working directory.

    When run on Watson Machine Learning, all input attachments are copied to the working directory before
    the program is run. ``get_input_stream`` lets you open the input attachments of the job.

    Args:
        name: Name of the input object.
    Returns:
        A file object to read the input from.
    '''
    return default_environment.get_input_stream(name)


def set_output_attachment(name, filename):
    ''' Attach the file which filename is specified as an output of the
    program.

    The file is recorded as being part of the program output.
    This method can be called multiple times if the program contains
    multiple output objects.

    When run on premise, ``filename`` is copied to the the working
    directory (if not already there) under the name ``name``.

    When run on Watson Machine Learning, the file is attached as output attachment.

    Args:
        name: Name of the output object.
        filename: The name of the file to attach.
    '''
    return set_output_attachments({name, filename})


def set_output_attachments(attachments):
    """Sets the specified attachments

    The attachments are recorded as being part of the program output.

    The `attachments` are a dict of `name` to `filename` mapping.

    When run on premise, filenames are copied to the working
    directory (if not already there) under the names.

    When run on Watson Machine Learning, the files are attached as output attachment.

    Args:
        attachments: a dict of `name`:`filename` mapping

    *New in version 2.26*
    """
    return default_environment.set_output_attachments(attachments)


def get_output_stream(name, transaction = None):
    ''' Get a file-like object to write the output of the program.

    The file is recorded as being part of the program output.
    This method can be called multiple times if the program contains
    multiple output objects.

    When run on premise, the output of the program is written as files in
    the working directory. When run on Watson Machine Learning, the files are attached
    as output attachments.

    The stream is opened in binary mode, and will accept 8 bits data.

    Args:
        name: Name of the output object.
        transaction (optional): The transaction handler. Can be ignored if the environment does not support
           transactions.
    Returns:
        A file object to write the output to.
    '''
    return default_environment.get_output_stream(name)


def read_df(name, reader=None, **kwargs):
    ''' Reads an input of the program as a ``pandas.DataFrame`` with the
    default environment.

    ``pandas`` must be installed.

    ``name`` is the name of the input object, as a filename. If a reader
    is not user provided, the reader used depends on the filename extension.

    The default reader used depending on extension are:

        * ``.csv``: ``pandas.read_csv()``
        * ``.msg``: ``pandas.read_msgpack()``

    Args:
        name: The name of the input object
        reader: an optional reader function
        **kwargs: additional parameters passed to the reader
    Raises:
        NotAvailableError: raises this error when ``pandas`` is not
            available.
    '''
    return default_environment.read_df(name, reader=reader, **kwargs)


def write_df(df, name, writer=None, **kwargs):
    ''' Write a ``pandas.DataFrame`` as an output of the program with the
    default environment.

    ``pandas`` must be installed.

    ``name`` is the name of the input object, as a filename. If a writer
    is not user provided, the writer used depends on the filename extension.

    The default writer used depending on extension are:

        * ``.csv``: ``DataFrame.to_csv()``
        * ``.msg``: ``DataFrame.to_msgpack()``

    Args:
        name: The name of the input object
        writer: an optional writer function
        **kwargs: additional parameters passed to the writer
    Raises:
        NotAvailableError: raises this error when ``pandas`` is not
            available.
    '''
    return default_environment.write_df(df, name, writer=writer, **kwargs)


def get_available_core_count():
    ''' Returns the number of cores available for processing if the environment
    sets a limit, with the default environment.

    This number is used in the solving engine as the number of threads.

    Returns:
        The available number of cores or ``None`` if the environment does not
        limit the number of cores.
    '''
    return default_environment.get_available_core_count()


def get_parameter(name):
    ''' Returns a parameter of the program, with the default environment.

    on Watson Machine Learning, this method returns the job parameter whose name is specified.

    Args:
        name: The name of the parameter.
    Returns:
        The parameter whose name is specified.
    '''
    return default_environment.get_parameter(name)


def update_solve_details(details):
    '''Update the solve details, with the default environment

    You use this method to send solve details to the DOcplexcloud service.
    If ``context.solver.auto_publish`` is set, the underlying
    solver will automatically update solve details once the solve has
    finished.

    Args:
        details: A ``dict`` with solve details as key/value pairs.
    '''
    return default_environment.update_solve_details(details)


def add_abort_callback(cb):
    '''Adds the specified callback to the default environment.

    The abort callback is called when the script is run on
    DOcplexcloud and a job abort operation is requested.

    Args:
        cb: The abort callback
    '''
    default_environment.abort_callbacks += [cb]


def remove_abort_callback(cb):
    '''Adds the specified callback to the default environment.

    The abort callback is called when the script is run on
    DOcplexcloud and a job abort operation is requested.

    Args:
        cb: The abort callback
    '''
    default_environment.abort_callbacks.remove(cb)


attachment_invalid_characters = '/\\?%*:|"#<> '
attachment_trans_table = maketrans(attachment_invalid_characters, '_' * len(attachment_invalid_characters))


def make_attachment_name(name):
    r'''From `name`, create an attachment name that is correct for DOcplexcloud.

    Attachment filenames in DOcplexcloud has certain restrictions. A file name:

        - is limited to 255 characters;
        - can include only ASCII characters;
        - cannot include the characters `/\\?%*:|"<>`, the space character, or the null character; and
        - cannot include _ as the first character.

    This method replace all unauthorized characters with _, then removing leading
    '_'.

    Args:
        name: The original attachment name
    Returns:
        An attachment name that conforms to the restrictions.
    Raises:
        ValueError if the attachment name is more than 255 characters
    '''
    new_name = translate(name, attachment_trans_table)
    while (new_name.startswith('_')):
        new_name = new_name[1:]
    if len(new_name) > 255:
        raise ValueError('Attachment names are limited to 255 characters')
    return new_name
