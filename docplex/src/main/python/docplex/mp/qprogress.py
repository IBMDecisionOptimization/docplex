# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2022
# --------------------------------------------------------------------------

# gendoc: ignore

from collections import namedtuple

from docplex.mp.progress import _AbstractProgressListener

_TQProgressData_ = namedtuple('_TQProgressData',
                              ['id', 'current_nb_iterations',
                               'primal_objective_value', 'dual_objective_value',
                               'primal_infeasibility', 'dual_infeasibility',
                               'time', 'det_time'])


# noinspection PyUnresolvedReferences
class QProgressData(_TQProgressData_):
    """ A named tuple class to hold progress data for quadratic problems.

    Attributes:
        current_objective: contains the current objective, if an incumbent is available, else None.
        current_nb_iterations: the current number of iterations.
        time: the elapsed time since solve started.
        det_time: the deterministic time since solve started.
    """

    def __str__(self):  # pragma: no cover
        fmt = 'QProgressData({0}, {1}, primal={2}, dual={3})'. \
            format(self.id, self.current_nb_iterations, self.primal_objective_value, self.dual_objective_value)
        return fmt


class QProgressListener(_AbstractProgressListener):
    '''  The base class for progress listeners.
    '''

    def __init__(self):
        super().__init__()

    @property
    def current_progress_data(self):
        return self._current_progress_data

    def notify_progress(self, qprogress_data):
        pass


class QTextProgressListener(QProgressListener):

    def __init__(self, obj_fmt=None, inf_fmt=None):
        super().__init__()
        self._obj_fmt = obj_fmt or "{0:.3f}"
        self._inf_fmt = inf_fmt or "{0:.5g}"
        self._count = 0

    def notify_start(self):
        super().notify_start()
        self._count = 0

    def notify_progress(self, progress_data):
        self._count += 1

        # noinspection PyPep8
        primal_obj = progress_data.primal_objective_value
        dual_obj   = progress_data.dual_objective_value
        primal_inf = progress_data.primal_infeasibility
        dual_inf   = progress_data.dual_infeasibility
        s_primal_obj = self._obj_fmt.format(primal_obj)
        s_dual_obj = self._obj_fmt.format(dual_obj)

        s_primal_inf = self._inf_fmt.format(primal_inf)
        s_dual_inf = self._inf_fmt.format(dual_inf)

        raw_time = progress_data.time

        print("{0:>3}: Primal={1} Dual={2} Primal inf={3} Dual inf.={4} ItCnt={5} [{6:.2f}s]"
              .format(self._count, s_primal_obj, s_dual_obj, s_primal_inf, s_dual_inf, progress_data.current_nb_iterations, raw_time))
