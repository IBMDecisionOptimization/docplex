# --------------------------------------------------------------------------
# Source file provided under Apache License, Version 2.0, January 2004,
# http://www.apache.org/licenses/
# (c) Copyright IBM Corp. 2015, 2022
# ----------------------------------

# gendoc: ignore

from sys import stdout

from docplex.mp.utils import is_iterable


class SolutionPrinter(object):

    @classmethod
    def extension(cls):
        raise NotImplementedError

    def print_one_solution(self, solution, out, **kwargs):
        raise NotImplementedError

    def print_many_solutions(self, solutions, out, **kwargs):
        raise NotImplementedError

    def _print_to_stream2(self, out, solutions, **kwargs):
        if not is_iterable(solutions):
            self.print_one_solution(solutions, out, **kwargs)
        else:
            sol_list = list(solutions)
            nb_solutions = len(sol_list)
            if 1 == nb_solutions:
                self.print_one_solution(sol_list[0], out, **kwargs)
            else:
                self.print_many_solutions(sol_list, out, **kwargs)

    def print_to_stream(self, solutions, out, **kwargs):
        if out is None:
            # prints on standard output
            self._print_to_stream2(stdout, solutions, **kwargs)
        elif isinstance(out, str):
            # a string is interpreted as a path name
            extension = self.extension()
            path = out if out.endswith(extension) else out + extension
            with open(path, "w") as of:
                self.print_to_stream(solutions, of, **kwargs)
                # print("* file: %s overwritten" % path)
        else:
            self._print_to_stream2(out, solutions, **kwargs)

    def print_to_string(self, solutions, **kwargs):
        from io import StringIO
        with StringIO() as oss:
            self.print_to_stream(solutions, out=oss, **kwargs)
            return oss.getvalue()
