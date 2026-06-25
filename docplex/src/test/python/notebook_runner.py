import os
import json

try:
    # noinspection PyPackageRequirements
    from jupyter_client.manager import KernelManager

except ImportError:
    try:
        # noinspection PyPackageRequirements
        from IPython.kernel import KernelManager

    except ImportError:
        KernelManager = None


class KernelStartContext(object):
    def __init__(self, km, working_dir):
        self.km = km
        self.working_dir = working_dir
        self.saved_working_dir = os.getcwd()

    def __enter__(self):
        os.chdir(self.working_dir)
        self.km.start_kernel()
        kc = self.km.client()
        kc.start_channels()
        try:
            kc.wait_for_ready()
        except AttributeError:
            raise Exception("ERROR: this version of jupyter is not supported")
        return kc

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.km.shutdown_kernel()
        os.chdir(self.saved_working_dir)


class NoteBookRunner(object):
    ipython_extension = ".ipynb"

    def __init__(self, km=None):
        self.km = km

    @staticmethod
    def execute_cell(kc, text, cell_number):
        kc.execute(text)
        reply = kc.get_shell_msg()
        status = reply['content']['status']
        if status == 'error':

            traceback_text = '\n'.join(reply['content']['traceback'])
            msg = 'Cell {0} raised uncaught exception:\n{1}'.format(cell_number, traceback_text)
            print('RAW TRACEBACK_TEXT: ' + msg)
            ns = ''
            for c in traceback_text:
                if ord(c) == 10 or ord(c) == 13:
                    ns += c
                elif ord(c) < 31:
                    ns += '#'
                else:
                    ns += c
            print('UNCOLORED TRACEBACK_TEXT: ' + ns)

            raise Exception(ns)

    def execute_code(self, kc, cells, source_marker, store=False):
        count_run_cells = 0
        cell_count = 1
        cell_values = []
        for cell in cells:
            if "code" == cell["cell_type"]:
                source_code = "\n".join(cell[source_marker]) + "\n"
                ret = self.execute_cell(kc, source_code, cell_count)
                if store:
                    cell_values.append(ret)
                count_run_cells += 1
            cell_count += 1
        return cell_values

    def run(self, nb_path):
        if not KernelManager:
            raise ImportError

        if not nb_path.endswith(self.ipython_extension):
            nb_path += self.ipython_extension

        with open(nb_path) as nb_ifs:
            print('*running notebook: {0}'.format(nb_path))
            f = json.loads(nb_ifs.read())

            notebook_format = f['nbformat']
            if notebook_format == 4:
                cells = f['cells']
                code_marker = 'source'
            elif notebook_format == 3:
                cells = f['worksheets'][0]['cells']
                code_marker = 'input'
            else:
                raise ValueError("unsupported notebook format: {}".format(notebook_format))

            nb_dir = os.path.dirname(nb_path)
            km = self.km or KernelManager()
            with KernelStartContext(km, working_dir=nb_dir) as kc:
                self.execute_code(kc, cells, code_marker)

        return True


if __name__ == '__main__':
    nbr = NoteBookRunner()
    # where are the notebooks???
    wd = os.path.abspath(__file__)
    d = os.path.dirname
    delivered_nbs = os.path.join(d(wd), "../../../../docplex/src/samples/examples/delivery/jupyter")

    nbr.run(os.path.join(delivered_nbs, 'oil_blending.ipynb'))
