# --------------------------------------------------------------------------
# File: cli.py
# ---------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2023, 2024. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# --------------------------------------------------------------------------
import sys
import os
import distutils.ccompiler
import subprocess
import shutil
import glob
import argparse



def get_cos_archname(bindir, so):
    arch = [a for a in glob.glob("{}/**/{}".format(bindir, so), recursive=True)]
    if len(arch) == 0:
        return None
    elif len(arch) > 1:
        print("Error: multiple source files found: {}".format(arch))
        return None
    return os.path.basename(os.path.dirname(arch[0]))


def get_cpo_name(version):
    return "cpoptimizer.exe" if sys.platform == "win32" else "cpoptimizer"


def get_so_name(version):
    ext = distutils.ccompiler.new_compiler().shared_lib_extension
    if sys.platform == "darwin" : ext = ".dylib"
    prefix = "" if sys.platform == "win32" else "lib"
    return "{}cplex{}{}".format(prefix, version, ext)


def check_file(fname, write=False):
    if write:
        ok = os.access(fname, os.W_OK)
        if not ok:
            print("Error: {} should be present and writable but is not".format(fname))
    else:
        ok = os.access(fname, os.R_OK)
        if not ok:
            print("Error: {} should be present and readable but is not".format(fname))
    return ok


def get_cplex_info():
    sub_program = """
try:
    import cplex
    print(cplex.__path__[0], cplex.__version__)
except ModuleNotFoundError as e:
    print('Error: could not import module \"cplex\"')

    """

    # This needs to be run as another process as there is no other
    # way of unloading the 'cplex' module and since we will copy
    # a new shared object over the already loaded one, leaving the
    # cplex module loaded can create a segmentation fault.
    out = subprocess.run(["python", "-c", sub_program], capture_output=True)
    if out.returncode == 0:
        stdout = out.stdout.decode("utf-8").strip().split(" ")
        if stdout[0] != "Error:":
            return stdout[0], stdout[1]
    return None, None


def copy_so(cos):
    cos = os.path.realpath(cos)
    pcplex, version = get_cplex_info()

    version_mneumonic = "".join(version.split(".")[:3])
    so_name = get_so_name(version_mneumonic)
    cpo_name = get_cpo_name(version_mneumonic)

    target_bindir = os.path.dirname(sys.executable)
    target_root = os.path.dirname(target_bindir)

    so_targets = [t for t in glob.glob("{}/**/{}".format(target_root, so_name), recursive=True)]
    cpo_targets = [t for t in glob.glob("{}/**/{}".format(target_root, cpo_name), recursive=True)]
    if len(so_targets) == 0:
        print("ERROR: did not find shared object file {} in {}".format(so_name, target_root))
        return 1
    if len(cpo_targets) == 0:
        print("ERROR: did not find executable file {} in {}".format(cpo_name, target_root))
        return 1

    #
    # Find sources so_source, cpo_target
    #
    bindir = os.path.join(cos, "cplex", "bin")
    platform = get_cos_archname(bindir, so_name)
    if platform is None:
        print(
            "ERROR: unable to determine COS architecture mneumonic by searching for {} in {}. Please check your COS installation".format(
                so_name, bindir))
        return 1

    so_source = os.path.join(cos, "cplex", "bin", platform, so_name)
    cpo_source = os.path.join(cos, "cpoptimizer", "bin", platform, cpo_name)

    ok = check_file(so_source, False)
    ok = check_file(cpo_source, False) and ok
    for f in so_targets + cpo_targets:
        ok = check_file(f, True)

    if not ok:
        return 1

    #
    # Make copies
    #
    copies = tuple((so_source, t) for t in so_targets) + tuple((cpo_source, t) for t in cpo_targets)

    print("Performing copies:")
    code = 0
    try:
        for s, t in copies:
            print("    {} -> {}".format(s, t))
            shutil.copyfile(s, t)
    except EnvironmentError as e:
        print("ERROR: Could not upgrade packages due to an EnvironmentError: {}".format(e))
        print("Consider using the `--user` option or check the permissions.")
        code = e.code
    except Exception as e:
        print("Error: Something went wrong during copying: {}".format(e))
        code = 1

    return code


def config(args):
    if args.cos_root is not None:
        if not os.path.isdir(args.cos_root):
            print("ERROR: '{}' does not exist or is not a directory".format(args.cos_root))
            code = 1
        else:
            code = copy_so(args.cos_root)

        sys.exit(code)


def main():
    parser = argparse.ArgumentParser(prog="docplex")

    subparsers = parser.add_subparsers(dest="command", title="commands", metavar="<command>")

    parser_command1 = subparsers.add_parser("config", help="Manage configuration")
    parser_command1.add_argument("--upgrade", dest="cos_root", type=str,
                                 help="Upgrade this module from a Cplex Optimization Studio installation ",
                                 metavar="<cplex_studio_location>", required=True)
    parser_command1.set_defaults(func=config)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
    else:
        args.func(args)


if __name__ == "__main__":
    main()
