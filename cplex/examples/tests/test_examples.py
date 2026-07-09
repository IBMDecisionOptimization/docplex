#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pytest suite to ensure all examples in src execute without exceptions.

This test suite runs each example as a subprocess with appropriate arguments and data files.
"""
import subprocess
import sys
import pytest
from pathlib import Path

# Directories
PROJECT_DIR = Path(__file__).parent.parent
SCRIPTS_DIR = PROJECT_DIR / "src"
DATA_DIR = PROJECT_DIR / "data"
TEST_OUTPUT_DIR = PROJECT_DIR / "build" / "test-output"


def run_example(script_name, args=None):
    """
    Run an example script as a subprocess.
    
    Args:
        script_name: Name of the Python script (e.g., 'admipex1.py')
        args: List of command-line arguments to pass to the script
    
    Raises:
        subprocess.CalledProcessError: If the script exits with non-zero status
    """
    script_path = SCRIPTS_DIR / script_name
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)

    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300,  # 5 minute timeout
        cwd=str(TEST_OUTPUT_DIR),
    )
    
    if result.returncode != 0:
        print(f"\n--- STDOUT ---\n{result.stdout}")
        print(f"\n--- STDERR ---\n{result.stderr}")
        raise subprocess.CalledProcessError(
            result.returncode, cmd, result.stdout, result.stderr
        )
    
    return result


class TestExamples:
    """Test suite for all CPLEX examples."""

    def test_admipex1(self):
        """Test admipex1 with noswot.mps file."""
        filename = str(DATA_DIR / "p0033.mps")
        run_example("admipex1.py", [filename])

    def test_admipex2(self):
        """Test admipex2 with noswot.mps file."""
        filename = str(DATA_DIR / "mexample.mps")
        run_example("admipex2.py", [filename])

    def test_admipex3(self):
        """Test admipex3 with sosex3.lp file."""
        filename = str(DATA_DIR / "sosex3.lp")
        run_example("admipex3.py", [filename])

    def test_admipex4(self):
        """Test admipex4 with data directory."""
        datadir = str(DATA_DIR)
        run_example("admipex4.py", [datadir])

    def test_admipex5(self):
        """Test admipex5 with facility data."""
        run_example("admipex5.py", [f'-data={DATA_DIR}'])

    def test_admipex6(self):
        """Test admipex6 with noswot.mps file."""
        filename = str(DATA_DIR / "mexample.mps.gz")
        run_example("admipex6.py", [filename])

    def test_admipex8(self):
        """Test admipex8 with facility data."""
        datadir = str(DATA_DIR)
        run_example("admipex8.py", [f'-data={DATA_DIR}'])

    def test_admipex9(self):
        """Test admipex9 with noswot.mps file."""
        filename = str(DATA_DIR / "noswot.mps")
        run_example("admipex9.py", [filename])

    def test_benders(self):
        """Test benders with example.mps and UFL annotation."""
        filename = str(DATA_DIR / "UFL_25_35_1.mps")
        annofile = str(DATA_DIR / "UFL_25_35_1.ann")
        run_example("benders.py", [filename, annofile])

    def test_bendersatsp(self):
        """Test bendersatsp with atsp data."""
        filename = str(DATA_DIR / "atsp.dat")
        run_example("bendersatsp.py", ["0", filename])

    def test_bendersatsp2(self):
        """Test bendersatsp2 with atsp data."""
        filename = str(DATA_DIR / "atsp.dat")
        run_example("bendersatsp2.py", ["0", filename])

    def test_blend(self):
        """Test blend example."""
        run_example("blend.py")

    def test_conflictex1(self):
        """Test conflictex1 with infeasible model."""
        filename = str(DATA_DIR / "infeasible.lp")
        run_example("conflictex1.py", [filename])

    def test_cutstock(self):
        """Test cutstock with cutstock data."""
        datafile = str(DATA_DIR / "cutstock.dat")
        run_example("cutstock.py", [datafile])

    def test_diet(self):
        """Test diet example with diet data."""
        filename = str(DATA_DIR / "diet.dat")
        run_example("diet.py", [filename])

    def test_diet_bycolumn(self):
        """Test diet example by column."""
        filename = str(DATA_DIR / "diet.dat")
        run_example("diet.py", ["-c", filename])

    def test_diet_integral(self):
        """Test diet example with integral variables."""
        filename = str(DATA_DIR / "diet.dat")
        run_example("diet.py", ["-i", filename])

    def test_etsp(self):
        """Test etsp with etsp data."""
        datafile = str(DATA_DIR / "etsp.dat")
        run_example("etsp.py", [datafile])

    def test_facility(self):
        """Test facility with facility data."""
        datafile = str(DATA_DIR / "facility.dat")
        run_example("facility.py", [datafile])

    def test_fixcost1(self):
        """Test fixcost1 example."""
        run_example("fixcost1.py")

    def test_fixnet(self):
        """Test fixnet example."""
        run_example("fixnet.py")

    def test_foodmanu(self):
        """Test foodmanu example."""
        run_example("foodmanu.py")

    def test_genericbranch(self):
        """Test genericbranch with noswot model."""
        model = str(DATA_DIR / "noswot.mps")
        run_example("genericbranch.py", [model])

    def test_globalqpex1(self):
        """Test globalqpex1 with nonconvex QP."""
        filename = str(DATA_DIR / "nonconvexqp.lp")
        run_example("globalqpex1.py", [filename, "g"])

    def test_indefqpex1(self):
        """Test indefqpex1 example."""
        run_example("indefqpex1.py")

    def test_inout1(self):
        """Test inout1 example."""
        run_example("inout1.py")

    def test_inout3(self):
        """Test inout3 example."""
        run_example("inout3.py")

    def test_lpex1_row(self):
        """Test lpex1 with row generation."""
        run_example("lpex1.py", ["-r"])

    def test_lpex1_column(self):
        """Test lpex1 with column generation."""
        run_example("lpex1.py", ["-c"])

    def test_lpex1_nonzero(self):
        """Test lpex1 with nonzero generation."""
        run_example("lpex1.py", ["-n"])

    def test_lpex1_copylp(self):
        """Test lpex1 with copylp method."""
        run_example("lpex1.py", ["-l"])

    def test_lpex2(self):
        """Test lpex2 with example.mps."""
        filename = str(DATA_DIR / "example.mps")
        run_example("lpex2.py", [filename, "o"])

    def test_lpex3(self):
        """Test lpex3 example."""
        run_example("lpex3.py")

    def test_lpex4(self):
        """Test lpex4 example."""
        run_example("lpex4.py")

    def test_lpex5(self):
        """Test lpex5 example."""
        run_example("lpex5.py")

    def test_lpex6(self):
        """Test lpex6 example."""
        run_example("lpex6.py")

    def test_lpex7(self):
        """Test lpex7 with example.mps."""
        filename = str(DATA_DIR / "example.mps")
        run_example("lpex7.py", [filename, "o"])

    def test_mipex1_row(self):
        """Test mipex1 with row generation."""
        run_example("mipex1.py", ["-r"])

    def test_mipex1_column(self):
        """Test mipex1 with column generation."""
        run_example("mipex1.py", ["-c"])

    def test_mipex2(self):
        """Test mipex2 with example.mps."""
        filename = str(DATA_DIR / "example.mps")
        run_example("mipex2.py", [filename])

    def test_mipex3(self):
        """Test mipex3 example."""
        run_example("mipex3.py")

    def test_mipex4(self):
        """Test mipex4 with noswot.mps."""
        filename = str(DATA_DIR / "noswot.mps")
        run_example("mipex4.py", [filename, "t"])

    def test_miqpex1(self):
        """Test miqpex1 example."""
        run_example("miqpex1.py")

    def test_multiobjex1(self):
        """Test multiobjex1 with multiobj model."""
        filename = str(DATA_DIR / "multiobj.lp")
        run_example("multiobjex1.py", [filename])

    def test_populate(self):
        """Test populate with location model."""
        filename = str(DATA_DIR / "location.lp")
        run_example("populate.py", [filename])

    def test_qcpdual(self):
        """Test qcpdual example."""
        run_example("qcpdual.py")

    def test_qcpex1(self):
        """Test qcpex1 example."""
        run_example("qcpex1.py")

    def test_qpex1(self):
        """Test qpex1 example."""
        run_example("qpex1.py")

    def test_qpex2(self):
        """Test qpex2 with qpex model."""
        filename = str(DATA_DIR / "qpex.lp")
        run_example("qpex2.py", [filename, "o"])

    def test_rates(self):
        """Test rates with rates data."""
        datafile = str(DATA_DIR / "rates.dat")
        run_example("rates.py", [datafile])

    def test_socpex1(self):
        """Test socpex1 example."""
        run_example("socpex1.py")

    def test_steel(self):
        """Test steel example."""
        run_example("steel.py")

    def test_transport_convex(self):
        """Test transport with convex piecewise linear."""
        run_example("transport.py", ["1"])

    def test_transport_nonconvex(self):
        """Test transport with non-convex piecewise linear."""
        run_example("transport.py", ["0"])

    def test_warehouse(self):
        """Test warehouse example."""
        run_example("warehouse.py")


if __name__ == "__main__":
    # Run pytest with verbose output
    pytest.main([__file__, "-v", "-s"])

# Made with Bob
