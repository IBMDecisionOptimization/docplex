# ---------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55 5655-Y21
# Copyright IBM Corporation 2013, 2026. All Rights Reserved.
#
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# ---------------------------------------------------------------------------
import re

import cplex


class OutputProcessorInfo():
    """Simple data object used by OutputProcessor."""
    # NB: Using __slots__ can save a lot of memory, but in this case,
    # it's commented out because there's no reason to optimize this.
    #__slots__ = ['regex_string', 'regex_pattern', 'num_matches']
    def __init__(self, regex_string):
        self.regex_string = regex_string
        self.regex_pattern = re.compile(regex_string)
        self.num_matches = 0


class OutputProcessor():
    """File-like object that processes output."""

    def __init__(self, regex_list):
        self.regex_list = [OutputProcessorInfo(r) for r in regex_list]

    def write(self, line):
        #print("OUTPUTPROCESSOR: {0}".format(line))
        for item in self.regex_list:
            pat = item.regex_pattern
            mat = pat.search(line)
            #print("SEARCHING FOR: {0}".format(item.regex_string))
            if mat:
                #print("FOUND: {0}".format(item.regex_string))
                item.num_matches += 1

    def flush(self):
        pass


def create_markshare1(cpx):
    """Create the markshare1.mps.gz model in Cplex instance cpx."""
    cname = ["s0", "s1", "s2", "s3", "s4", "s5",
             "x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9",
             "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19",
             "x20", "x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x29",
             "x30", "x31", "x32", "x33", "x34", "x35", "x36", "x37", "x38", "x39",
             "x40", "x41", "x42", "x43", "x44", "x45", "x46", "x47", "x48", "x49"]
    obj = [1.0] * 6 + [0.0] * 50
    x = list(cpx.variables.add(lb=[0] * len(cname),
                               ub=[1e20] * 6 + [1.0] * 50,
                               names=cname,
                               types=['C'] * 6 + ['B'] * 50))
    cpx.objective.set_linear((x[j], o) for j, o in enumerate(obj))

    rmatval = [+1.0, +25.0, +35.0, +14.0, +76.0, +58.0, +10.0, +20.0, +51.0,
               +58.0, +1.0, +35.0, +40.0, +65.0, +59.0, +24.0, +44.0, +1.0,
               +93.0, +24.0, +68.0, +38.0, +64.0, +93.0, +14.0, +83.0, +6.0,
               +58.0, +14.0, +71.0, +17.0, +18.0, +8.0, +57.0, +48.0, +35.0,
               +13.0, +47.0, +46.0, +8.0, +82.0, +51.0, +49.0, +85.0, +66.0,
               +45.0, +99.0, +21.0, +75.0, +78.0, +43.0, +1.0, +97.0, +64.0,
               +24.0, +63.0, +58.0, +45.0, +20.0, +71.0, +32.0, +7.0, +28.0,
               +77.0, +95.0, +96.0, +70.0, +22.0, +93.0, +32.0, +17.0, +56.0,
               +74.0, +62.0, +94.0, +9.0, +92.0, +90.0, +40.0, +45.0, +84.0,
               +62.0, +62.0, +34.0, +21.0, +2.0, +75.0, +42.0, +75.0, +29.0,
               +4.0, +64.0, +80.0, +17.0, +55.0, +73.0, +23.0, +13.0, +91.0,
               +70.0, +73.0, +28.0, +1.0, +95.0, +71.0, +19.0, +15.0, +66.0,
               +76.0, +4.0, +50.0, +50.0, +97.0, +83.0, +14.0, +27.0, +14.0,
               +34.0, +9.0, +99.0, +62.0, +92.0, +39.0, +56.0, +53.0, +91.0,
               +81.0, +46.0, +94.0, +76.0, +53.0, +58.0, +23.0, +15.0, +63.0,
               +2.0, +31.0, +55.0, +71.0, +97.0, +71.0, +55.0, +8.0, +57.0,
               +14.0, +76.0, +1.0, +46.0, +87.0, +22.0, +97.0, +99.0, +
               92.0, +1.0, +1.0, +27.0, +46.0, +48.0, +66.0, +58.0, +52.0,
               +6.0, +14.0, +26.0, +55.0, +61.0, +60.0, +3.0, +33.0, +99.0,
               +36.0, +55.0, +70.0, +73.0, +70.0, +38.0, +66.0, +39.0, +43.0,
               +63.0, +88.0, +47.0, +18.0, +73.0, +40.0, +91.0, +96.0, +49.0,
               +13.0, +27.0, +22.0, +71.0, +99.0, +66.0, +57.0, +1.0, +54.0,
               +35.0, +52.0, +66.0, +26.0, +1.0, +26.0, +12.0, +1.0, +3.0,
               +94.0, +51.0, +4.0, +25.0, +46.0, +30.0, +2.0, +89.0, +65.0,
               +28.0, +46.0, +36.0, +53.0, +30.0, +73.0, +37.0, +60.0, +21.0,
               +41.0, +2.0, +21.0, +93.0, +82.0, +16.0, +97.0, +75.0, +50.0,
               +13.0, +43.0, +45.0, +64.0, +78.0, +78.0, +6.0, +35.0, +72.0,
               +31.0, +28.0, +56.0, +60.0, +23.0, +70.0, +46.0, +88.0, +20.0,
               +69.0, +13.0, +40.0, +73.0, +1.0, +69.0, +72.0, +94.0, +56.0,
               +90.0, +20.0, +56.0, +50.0, +79.0, +59.0, +36.0, +24.0, +42.0,
               +9.0, +29.0, +68.0, +10.0, +1.0, +44.0, +74.0, +61.0, +37.0,
               +71.0, +63.0, +44.0, +77.0, +57.0, +46.0, +51.0, +43.0, +4.0,
               +85.0, +59.0, +7.0, +25.0, +46.0, +25.0, +70.0, +78.0, +88.0,
               +20.0, +40.0, +40.0, +16.0, +3.0, +3.0, +5.0, +77.0, +88.0,
               +16.0]
    rmatind = [0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
               22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
               37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
               52, 53, 54, 55, 1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
               18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
               34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
               50, 51, 52, 53, 54, 55, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
               16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
               32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
               48, 49, 50, 51, 52, 53, 54, 55, 3, 6, 7, 8, 9, 10, 11, 12, 13,
               14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
               30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
               46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 4, 6, 7, 8, 9, 10, 11,
               12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
               28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
               44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 5, 6, 7, 8, 9,
               10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
               26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
               42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55]
    rmatbeg = [0, 51, 102, 153, 204, 255]
    rhs = [+1116.0, +1325.0, +1353.0, +1169.0, +1160.0, +1163.0]
    sense = ['E', 'E', 'E', 'E', 'E', 'E']
    rname = ["c1", "c2", "c3", "c4", "c5", "c6"]
    for i in range(len(rhs)):
        end = rmatbeg[i + 1] if i < len(rhs) - 1 else len(rmatind)
        rng = range(rmatbeg[i], end)
        cpx.linear_constraints.add(
            lin_expr=[[[x[rmatind[j]] for j in rng],
                       [rmatval[j] for j in rng]]],
            senses=[sense[i]],
            rhs=[rhs[i]],
            names=[rname[i]])

def get_markshare1_optimal():
    """Get dense solution vector for optimal solution of markshare1.
    The objective value of this vector is 1"""
    return [1, 0, 0, 0, 0, 0,
            0, 1, 1, 0, 0, 1, 1, 0, 0, 0,
            1, 0, 0, 1, 0, 1, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 0, 0, 0, 1, 1,
            0, 1, 0, 0, 1, 1, 0, 1, 0, 1,
            1, 0, 0, 0, 1, 1, 1, 1, 0, 1]


def create_lpex1(cpx):
    """Builds the simple model from lpex1.py."""
    cpx.objective.set_sense(cpx.objective.sense.maximize)
    cpx.variables.add(obj=[1., 2., 3.],
                      ub=[40., cplex.infinity, cplex.infinity],
                      names=["x1", "x2", "x3"])
    cpx.linear_constraints.add(lin_expr=[[[0, 1, 2], [-1., 1., 1.]],
                                         [[0, 1, 2], [1., -3., 1.]]],
                               senses="LL",
                               rhs=[20., 30.],
                               names=["c1", "c2"])


def create_socpex1(cpx):
    """Builds the simple model from socpex1.py"""
    cpx.variables.add(obj=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                      lb=[0.0, -cplex.infinity, -cplex.infinity,
                          0.0, -cplex.infinity, 0.0],
                      names=["x1", "x2", "x3", "x4", "x5", "x6"])
    cpx.linear_constraints.add(lin_expr=[[[0, 1, 4], [1, 1, 1]],
                                         [[1, 4, 5], [1, 1, 1]]],
                               senses=['E', 'E'],
                               rhs=[8.0, 10.0],
                               names=["c1", "c2"])
    cpx.quadratic_constraints.add(quad_expr=[[0, 1, 2],
                                             [0, 1, 2],
                                             [-1.0, 1.0, 1.0]],
                                  sense='L',
                                  rhs=0.0,
                                  name="q1")
    cpx.quadratic_constraints.add(quad_expr=[[3, 4],
                                             [3, 4],
                                             [-1.0, 1.0]],
                                  sense='L',
                                  rhs=0.0,
                                  name="q2")
