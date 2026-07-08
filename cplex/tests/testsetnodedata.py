# ---------------------------------------------------------------------------
# Licensed Materials - Property of IBM
# 5725-A06 5725-A29 5724-Y48 5724-Y49 5724-Y54 5724-Y55
# Copyright IBM Corporation 2009, 2026. All Rights Reserved.
# 
# US Government Users Restricted Rights - Use, duplication or
# disclosure restricted by GSA ADP Schedule Contract with
# IBM Corp.
# ---------------------------------------------------------------------------
"""Test the set_node_data function of callbacks."""
import cplex  as CPX
import cplex.callbacks as CPX_CB
import sys
import threading
import random
import gc
import testutil


class UserData:
    # Static members and functions.
    instances = 0
    maxInstances = 0
    dogen = 0
    id2ser = dict()
    classLock = threading.Lock()
    @staticmethod
    def incInstances():
        with UserData.classLock:
            UserData.instances += 1
            UserData.maxInstances = max(UserData.maxInstances,
                                        UserData.instances)
    @staticmethod
    def decInstances():
        with UserData.classLock:
            UserData.instances -= 1
    @staticmethod
    def genData():
        with UserData.classLock:
            UserData.dogen += 1
            res = (UserData.dogen % 2) == 0
        return res
    @staticmethod
    def clear():
        with UserData.classLock:
            UserData.id2ser = dict()
    @staticmethod
    def reset():
        with UserData.classLock:
            UserData.instances = 0
            UserData.maxInstances = 0
            UserData.dogen = 0
            UserData.id2ser = dict()
    @staticmethod
    def getSerial(nodeid):
        with UserData.classLock:
            UserData.res = id2ser[nodeid]
        return res
    @staticmethod
    def setSerial(nodeid, serial):
        with UserData.classLock:
            UserData.id2ser[nodeid] = serial
    @staticmethod
    def check():
        with UserData.classLock:
            if UserData.maxInstances <= 0:
                raise Exception("No objects created")
            if UserData.instances > 0:
                raise Exception("Not all objects deleted ({0})".format(
                    UserData.instances))

    def __init__(self):
        self._disposed = False
        self.nodeid = None
        self.serial = 0
        UserData.incInstances()

    def end(self):
        if self._disposed:
            return
        self._disposed = True
        UserData.decInstances()

    def __del__(self):
        self.end()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end()

# Global lock
callbackLock = threading.Lock()

def check_node_data(cb):
    with callbackLock:
        if isinstance(cb, MyNodeCallback):
            num = random.randint(0, cb.get_num_remaining_nodes() - 1)
            udata = cb.get_node_data(num)
        else:
            udata = cb.get_node_data()
        newdata = None
        if udata:
            newdata = UserData()
            newdata.nodeid = udata.nodeid
            newdata.serial = udata.serial + 1
            UserData.setSerial(newdata.nodeid, newdata.serial)
        else:
            newdata = UserData()

        if isinstance(cb, MyNodeCallback):
            old = cb.set_node_data(num, newdata)
        else:
            old = cb.set_node_data(newdata)

        if old is not udata:
            raise Exception(
                "Unexpected old data {0} (expected {1})".format(old, udata))

        if old:
            # Pro-actively cleanup the UserData instance that is
            # no-longer used.
            old.end()

        if isinstance (cb, MyNodeCallback):
            old = cb.get_node_data(num)
        else:
            old = cb.get_node_data()

        if old is not newdata:
            raise Exception(
                "Unexpected new data {0} (expected {1})".format(old, newdata))

        if isinstance(cb, MyBranchCallback):
            for b in [cb.get_branch(x) for x in range(cb.get_num_branches())]:
                if UserData.genData():
                    udata = UserData()
                    udata.nodeid = cb.make_branch(b[0], variables=b[1],
                                                  node_data=udata)
                    UserData.setSerial(udata.nodeid, udata.serial)
                else:
                    cb.make_branch(b[0], variables=b[1])


class MyUserCutCallback(CPX_CB.UserCutCallback):

    def __call__(self):
        check_node_data(self)


class MyLazyConstraintCallback(CPX_CB.LazyConstraintCallback):

    def __call__(self):
        check_node_data(self)


class MyHeuristicCallback(CPX_CB.HeuristicCallback):

    def __call__(self):
        check_node_data(self)


class MySolveCallback(CPX_CB.SolveCallback):

    def __call__(self):
        check_node_data(self)


class MyIncumbentCallback(CPX_CB.IncumbentCallback):

    def __call__(self):
        check_node_data(self)


class MyBranchCallback(CPX_CB.BranchCallback):

    def __call__(self):
        check_node_data(self)


class MyNodeCallback(CPX_CB.NodeCallback):

    def __call__(self):
        check_node_data(self)


def main():
    for mode in (-1, 0, 1):
        for t in (0, 1, 2, 3):
            for clazz in (MyUserCutCallback,
                          MyLazyConstraintCallback,
                          MyHeuristicCallback,
                          MySolveCallback,
                          MyIncumbentCallback,
                          MyBranchCallback,
                          MyNodeCallback):
                print("#### TEST {0}, {1}, {2}".format(clazz, mode, t))
                UserData.reset()
                with CPX.Cplex() as c:
                    testutil.create_markshare1(c)

                    c.parameters.threads.set(t)
                    c.parameters.parallel.set(mode)
                    c.parameters.mip.limits.nodes.set(1000)
                    if clazz == MyBranchCallback:
                        reforms = c.parameters.preprocessing.reformulations.values.none
                    else:
                        reforms = c.parameters.preprocessing.reformulations.values.all                 
                    c.parameters.preprocessing.reformulations.set(reforms)
                    c.register_callback(clazz)
                    c.solve()

                    # Unregister the callbacks. This will clear out the
                    # tree and will delete all user data objects.
                    c.unregister_callback(clazz)

                # Also clear out the UserData.id2ser dictionary and run a full garbage
                # collection so that all unreferenced UserData objects are deleted.
                UserData.clear()
                gc.collect()

                # Now check that all UserData objects are deleted.
                UserData.check()
    print("Test passed")


if __name__ == "__main__":
    main()
