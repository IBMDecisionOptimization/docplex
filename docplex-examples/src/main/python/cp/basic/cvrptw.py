import sys
import os
import math
from collections import namedtuple
from docplex.cp.model import CpoModel, CpoParameters

import docplex.cp.solver.solver as solver
from docplex.cp.utils import compare_natural

solver_version = solver.get_version_info()['SolverVersion']
if compare_natural(solver_version, '22.1.1.0') < 0:
    print('Warning solver version', solver_version, 'is too old for', __file__)
    exit(0)
    
TIME_FACTOR = 10


class CVRPTWProblem:
    def __init__(self):
        self.nb_trucks = -1
        self.truck_capacity = -1
        self.max_horizon = -1
        self.nb_customers = -1
        self.depot_xy = None
        self.customers_xy = []
        self.demands = []
        self.earliest_start = []
        self.latest_start = []
        self.service_time = []
        self._xy = None

    def read_elem(self, filename):
        with open(filename) as f:
            return [str(elem) for elem in f.read().split()]

    # The input files follow the "Solomon" format.
    def read(self, filename):

        def skip_elems(n):
            for _ in range(n):
                next(file_it)

        file_it = iter(self.read_elem(filename))

        skip_elems(4)

        self.nb_trucks = int(next(file_it))
        self.truck_capacity = int(next(file_it))

        skip_elems(13)

        self.depot_xy = (int(next(file_it)), int(next(file_it)))

        skip_elems(2)

        self.max_horizon = int(next(file_it))

        skip_elems(1)

        idx = 0
        while True:
            val = next(file_it, None)
            if val is None: break
            idx = int(val) - 1
            self.customers_xy.append((int(next(file_it)), int(next(file_it))))
            self.demands.append(int(next(file_it)))
            ready = int(next(file_it))
            due = int(next(file_it))
            stime = int(next(file_it))
            self.earliest_start.append(ready)
            self.latest_start.append(due)
            self.service_time.append(stime)

        self.nb_customers = idx + 1
        self._xy = [self.depot_xy] + self.customers_xy

    def get_num_nodes(self): return self.nb_customers + 1

    def get_nb_trucks(self): return self.nb_trucks

    def get_capacity(self): return self.truck_capacity

    def get_max_horizon(self): return TIME_FACTOR * self.max_horizon

    def get_demand(self, i):
        assert i >= 0
        assert i < self.get_num_nodes()
        if i == 0:
            return 0
        return self.demands[i - 1]

    def get_service_time(self, i):
        assert i >= 0
        assert i < self.get_num_nodes()
        if i == 0:
            return 0
        return TIME_FACTOR * self.service_time[i - 1]

    def get_earliest_start(self, i):
        assert i >= 0
        assert i < self.get_num_nodes()
        if i == 0:
            return 0
        return TIME_FACTOR * self.earliest_start[i - 1]

    def get_latest_start(self, i):
        assert i >= 0
        assert i < self.get_num_nodes()
        if i == 0:
            return 0
        return TIME_FACTOR * self.latest_start[i - 1]

    def _get_distance(self, from_, to_):
        c1, c2 = self._xy[from_], self._xy[to_]
        dx, dy, d = c2[0] - c1[0], c2[1] - c1[1], 0.0
        d = math.sqrt(dx * dx + dy * dy)
        return int(math.floor(d * TIME_FACTOR))

    def get_distance(self, from_, to_):
        assert from_ >= 0
        assert from_ < self.get_num_nodes()
        assert to_ >= 0
        assert to_ < self.get_num_nodes()
        return self._get_distance(from_, to_)


class VRP:
    VisitData = namedtuple("CustomerData", "demand service_time earliest, latest")
    def __init__(self, pb):
        # Sizes
        self._num_veh = pb.get_nb_trucks()
        self._num_cust = pb.get_num_nodes() - 1
        self._n = self._num_cust + self._num_veh * 2

        # First, last, customer groups
        self._first = tuple(self._num_cust + i for i in range(self._num_veh))
        self._last = tuple(self._num_cust + self._num_veh + i for i in range(self._num_veh))
        self._cust = tuple(range(self._num_cust))

        # Time and load limits
        self._max_horizon = pb.get_max_horizon()
        self._capacity = pb.get_capacity()

        # Node mapping
        pnode = [i + 1 for i in range(self._num_cust)] + [0] * (2 * self._num_veh)

        # Visit data
        self._visit_data = \
            tuple(VRP.VisitData(pb.get_demand(pnode[c]), pb.get_service_time(pnode[c]), pb.get_earliest_start(pnode[c]), pb.get_latest_start(pnode[c])) for c in self._cust) + \
            tuple(VRP.VisitData(0, 0, 0, self._max_horizon) for _ in self._first + self._last)

        # Distance
        self._distance = [
          [ pb.get_distance(pnode[i], pnode[j]) for j in range(self._n) ]
          for i in range(self._n)
        ]

    def first(self) : return self._first
    def last(self) : return self._last
    def vehicles(self) : return zip(range(self._num_veh), self._first, self._last)
    def customers(self) : return self._cust
    def all(self) : return range(self._n)
    def get_num_customers(self): return self._num_cust
    def get_num_visits(self): return self._n
    def get_num_vehicles(self): return self._num_veh
    def get_first(self, veh): return self._first[veh]
    def get_last(self, veh): return self._last[veh]
    def get_capacity(self): return self._capacity
    def get_max_horizon(self): return self._max_horizon
    def get_demand(self, i): return self._visit_data[i].demand
    def get_service_time(self, i): return self._visit_data[i].service_time
    def get_earliest_start(self, i): return self._visit_data[i].earliest
    def get_latest_start(self, i): return self._visit_data[i].latest
    def get_distance(self, i, j): return self._distance[i][j]


class DataModel:
    vrp = None
    prev = None
    veh = None
    load = None
    start_time = None
    params = None


def build_model(cvrp_prob, tlim):
    data = DataModel()
    vrp = VRP(cvrp_prob)
    num_cust = vrp.get_num_customers()
    num_vehicles = vrp.get_num_vehicles()
    n = vrp.get_num_visits()

    mdl = CpoModel()

    # Prev variables, circuit, first/last
    prev = [mdl.integer_var(0, n - 1, "P{}".format(i)) for i in range(n)]
    for v,fv,lv in vrp.vehicles():
        mdl.add(prev[fv] == vrp.get_last((v - 1) % num_vehicles))

    before = vrp.customers() + vrp.first()
    for c in vrp.customers():
        mdl.add(mdl.allowed_assignments(prev[c], before))
        mdl.add(prev[c] != c)

    for _,fv,lv in vrp.vehicles():
        mdl.add(mdl.allowed_assignments(prev[lv], vrp.customers() + (fv,)))

    mdl.add(mdl.sub_circuit(prev))

    # Vehicle
    veh = [ mdl.integer_var(0, num_vehicles - 1, "V{}".format(i)) for i in range(n) ]
    for v,fv,lv in vrp.vehicles():
        mdl.add(veh[fv] == v)
        mdl.add(veh[lv] == v)
        mdl.add(mdl.element(veh, prev[lv]) == v)
    for c in vrp.customers():
        mdl.add(veh[c] == mdl.element(veh, prev[c]))

    # Demand
    load = [ mdl.integer_var(0, vrp.get_capacity(), "L{}".format(i)) for i in range(num_vehicles) ]
    used = mdl.integer_var(0, num_vehicles, 'U')
    cust_veh = [ veh[c] for c in vrp.customers() ]
    demand = [ vrp.get_demand(c) for c in vrp.customers() ]
    mdl.add(mdl.pack(load, cust_veh, demand, used))

    # Time
    start_time = [ mdl.integer_var(vrp.get_earliest_start(i), vrp.get_latest_start(i), "T{}".format(i)) for i in range(n) ]
    for fv in vrp.first():
        mdl.add(start_time[fv] == 0)
    for i in vrp.customers() + vrp.last():
        arrive = mdl.element([start_time[j] + vrp.get_service_time(j) + vrp.get_distance(j, i) for j in range(n)], prev[i])
        mdl.add(start_time[i] == mdl.max(arrive, vrp.get_earliest_start(i)))

    # Distance
    all_dist = []
    for i in vrp.customers() + vrp.last():
        ldist = [ vrp.get_distance(j, i) for j in range(n) ]
        all_dist.append(mdl.element(ldist, prev[i]))
    total_distance = mdl.sum(all_dist) / TIME_FACTOR

    # Variables with inferred values
    mdl.add(mdl.inferred(cust_veh + load + [used] + start_time))
    
    # Objective
    mdl.add(mdl.minimize(total_distance))

    # KPIs
    mdl.add_kpi(used, 'Used')

    # Solver params setting
    params = CpoParameters()
    params.SearchType = 'Restart'
    params.LogPeriod = 10000
    if tlim != None:
        params.TimeLimit = tlim

    mdl.set_parameters(params=params)

    data.vrp = vrp
    data.prev = prev
    data.veh = veh
    data.load = load
    data.start_time = start_time
    data.params = params

    return mdl, data


def display_solution(sol, data):
    vrp = data.vrp
    sprev = tuple(sol.solution[p] for p in data.prev)

    for v,fv,lv in vrp.vehicles():
        route = []
        nd = lv
        while nd != fv:
            route.append(nd)
            nd = sprev[nd]
        route.append(fv)
        route.reverse()
        print('Veh {} --->'.format(v, route), end="")
        if len(route) > 2:
            arrive = 0
            total_distance = 0
            total_load = 0
            line = ""
            for idx, nd in enumerate(route):
                early = vrp.get_earliest_start(nd)
                late = vrp.get_latest_start(nd)
                start = max(arrive, early)
                assert(start == sol.solution[data.start_time[nd]])
                line += " {} (a = {}, t = {} <= {} <= {})".format(nd, arrive, early, start, late)
                if nd != route[-1]:
                    nxt = route[idx + 1]
                    locald = vrp.get_distance(nd, nxt)
                    serv = vrp.get_service_time(nd)
                    line += " -- {} + {} -->".format(serv, locald)
                    arrive = start + serv + locald
                    total_distance += locald
                    if nd != route[0]:
                        total_load += data.vrp.get_demand(nd)
            line += " --- D = {:.1f}, L = {}".format(total_distance, total_load)
            print(line)
        else:
            print(" empty")


if __name__ == "__main__":
    fname = os.path.dirname(os.path.abspath(__file__)) + "/data/cvrptw_C101_25.data"
    if len(sys.argv) != 1:
        if len(sys.argv) != 2:
            print(f'Usage: {sys.argv[0]} OR {sys.argv[0]} <filename>')
            exit(1)
        else:
            fname = sys.argv[1]

    tlim = 15
    if len(sys.argv) == 3:
      tlim = float(sys.argv[2])
    cvrptw_prob = CVRPTWProblem()
    cvrptw_prob.read(fname)
    model, data_model = build_model(cvrptw_prob, tlim)
    solution = model.solve()
    if solution:
        display_solution(solution, data_model)
